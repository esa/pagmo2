/* Copyright 2017-2021 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>


namespace pagmo{

namespace detail{

/// Gaussian Elimination
/**
 * Solves the linear system Ax = b by performing Gaussian elimination with partial
 * pivoting on an augmented matrix A|b.
 *
 * A singular, or numerically singular, system is not an error: it is reported by
 * returning an empty optional so that callers may apply their own fallback. The
 * singularity threshold scales with the magnitude of the entries of A, so that the
 * test is invariant under a uniform rescaling of the system.
 *
 * @param A the square coefficient matrix, taken by value as it is modified in place
 * @param b the right hand side vector
 * @return the solution x, or std::nullopt if A is (numerically) singular
 * @throws std::invalid_argument if A is empty or not square, or if the dimension of
 * b does not match that of A
 */
std::optional<vector_double> gaussian_elimination(std::vector<std::vector<double>> A, const vector_double &b){
    // Validate dimensions
    const size_t N = A.size();
    if(N == 0u){
        pagmo_throw(std::invalid_argument, "The coefficient matrix of a linear system cannot be empty");
    }
    for(size_t i=0; i<N; i++){
        if(A[i].size() != N){
            pagmo_throw(std::invalid_argument, "The coefficient matrix of a linear system must be square, while row "
                                               + std::to_string(i) + " has " + std::to_string(A[i].size())
                                               + " entries and the matrix has " + std::to_string(N) + " rows");
        }
    }
    if(b.size() != N){
        pagmo_throw(std::invalid_argument, "The right hand side of a linear system must match the dimension of the "
                                           "coefficient matrix, while a vector of size " + std::to_string(b.size())
                                           + " was detected for a matrix of size " + std::to_string(N));
    }

    /*  Singularity is declared when a pivot cannot be distinguished from zero at the
     *  scale of the matrix entries. The floor keeps an all-zero matrix singular.
     */
    double max_entry = 0.0;
    for(size_t i=0; i<N; i++){
        for(size_t j=0; j<N; j++){
            if(!std::isfinite(A[i][j])){
                return std::nullopt;
            }
            max_entry = std::max(max_entry, std::abs(A[i][j]));
        }
    }
    const double tol = std::max(std::numeric_limits<double>::epsilon()*static_cast<double>(N)*max_entry,
                                std::numeric_limits<double>::min());

    // Build augmented matrix
    for(size_t i=0; i<N; i++){
        A[i].push_back(b[i]);
    }

    // Eliminate subordinate entries, selecting the largest available pivot in each column
    for(size_t p=0; p<N; p++){
        size_t pivot_row = p;
        for(size_t i=p+1; i<N; i++){
            if(std::abs(A[i][p]) > std::abs(A[pivot_row][p])){
                pivot_row = i;
            }
        }
        if(std::abs(A[pivot_row][p]) <= tol){
            return std::nullopt;
        }
        std::swap(A[p], A[pivot_row]);
        for(size_t i=p+1; i<N; i++){
            double quot = A[i][p]/A[p][p];
            for(size_t j=p; j<A[p].size(); j++){
                A[i][j] -= A[p][j]*quot;
            }
        }
    }

    // Back substitution
    vector_double x(N);
    size_t i = N-1;
    while(true){
        for(size_t var=i+1; var<N; var++){
            A[i][N] -= A[i][var]*x[var];
        }
        x[i] = A[i][N]/A[i][i];
        if(!std::isfinite(x[i])){
            return std::nullopt;
        }
        if(i == 0){
            break;
        }
        i--;
    }

    return x;
}


/// Achievement Scalarization Function
/**
 * Computes the weighted Chebyshev scalarization max_i(objs[i]/weights[i]) used by
 * NSGA-III to identify the extreme point of a front along a given axis direction.
 * Weights below a small floor are replaced by that floor, so that a zero weight does
 * not produce a division by zero.
 *
 * @param objs the (translated) objective vector
 * @param weights the weight vector, of the same dimension as objs
 * @return the largest weighted ratio over all the components
 */
double achievement(const vector_double &objs, const vector_double &weights){
    const double default_weight = 1e-5;
    double max_ratio = -std::numeric_limits<double>::max();
    double w = 0.0;

    for(size_t i=0; i<objs.size(); i++){
        w = weights[i] > default_weight ? weights[i] : default_weight;
        max_ratio = std::max(max_ratio, objs[i]/w);
    }

    return max_ratio;
}

/// Distance from objective point to perpendicular intersection with reference point vector
/**
 * Computes the distance from obj_point to its orthogonal projection onto the line
 * spanned by ref_point. This is the association measure of NSGA-III: each individual
 * is associated to the reference point whose line it lies closest to.
 *
 * @param ref_point the reference point defining the direction
 * @param obj_point the (normalized) objective vector
 * @return the perpendicular distance between obj_point and the ref_point direction
 */
double perpendicular_distance(const std::vector<double> &ref_point, const std::vector<double> &obj_point){
    double num = 0.0, denom = 0.0, sq_dist = 0.0;
    for(size_t i=0; i<ref_point.size(); i++){
        num += ref_point[i]*obj_point[i];
        denom += ref_point[i]*ref_point[i];
    }
    double coeff = num/denom;
    for(size_t i=0; i<ref_point.size(); i++){
        double term = coeff*ref_point[i] - obj_point[i];
        sq_dist += term*term;
    }
    return std::sqrt(sq_dist);
}

} // namespace detail

} // namespace pagmo
