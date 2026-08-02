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
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>  // ideal, nadir, fast_non_dominated_sorting


namespace{

/*  Relative tolerance used when deciding whether two extreme points coincide.
 */
constexpr double extreme_point_tol = 1e-12;

/*  Two extreme points are duplicates only when *every* coordinate matches. The
 *  comparison is relative to the magnitude of the coordinates, so that it does
 *  not depend on the scale of the objectives, with an absolute floor of tol for
 *  coordinates close to zero.
 */
bool close_vectors(const std::vector<double> &lhs, const std::vector<double> &rhs, double tol){
    if(lhs.size() != rhs.size()){
        return false;
    }
    for(size_t i=0; i<lhs.size(); i++){
        double scale = std::max(1.0, std::max(std::abs(lhs[i]), std::abs(rhs[i])));
        if(!(std::abs(lhs[i] - rhs[i]) <= tol*scale)){
            return false;
        }
    }
    return true;
}

}  // namespace


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

/*  The ideal point used to translate the objectives.
 *  When running_ideal is not null this is the best value found for each objective
 *  since the start of the run, as described in Deb & Jain Section IV.B, and it is
 *  updated in place so that it is retained across generations.
 */
std::vector<double> nsga3_compute_ideal(const std::vector<vector_double> &objs, std::vector<double> *running_ideal){
    std::vector<double> p_ideal = ideal(objs);

    if(running_ideal != nullptr){
        if(running_ideal->size() == p_ideal.size()){  // i.e. not first gen
            for(size_t i=0; i<p_ideal.size(); i++){
                p_ideal[i] = std::min(p_ideal[i], (*running_ideal)[i]);
            }
        }
        *running_ideal = p_ideal;
    }

    return p_ideal;
}


std::vector<std::vector<double>> nsga3_translate_objectives(const std::vector<vector_double> &objs,
                                                            const std::vector<double> &ideal_point){
    size_t NP = objs.size();
    size_t nobj = ideal_point.size();
    std::vector<std::vector<double>> translated_objs(NP, std::vector<double>(nobj));

    for(size_t obj=0; obj<nobj; obj++){
        for(size_t i=0; i<NP; i++){
            translated_objs[i][obj] = objs[i][obj] - ideal_point[obj];
        }
    }

    return translated_objs;
}

// fronts arg is NDS return type
std::vector<std::vector<double>> nsga3_find_extreme_points(const std::vector<std::vector<pop_size_t>> &fronts,
                                               const std::vector<std::vector<double>> &translated_objs,
                                               const std::vector<double> &ideal_point,
                                               std::vector<std::vector<double>> *retained_extremes){
    std::vector<std::vector<double>> points;
    size_t nobj = ideal_point.size();

    if(retained_extremes != nullptr && retained_extremes->size() != nobj){
        retained_extremes->assign(nobj, std::vector<double>{});
    }

    for(size_t i=0; i<nobj; i++){
        std::vector<double> weights(nobj, 1e-6);
        weights[i] = 1.0;
        double min_asf = std::numeric_limits<double>::max();
        std::vector<double> min_obj{};

        /*  Extreme points retained from previous generations are stored in the
         *  original objective coordinates: each must be translated by the
         *  *current* ideal point before it can be compared, on the same footing,
         *  with the candidates of this generation.
         */
        if(retained_extremes != nullptr){
            for(size_t p=0; p<retained_extremes->size(); p++){
                if((*retained_extremes)[p].size() != nobj){
                    continue;  // Nothing retained for this objective yet
                }
                std::vector<double> retained(nobj);
                for(size_t obj=0; obj<nobj; obj++){
                    retained[obj] = (*retained_extremes)[p][obj] - ideal_point[obj];
                }
                double asf = achievement(retained, weights);
                if(asf < min_asf){
                    min_asf = asf;
                    min_obj = retained;
                }
            }
        }

        // Only first front need be considered for extremes
        for(size_t ind=0; ind<fronts[0].size(); ind++){
            // Calculate ASF value for translated objectives
            double asf = achievement(translated_objs[fronts[0][ind]], weights);
            if(asf < min_asf){
                min_asf = asf;
                min_obj = translated_objs[fronts[0][ind]];
            }
        }
        if(min_obj.empty()){  // Only reachable if every ASF value was NaN
            min_obj = translated_objs[fronts[0][0]];
        }
        points.push_back(min_obj);
        if(retained_extremes != nullptr){
            // Retain in the original coordinates, so a moving ideal point does not invalidate it
            std::vector<double> original(nobj);
            for(size_t obj=0; obj<nobj; obj++){
                original[obj] = min_obj[obj] + ideal_point[obj];
            }
            (*retained_extremes)[i] = original;
        }
    }

    return points;
}

std::vector<double> nsga3_find_intercepts(const std::vector<std::vector<double>> &ext_points,
                                          const std::vector<std::vector<double>> &translated_objs){
    /*  1. Check duplicate extreme points
     *  2. A = translated objectives of extreme points;  b = [1,1,...] to n_objs
     *  3. Solve Ax = b via Gaussian Elimination
     *  4. Return reciprocals as intercepts
     *  NB Duplicate ext_points, a singular system and non-positive solutions
     *  all fall back to the nadir point. Both the extreme points and the
     *  returned intercepts are expressed in the translated coordinate system.
     */

    size_t n_obj = ext_points.size();
    std::vector<double> intercepts(n_obj, 1.0);
    bool fallback_to_nadir = false;

    for(size_t p=0; !fallback_to_nadir && p<n_obj; p++){
        if(ext_points[p].size() != n_obj){
            fallback_to_nadir = true;
            break;
        }
        for(size_t q=p+1; !fallback_to_nadir && q<n_obj; q++){
            // Extreme points coincide only when the *complete* vectors match
            fallback_to_nadir = close_vectors(ext_points[p], ext_points[q], extreme_point_tol);
        }
    }

    if(!fallback_to_nadir){
        std::vector<double> b(n_obj, 1.0);

        // Ax = b
        std::optional<vector_double> x = gaussian_elimination(ext_points, b);

        if(x.has_value()){
            // Express as intercepts, 1/x
            for(size_t i=0; i<n_obj; i++){
                // A zero, negative or non-finite coefficient has no usable reciprocal
                if(!std::isfinite((*x)[i]) || (*x)[i] <= 0.0){
                    fallback_to_nadir = true;
                    break;
                }
                intercepts[i] = 1.0/(*x)[i];
            }
        }else{
            fallback_to_nadir = true;  // Singular, or numerically singular, system
        }
    }

    if(fallback_to_nadir){
        /*  Translation by a constant vector preserves the dominance relation, so
         *  the nadir point of the translated objectives is exactly (worst - ideal):
         *  the same coordinate system as the objectives these intercepts divide.
         */
        std::vector<double> v_nadir = nadir(translated_objs);
        for(size_t i=0; i<n_obj && i<v_nadir.size(); i++){
            intercepts[i] = v_nadir[i];
        }
    }

    /*  A degenerate objective, identical across the whole population, has zero
     *  extent. Its translated coordinate is zero everywhere, so dividing by one
     *  keeps it at zero instead of producing an infinity or a NaN.
     */
    for(size_t i=0; i<n_obj; i++){
        if(!std::isfinite(intercepts[i]) || intercepts[i] <= 0.0){
            intercepts[i] = 1.0;
        }
    }

    return intercepts;
}

std::vector<std::vector<double>> nsga3_normalize_objectives(const std::vector<std::vector<double>> &translated_objs,
                                                      const std::vector<double> &intercepts){
    /*  Algorithm 2, step 7 and Equation 4
     *  Note that Objectives and therefore intercepts
     *  are already translated by ideal point.
     */

    if(translated_objs.empty()){
        return {};
    }

    size_t nobj = translated_objs[0].size();
    std::vector<std::vector<double>> norm_objs(translated_objs.size(), std::vector<double>(nobj));

    for(size_t i=0; i<translated_objs.size(); i++){
        for(size_t idx=0; idx<nobj; idx++){
            double intercept_or_eps = std::max(intercepts[idx], std::numeric_limits<double>::epsilon());
            norm_objs[i][idx] = translated_objs[i][idx]/intercept_or_eps;
        }
    }

    return norm_objs;
}

/*  Selects members of a population for survival into the next generation
 *  arguments:
 *    objs:  The objective vectors of the combined parent and offspring
 *           populations, of size 2*N_pop
 *    N_pop: The target population size to return
 *
 */
std::vector<size_t> nsga3_selection(const std::vector<vector_double> &objs, size_t N_pop, size_t divisions,
                                    std::vector<double> *running_ideal,
                                    std::vector<std::vector<double>> *retained_extremes,
                                    random_engine_type &reng){

    std::vector<size_t> next;
    next.reserve(N_pop);
    size_t last_front = 0;
    size_t next_size = 0;
    size_t nobj = objs[0].size();

    fnds_return_type nds = fast_non_dominated_sorting(objs);
    auto fronts = std::get<0>(nds);

    while(next_size < N_pop){
        next_size += fronts[last_front++].size();
    }
    fronts.erase(fronts.begin() + static_cast<std::vector<vector_double>::difference_type>(last_front), fronts.end());

    // Accept all members of first l-1 fronts
    for(size_t f=0; f<fronts.size()-1; f++){
        for(size_t i=0; i<fronts[f].size(); i++){
            next.push_back(fronts[f][i]);
        }
    }

    if(next.size() == N_pop){
        return next;
    }

    /*  A null memory pointer means the corresponding quantity is recomputed from
     *  scratch every generation instead of being retained across them.
     */
    auto ideal_point = nsga3_compute_ideal(objs, running_ideal);
    auto translated_objectives = nsga3_translate_objectives(objs, ideal_point);
    auto ext_points = nsga3_find_extreme_points(fronts, translated_objectives, ideal_point, retained_extremes);
    auto intercepts = nsga3_find_intercepts(ext_points, translated_objectives);
    auto norm_objs = nsga3_normalize_objectives(translated_objectives, intercepts);
    std::vector<reference_point> rps = generate_uniform_reference_points(nobj, divisions);
    associate_with_reference_points(rps, norm_objs, fronts);

    // Apply RP selection to final front until N_pop reached
    while(next.size() < N_pop){
        size_t min_rp_idx = identify_niche_point(rps, reng);
        std::optional<size_t> selected_idx = rps[min_rp_idx].select_member(reng);
        if(selected_idx.has_value()){
            rps[min_rp_idx].increment_members();
            rps[min_rp_idx].remove_candidate(selected_idx.value());
            next.push_back(selected_idx.value());
        }else{
            rps.erase(rps.begin() + static_cast<std::vector<vector_double>::difference_type>(min_rp_idx));
        }
    }

    return next;
}

} // namespace detail

} // namespace pagmo
