/* Copyright 2017-2018 PaGMO development team

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
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

namespace detail
{

// Checks that all elements of the problem bounds are not equal
bool some_bound_is_equal(const problem &prob)
{
    // Some variable renaming
    const auto &lb = prob.get_lb();
    const auto &ub = prob.get_ub();
    // Since the bounds are extracted from problem we can be sure they have equal length
    for (decltype(lb.size()) i = 0u; i < lb.size(); ++i) {
        if (lb[i] == ub[i]) {
            return true;
        }
    }
    return false;
}

} // namespace detail

/// Binomial coefficient
/**
 * An implementation of the binomial coefficient using gamma functions
 * @param  n first parameter \f$n\f$
 * @param  k second paramater \f$k\f$
 * @return the binomial coefficient \f$ n \choose k \f$
 */
double binomial_coefficient(vector_double::size_type n, vector_double::size_type k)
{
    if (k <= n) {
        return std::round(std::exp(std::lgamma(static_cast<double>(n) + 1.) - std::lgamma(static_cast<double>(k) + 1.)
                                   - std::lgamma(static_cast<double>(n) - static_cast<double>(k) + 1.)));
    } else {
        pagmo_throw(std::invalid_argument, "The binomial coefficient is only defined for k<=n, you requested n="
                                               + std::to_string(n) + " and k=" + std::to_string(k));
    }
}

/// K-Nearest Neighbours
/**
 * Computes the indexes of the k nearest neighbours (euclidean distance) to each of the input points.
 * The algorithm complexity (naive implementation) is \f$ O(MN^2)\f$ where \f$N\f$ is the number of
 * points and \f$M\f$ their dimensionality
 *
 * Example:
 * @code{.unparsed}
 * auto res = kNN({{1, 1}, {2, 2}, {3.1, 3.1}, {5, 5}}, 2u);
 * @endcode
 *
 * @param points the \f$N\f$ points having dimension \f$M\f$
 * @param k number of neighbours to detect
 * @return An <tt>std::vector<std::vector<population::size_type> > </tt> containing the indexes of the k nearest
 * neighbours sorted by distance
 * @throws std::invalid_argument If the points do not all have the same dimension.
 */
std::vector<std::vector<vector_double::size_type>> kNN(const std::vector<vector_double> &points,
                                                       std::vector<vector_double>::size_type k)
{
    std::vector<std::vector<vector_double::size_type>> neigh_idxs;
    auto N = points.size();
    if (N == 0u) {
        return {};
    }
    auto M = points[0].size();
    if (!std::all_of(points.begin(), points.end(), [M](const vector_double &p) { return p.size() == M; })) {
        pagmo_throw(std::invalid_argument, "All points must have the same dimensionality for k-NN to be invoked");
    }
    // loop through the points
    for (decltype(N) i = 0u; i < N; ++i) {
        // We compute all the distances to all other points including the self
        vector_double distances;
        for (decltype(N) j = 0u; j < N; ++j) {
            double dist = 0.;
            for (decltype(M) l = 0u; l < M; ++l) {
                dist += (points[i][l] - points[j][l]) * (points[i][l] - points[j][l]);
            }
            distances.push_back(std::sqrt(dist));
        }
        // We sort the indexes with respect to the distance
        std::vector<vector_double::size_type> idxs(N);
        std::iota(idxs.begin(), idxs.end(), vector_double::size_type(0u));
        std::sort(idxs.begin(), idxs.end(), [&distances](vector_double::size_type idx1, vector_double::size_type idx2) {
            return detail::less_than_f(distances[idx1], distances[idx2]);
        });
        // We remove the first element containg the self-distance (0)
        idxs.erase(std::remove(idxs.begin(), idxs.end(), i), idxs.end());
        neigh_idxs.push_back(idxs);
    }
    // We trim to k the lists if needed
    if (k < N - 1u) {
        for (decltype(neigh_idxs.size()) i = 0u; i < neigh_idxs.size(); ++i) {
            neigh_idxs[i].erase(neigh_idxs[i].begin() + static_cast<int>(k),
                                neigh_idxs[i].end()); // TODO: remove the static_cast
        }
    }
    return neigh_idxs;
}

namespace detail
{

// modifies a chromosome so that it will be in the bounds. Elements that are off are reflected in the bounds
void force_bounds_reflection(vector_double &x, const vector_double &lb, const vector_double &ub)
{
    assert(x.size() == lb.size());
    assert(x.size() == ub.size());
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        while (x[j] < lb[j] || x[j] > ub[j]) {
            if (x[j] < lb[j]) {
                x[j] = 2 * lb[j] - x[j];
            }
            if (x[j] > ub[j]) {
                x[j] = 2 * ub[j] - x[j];
            }
        }
    }
}

// modifies a chromosome so that it will be in the bounds. Elements that are off are set on the bounds
void force_bounds_stick(vector_double &x, const vector_double &lb, const vector_double &ub)
{
    assert(x.size() == lb.size());
    assert(x.size() == ub.size());
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        if (x[j] < lb[j]) {
            x[j] = lb[j];
        }
        if (x[j] > ub[j]) {
            x[j] = ub[j];
        }
    }
}

} // namespace detail

} // namespace pagmo
