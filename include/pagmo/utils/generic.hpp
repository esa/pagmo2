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

#ifndef PAGMO_UTILS_GENERIC_HPP
#define PAGMO_UTILS_GENERIC_HPP

/** \file generic.hpp
 * \brief Utilities of general interest
 *
 * This header contains utilities useful in general for PaGMO purposes
 */

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Generates a random number within some lower and upper bounds
/**
 * Creates a random number within a closed range. If
 * both the lower and upper bounds are finite numbers, then the generated value
 * \f$ x \f$ will be such that \f$lb \le x < ub\f$. If \f$lb == ub\f$ then \f$lb\f$ is
 * returned.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    This helper function has to be preferred to ``std::uniform_real<double>(r_engine)`` as it
 *    also performs additional checks avoiding undefined behaviour in pagmo.
 *
 * \endverbatim
 *
 * Example:
 *
 * @code{.unparsed}
 * std::mt19937 r_engine(32u);
 * auto x = uniform_real_from_range(3,5,r_engine); // a random value
 * auto x = uniform_real_from_range(2,2,r_engine); // the value 2.
 * @endcode
 *
 * @param lb lower bound
 * @param ub upper bound
 * @param r_engine a <tt>std::mt19937</tt> random engine
 *
 * @throws std::invalid_argument if:
 * - the bounds contain NaNs or infs,
 *   or \f$ lb > ub \f$,
 * - if \f$ub-lb\f$ is larger than implementation-defined value
 *
 * @returns a random floating-point value
 */
inline double uniform_real_from_range(double lb, double ub, detail::random_engine_type &r_engine)
{
    // NOTE: see here for the requirements for floating-point RNGS:
    // http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution/uniform_real_distribution

    // 0 - Forbid random generation when bounds are not finite.
    if (!std::isfinite(lb) || !std::isfinite(ub)) {
        pagmo_throw(std::invalid_argument, "Cannot generate a random point if the bounds are not finite");
    }
    // 1 - Check that lb is <= ub
    if (lb > ub) {
        pagmo_throw(std::invalid_argument,
                    "Lower bound is greater than upper bound. Cannot generate a random point in [lb, ub]");
    }
    // 2 - Bounds cannot be too large
    const auto delta = ub - lb;
    if (!std::isfinite(delta) || delta > std::numeric_limits<double>::max()) {
        pagmo_throw(std::invalid_argument, "Cannot generate a random point within bounds that are too large");
    }
    // 3 - If the bounds are equal we don't call the RNG, as that would be undefined behaviour.
    if (lb == ub) {
        return lb;
    }
    return std::uniform_real_distribution<double>(lb, ub)(r_engine);
}

/// Generates a random decision vector
/**
 * Creates a random decision vector within some bounds. If
 * both the lower and upper bounds are finite numbers, then the \f$i\f$-th
 * component of the randomly generated pagmo::vector_double will be such that
 * \f$lb_i \le x_i < ub_i\f$. If \f$lb_i == ub_i\f$ then \f$lb_i\f$ is
 * returned. If an integer part is specified then the corresponding components
 * are guaranteed to be integers.
 *
 * Example:
 *
 * @code{.unparsed}
 * std::mt19937 r_engine(32u);
 * auto x = random_decision_vector({{1,3},{3,5}}, r_engine); // a random vector
 * auto x = random_decision_vector({{1,3},{1,3}}, r_engine); // the vector {1,3}
 * auto x = random_decision_vector({{1,3},{1,5}}, r_engine, 1); // the vector {1,3} or {1,4} or {1,5}
 * @endcode
 *
 * @param bounds an <tt>std::pair</tt> containing the bounds
 * @param r_engine a <tt>std::mt19937</tt> random engine
 * @param nix size of the integer part
 *
 * @throws std::invalid_argument if:
 * - the bounds are not of equal length, they have zero size, they contain NaNs or infs,
 *   \f$ \mathbf{ub} < \mathbf {lb}\f$, the integer part is larger than the bounds size or
 *   the bounds of the integer part are not integers.
 * - if \f$ub_i-lb_i\f$ is larger than implementation-defined value
 *
 * @returns a pagmo::vector_double containing a random decision vector
 */
inline vector_double random_decision_vector(const std::pair<vector_double, vector_double> &bounds,
                                            detail::random_engine_type &r_engine, vector_double::size_type nix = 0u)
{
    // This will check for consistent vector lengths, non-null sizes, lb <= ub, no NaNs and consistency in
    // the integer part
    detail::check_problem_bounds(bounds, nix);
    auto nx = bounds.first.size();
    vector_double retval(nx);
    auto ncx = nx - nix;

    for (decltype(ncx) i = 0u; i < ncx; ++i) {
        retval[i] = uniform_real_from_range(bounds.first[i], bounds.second[i], r_engine);
    }
    for (auto i = ncx; i < nx; ++i) {
        // To ensure a uniform int distribution from a uniform float distribution we floor the result
        // adding 1 tot he upper bound so that e.g.
        // [0,1] -> draw a float from [0,2] and floor it (that is 0. or 1. with 50%)
        // [3,3] -> draw a float from [3,4] and floor it (that is always 3.)
        double lb = bounds.first[i], ub = bounds.second[i];
        auto tmp = uniform_real_from_range(lb, ub + 1, r_engine);
        retval[i] = std::floor(tmp);
    }
    return retval;
}

/// Generates a random decision vector
/**
 * Creates a random decision vector within some bounds. If
 * both the lower and upper bounds are finite numbers, then the \f$i\f$-th
 * component of the randomly generated pagmo::vector_double will be such that
 * \f$lb_i \le x_i < ub_i\f$. If \f$lb_i == ub_i\f$ then \f$lb_i\f$ is
 * returned. If an integer part is specified then the corresponding components
 * are guaranteed to be integers.
 *
 * Example:
 *
 * @code{.unparsed}
 * std::mt19937 r_engine(32u);
 * auto x = random_decision_vector({1,3},{3,5}, r_engine); // a random vector
 * auto x = random_decision_vector({1,3},{1,3}, r_engine); // the vector {1,3}
 * auto x = random_decision_vector({{1,3},{1,5}}, r_engine, 1); // the vector {1,3} or {1,4} or {1,5}
 * @endcode
 *
 * @param lb a vector_double containing the lower bounds
 * @param ub a vector_double containing the upper bounds
 * @param r_engine a <tt>std::mt19937</tt> random engine
 * @param nix size of the integer part
 *
 * @throws std::invalid_argument if:
 * - the bounds are not of equal length, they have zero size, they contain NaNs or infs,
 *   \f$ \mathbf{ub} < \mathbf {lb}\f$, the integer part is larger than the bounds size or
 *   the bounds of the integer part are not integers.
 * - if \f$ub_i-lb_i\f$ is larger than implementation-defined value
 *
 * @returns a pagmo::vector_double containing a random decision vector
 */
inline vector_double random_decision_vector(const vector_double &lb, const vector_double &ub,
                                            detail::random_engine_type &r_engine, vector_double::size_type nix = 0u)
{
    return random_decision_vector({lb, ub}, r_engine, nix);
}

/// Binomial coefficient
/**
 * An implementation of the binomial coefficient using gamma functions
 * @param  n first parameter \f$n\f$
 * @param  k second paramater \f$k\f$
 * @return the binomial coefficient \f$ n \choose k \f$
 */
inline double binomial_coefficient(vector_double::size_type n, vector_double::size_type k)
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
inline std::vector<std::vector<vector_double::size_type>> kNN(const std::vector<vector_double> &points,
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
// modifies a chromosome so that it will be in the bounds. elements that are off are resampled at random in the bounds
inline void force_bounds_random(vector_double &x, const vector_double &lb, const vector_double &ub,
                                detail::random_engine_type &r_engine)
{
    assert(x.size() == lb.size());
    assert(x.size() == ub.size());
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        if ((x[j] < lb[j]) || (x[j] > ub[j])) {
            x[j] = pagmo::uniform_real_from_range(lb[j], ub[j], r_engine);
        }
    }
}
// modifies a chromosome so that it will be in the bounds. Elements that are off are reflected in the bounds
inline void force_bounds_reflection(vector_double &x, const vector_double &lb, const vector_double &ub)
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
inline void force_bounds_stick(vector_double &x, const vector_double &lb, const vector_double &ub)
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

#endif
