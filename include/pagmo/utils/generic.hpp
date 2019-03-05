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

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

// Checks that all elements of the problem bounds are not equal
inline bool some_bound_is_equal(const problem &prob)
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

// Check that the lower/upper bounds lb/ub are suitable for the
// generation of a real number. The boolean flags specify at
// compile time which checks to run.
template <bool FiniteCheck, bool LbUbCheck, bool RangeCheck>
inline void uniform_real_from_range_checks(double lb, double ub)
{
    // NOTE: see here for the requirements for floating-point RNGS:
    // http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution/uniform_real_distribution

    // 0 - Forbid random generation when bounds are not finite.
    if (FiniteCheck) {
        if (!std::isfinite(lb) || !std::isfinite(ub)) {
            pagmo_throw(std::invalid_argument, "Cannot generate a random real if the bounds are not finite");
        }
    } else {
        assert(std::isfinite(lb) && std::isfinite(ub));
    }

    // 1 - Check that lb is <= ub
    if (LbUbCheck) {
        if (lb > ub) {
            pagmo_throw(std::invalid_argument,
                        "Cannot generate a random real if the lower bound is larger than the upper bound");
        }
    } else {
        assert(lb <= ub);
    }

    // 2 - Bounds cannot be too large
    if (RangeCheck) {
        const auto delta = ub - lb;
        if (!std::isfinite(delta) || delta > std::numeric_limits<double>::max()) {
            pagmo_throw(std::invalid_argument, "Cannot generate a random real within bounds that are too large");
        }
    } else {
        assert(std::isfinite(ub - lb) && (ub - lb) <= std::numeric_limits<double>::max());
    }
}

// Implementation of the uniform_real_from_range() function.
template <bool FiniteCheck, bool LbUbCheck, bool RangeCheck, typename Rng>
inline double uniform_real_from_range_impl(double lb, double ub, Rng &r_engine)
{
    // Run the checks on the bounds.
    uniform_real_from_range_checks<FiniteCheck, LbUbCheck, RangeCheck>(lb, ub);
    // If the bounds are equal we don't call the RNG, as that would be undefined behaviour.
    return (lb == ub) ? lb : std::uniform_real_distribution<double>(lb, ub)(r_engine);
}

// Check that the lower/upper bounds lb/ub are suitable for the
// generation of an integral number. The boolean flags specify at
// compile time which checks to run.
template <bool FiniteCheck, bool LbUbCheck, bool IntCheck>
inline void uniform_integral_from_range_checks(double lb, double ub)
{
    // 0 - Check for finite bounds.
    if (FiniteCheck) {
        if (!std::isfinite(lb) || !std::isfinite(ub)) {
            pagmo_throw(std::invalid_argument, "Cannot generate a random integer if the bounds are not finite");
        }
    } else {
        assert(std::isfinite(lb) && std::isfinite(ub));
    }

    // 1 - Check that lb is <= ub
    if (LbUbCheck) {
        if (lb > ub) {
            pagmo_throw(std::invalid_argument,
                        "Cannot generate a random integer if the lower bound is larger than the upper bound");
        }
    } else {
        assert(lb <= ub);
    }

    // 2 - Check that lb/ub are integral values.
    if (IntCheck) {
        if (std::trunc(lb) != lb || std::trunc(ub) != ub) {
            pagmo_throw(std::invalid_argument,
                        "Cannot generate a random integer if the lower/upper bounds are not integral values");
        }
    } else {
        assert(std::trunc(lb) == lb && std::trunc(ub) == ub);
    }
}

// Implementation of the uniform_integral_from_range() function.
template <bool FiniteCheck, bool LbUbCheck, bool IntCheck, typename Rng>
inline double uniform_integral_from_range_impl(double lb, double ub, Rng &r_engine)
{
    // Run the checks on the bounds.
    uniform_integral_from_range_checks<FiniteCheck, LbUbCheck, IntCheck>(lb, ub);
    // We will convert ub/lb to the widest signed integral type possible (long long),
    // do the generation using uniform_int_distribution, and finally convert back the
    // result to double precision.
    // NOTE: the use of numeric_cast ensures that the conversion to long long is checked
    // (in case of overflow, an exception will be thrown).
    const auto l = boost::numeric_cast<long long>(lb);
    const auto u = boost::numeric_cast<long long>(ub);
    // NOTE: it should be safe here to do a raw cast, as the result
    // will be within the original bounds and thus representable by double.
    return static_cast<double>(std::uniform_int_distribution<long long>(l, u)(r_engine));
}

} // namespace detail

/// Generate a random integral number within some lower and upper bounds
/**
 * This function will create a random integral number within a closed range. If
 * both the lower and upper bounds are finite numbers, then the generated value
 * \f$ x \f$ will be such that \f$lb \le x \le ub\f$.
 *
 * Example:
 *
 * @code{.unparsed}
 * std::mt19937 r_engine(32u);
 * auto x = uniform_integral_from_range(3,5,r_engine); // one of [3, 4, 5].
 * auto x = uniform_integral_from_range(2,2,r_engine); // the value 2.
 * @endcode
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The return value is created internally via an integral random number
 *    generator based on the ``long long`` type, and then cast back to ``double``.
 *    Thus, if the absolute values of the lower/upper bounds are large enough, any of
 *    the following may happen:
 *
 *    * the conversion of the lower/upper bounds to ``long long`` may produce an overflow error,
 *    * the conversion of the randomly-generated ``long long`` integer back to ``double`` may yield an
 *      inexact result.
 *
 *    In pratice, on modern mainstream computer architectures, this function will produce uniformly-distributed
 *    integral values as long as the absolute values of the bounds do not exceed :math:`2^{53}`.
 *
 * \endverbatim
 *
 * @param lb lower bound
 * @param ub upper bound
 * @param r_engine a C++ random engine
 *
 * @throws std::invalid_argument if:
 * - the bounds are not finite,
 * - \f$ lb > ub \f$,
 * - \f$ lb \f$ and/or \f$ ub \f$ are not integral values.
 * @throws unspecified any exception raised by <tt>boost::numeric_cast()</tt>.
 *
 * @returns a random integral value
 */
template <typename Rng>
inline double uniform_integral_from_range(double lb, double ub, Rng &r_engine)
{
    // Activate all checks on lb/ub.
    return detail::uniform_integral_from_range_impl<true, true, true>(lb, ub, r_engine);
}

/// Generate a random real number within some lower and upper bounds
/**
 * This function will create a random real number within a half-open range. If
 * both the lower and upper bounds are finite numbers, then the generated value
 * \f$ x \f$ will be such that \f$lb \le x < ub\f$. If \f$lb = ub\f$, then \f$lb\f$ is
 * returned.
 *
 * Example:
 *
 * @code{.unparsed}
 * std::mt19937 r_engine(32u);
 * auto x = uniform_real_from_range(3,5,r_engine); // a random real in the [3, 5) range.
 * auto x = uniform_real_from_range(2,2,r_engine); // the value 2.
 * @endcode
 *
 * @param lb lower bound
 * @param ub upper bound
 * @param r_engine a C++ random engine
 *
 * @throws std::invalid_argument if:
 * - the bounds are not finite,
 * - \f$ lb > ub \f$,
 * - \f$ ub - lb \f$ is larger than an implementation-defined value.
 *
 * @returns a random floating-point value
 */
template <typename Rng>
inline double uniform_real_from_range(double lb, double ub, Rng &r_engine)
{
    // Activate all checks on lb/ub.
    return detail::uniform_real_from_range_impl<true, true, true>(lb, ub, r_engine);
}

/// Generate a random decision vector compatible with a problem
/**
 * This function will generate a decision vector whose values
 * are randomly chosen with uniform probability within
 * the input problem's bounds.
 *
 * For the continuous part of the decision vector, the values will be
 * generated via pagmo::uniform_real_from_range().
 *
 * For the discrete part of the decision vector, the values will be generated
 * via pagmo::uniform_integral_from_range().
 *
 * @param prob the input pagmo::problem
 * @param r_engine a C++ random engine
 *
 * @throws unspecified any exception thrown by pagmo::uniform_real_from_range() or pagmo::uniform_integral_from_range().
 *
 * @returns a pagmo::vector_double containing a random decision vector
 */
template <typename Rng>
inline vector_double random_decision_vector(const problem &prob, Rng &r_engine)
{
    // Prepare the return value.
    vector_double out(prob.get_nx());

    // Fetch a few quantities from prob.
    const auto nx = prob.get_nx();
    const auto nix = prob.get_nix();
    const auto ncx = nx - nix;
    const auto &lb = prob.get_lb();
    const auto &ub = prob.get_ub();

    // Continuous part.
    for (vector_double::size_type i = 0u; i < ncx; ++i) {
        // NOTE: the lb<=ub check is not needed, as it is ensured by the problem class.
        // Still need to check for finiteness and range.
        out[i] = detail::uniform_real_from_range_impl<true, false, true>(lb[i], ub[i], r_engine);
    }

    // Integer part.
    for (auto i = ncx; i < nx; ++i) {
        // NOTE: the lb<=ub check and the check that lb/ub are integral values are not needed,
        // as they are ensured by the problem class.
        // Still need to check for finiteness.
        out[i] = detail::uniform_integral_from_range_impl<true, false, false>(lb[i], ub[i], r_engine);
    }

    return out;
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
