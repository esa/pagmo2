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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

// Checks that all elements of the problem bounds are not equal
PAGMO_DLL_PUBLIC bool some_bound_is_equal(const problem &);

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
    long long l, u;
    try {
        l = boost::numeric_cast<long long>(lb);
        u = boost::numeric_cast<long long>(ub);
    } catch (...) {
        pagmo_throw(std::invalid_argument, "Cannot generate a random integer if the lower/upper bounds are not within "
                                           "the bounds of the long long type");
    }
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
 * - \f$ lb \f$ and/or \f$ ub \f$ are not integral values,
 * - \f$ lb \f$ and/or \f$ ub \f$ are not within the bounds of the ``long long`` type.
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
    const auto ncx = nx - prob.get_nix();
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

/// Generate a batch of random decision vectors compatible with a problem
/**
 * This function will generate \p n decision vectors whose values
 * are randomly chosen with uniform probability within
 * the input problem's bounds. The decision vectors are laid
 * out contiguously in the return value: for a problem with dimension \f$ d \f$,
 * the first decision vector in the return value occupies
 * the index range \f$ \left[0, d\right) \f$, the second decision vector occupies the range
 * \f$ \left[d, 2d\right) \f$, and so on.
 *
 * For the continuous parts of the decision vectors, the values will be
 * generated via pagmo::uniform_real_from_range().
 *
 * For the discrete parts of the decision vectors, the values will be generated
 * via pagmo::uniform_integral_from_range().
 *
 * @param prob the input pagmo::problem
 * @param n how many decision vectors will be generated
 * @param r_engine a C++ random engine
 *
 * @throws std::overflow_error in case of (unlikely) overflow errors.
 * @throws unspecified any exception thrown by pagmo::uniform_real_from_range() or pagmo::uniform_integral_from_range().
 *
 * @returns a pagmo::vector_double containing \p n random decision vectors
 */
template <typename Rng>
inline vector_double batch_random_decision_vector(const problem &prob, vector_double::size_type n, Rng &r_engine)
{
    // NOTE: it is possible in principle to do this in a parallel fashion, e.g., see code
    // at the commit 13d4182a41e4e71034c6e1085699c5138805d21c. The price to pay however is
    // the loss of determinism. We can reconsider in the future whether it's worth it
    // to add an option to this constructor, e.g., par_random, defaulting to false.

    // Fetch a few quantities from prob.
    const auto nx = prob.get_nx();
    const auto ncx = nx - prob.get_nix();
    const auto &lb = prob.get_lb();
    const auto &ub = prob.get_ub();

    // LCOV_EXCL_START
    if (n > std::numeric_limits<vector_double::size_type>::max() / nx) {
        pagmo_throw(std::overflow_error,
                    "Cannot generate " + std::to_string(n)
                        + " random decision vectors in batch mode, as that would result in an overflow error");
    }
    // LCOV_EXCL_STOP

    // Check the problem bounds.
    for (vector_double::size_type i = 0u; i < ncx; ++i) {
        // NOTE: the lb<=ub check is not needed, as it is ensured by the problem class.
        // Still need to check for finiteness and range.
        detail::uniform_real_from_range_checks<true, false, true>(lb[i], ub[i]);
    }
    for (auto i = ncx; i < nx; ++i) {
        // NOTE: the lb<=ub check and the check that lb/ub are integral values are not needed,
        // as they are ensured by the problem class.
        // Still need to check for finiteness.
        detail::uniform_integral_from_range_checks<true, false, false>(lb[i], ub[i]);
        try {
            // Check that the bounds can be converted safely to long long.
            boost::numeric_cast<long long>(lb[i]);
            boost::numeric_cast<long long>(ub[i]);
        } catch (...) {
            pagmo_throw(std::invalid_argument,
                        "Cannot generate a random integer if the lower/upper bounds are not within "
                        "the bounds of the long long type");
        }
    }

    // Prepare the return value.
    vector_double out(nx * n);

    // Proceed to the random number generation.
    std::uniform_real_distribution<double> rdist;
    std::uniform_int_distribution<long long> idist;
    for (vector_double::size_type i = 0; i < out.size(); i += nx) {
        for (vector_double::size_type j = 0; j < ncx; ++j) {
            out[i + j] = (lb[j] == ub[j])
                             ? lb[j]
                             : rdist(r_engine, std::uniform_real_distribution<double>::param_type(lb[j], ub[j]));
        }
        for (vector_double::size_type j = ncx; j < nx; ++j) {
            out[i + j] = static_cast<double>(
                idist(r_engine, std::uniform_int_distribution<long long>::param_type(static_cast<long long>(lb[j]),
                                                                                     static_cast<long long>(ub[j]))));
        }
    }

    return out;
}

// Binomial coefficient
PAGMO_DLL_PUBLIC double binomial_coefficient(vector_double::size_type, vector_double::size_type);

// K-Nearest Neighbours
PAGMO_DLL_PUBLIC std::vector<std::vector<vector_double::size_type>> kNN(const std::vector<vector_double> &,
                                                                        std::vector<vector_double>::size_type);

namespace detail
{

// modifies a chromosome so that it will be in the bounds. elements that are off are resampled at random in the bounds
template <typename Rng>
inline void force_bounds_random(vector_double &x, const vector_double &lb, const vector_double &ub, Rng &r_engine)
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
PAGMO_DLL_PUBLIC void force_bounds_reflection(vector_double &, const vector_double &, const vector_double &);

// modifies a chromosome so that it will be in the bounds. Elements that are off are set on the bounds
PAGMO_DLL_PUBLIC void force_bounds_stick(vector_double &, const vector_double &, const vector_double &);

} // namespace detail

} // namespace pagmo

#endif
