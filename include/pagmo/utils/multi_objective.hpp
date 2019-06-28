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

#ifndef PAGMO_MULTI_OBJECTIVE_HPP
#define PAGMO_MULTI_OBJECTIVE_HPP

#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/discrepancy.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

namespace detail
{

// Recursive function building all m-ple of elements of X summing to s
PAGMO_DLL_PUBLIC void reksum(std::vector<std::vector<double>> &, const std::vector<pop_size_t> &, pop_size_t,
                             pop_size_t, std::vector<double> = std::vector<double>());

} // namespace detail

// Pareto-dominance
PAGMO_DLL_PUBLIC bool pareto_dominance(const vector_double &, const vector_double &);

// Non dominated front 2D (Kung's algorithm)
PAGMO_DLL_PUBLIC std::vector<pop_size_t> non_dominated_front_2d(const std::vector<vector_double> &);

/// Return type for the fast_non_dominated_sorting algorithm
using fnds_return_type = std::tuple<std::vector<std::vector<pop_size_t>>, std::vector<std::vector<pop_size_t>>,
                                    std::vector<pop_size_t>, std::vector<pop_size_t>>;

// Fast non dominated sorting
PAGMO_DLL_PUBLIC fnds_return_type fast_non_dominated_sorting(const std::vector<vector_double> &);

// Crowding distance
PAGMO_DLL_PUBLIC vector_double crowding_distance(const std::vector<vector_double> &);

// Sorts a population in multi-objective optimization
PAGMO_DLL_PUBLIC std::vector<pop_size_t> sort_population_mo(const std::vector<vector_double> &);

// Selects the best N individuals in multi-objective optimization
PAGMO_DLL_PUBLIC std::vector<pop_size_t> select_best_N_mo(const std::vector<vector_double> &, pop_size_t);

// Ideal point
PAGMO_DLL_PUBLIC vector_double ideal(const std::vector<vector_double> &);

// Nadir point
PAGMO_DLL_PUBLIC vector_double nadir(const std::vector<vector_double> &);

/// Decomposition weights generation
/**
 * Generates a requested number of weight vectors to be used to decompose a multi-objective problem. Three methods are
 * available:
 * - "grid" generates weights on an uniform grid. This method may only be used when the number of requested weights to
 * be genrated is such that a uniform grid is indeed possible. In
 * two dimensions this is always the case, but in larger dimensions uniform grids are possible only in special cases
 * - "random" generates weights randomly distributing them uniformly on the simplex (weights are such that \f$\sum_i
 * \lambda_i = 1\f$)
 * - "low discrepancy" generates weights using a low-discrepancy sequence to, eventually, obtain a
 * better coverage of the Pareto front. Halton sequence is used since low dimensionalities are expected in the number of
 * objectives (i.e. less than 20), hence Halton sequence is deemed as appropriate.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    All genration methods are guaranteed to generate weights on the simplex (:math:`\sum_i \lambda_i = 1`). All
 *    weight generation methods are guaranteed to generate the canonical weights [1,0,0,...], [0,1,0,..], ... first.
 *
 * \endverbatim
 *
 * Example: to generate 10 weights distributed somewhat regularly to decompose a three dimensional problem:
 * @code{.unparsed}
 * std::mt19937 r_engine;
 * auto lambdas = decomposition_weights(3u, 10u, "low discrepancy", r_engine);
 * @endcode
 *
 * @param n_f dimension of each weight vector (i.e. fitness dimension)
 * @param n_w number of weights to be generated
 * @param method methods to generate the weights of the decomposed problems. One of "grid", "random",
 *"low discrepancy"
 * @param r_engine a C++ random engine
 *
 * @returns an <tt>std:vector</tt> containing the weight vectors
 *
 * @throws if \p nf and \p nw are not compatible with the selected weight generation method or if \p method
 * is not one of "grid", "random" or "low discrepancy"
 */
template <typename Rng>
inline std::vector<vector_double> decomposition_weights(vector_double::size_type n_f, vector_double::size_type n_w,
                                                        const std::string &method, Rng &r_engine)
{
    // Sanity check
    if (n_f > n_w) {
        pagmo_throw(std::invalid_argument,
                    "A fitness size of " + std::to_string(n_f)
                        + " was requested to the weight generation routine, while " + std::to_string(n_w)
                        + " weights were requested to be generated. To allow weight be generated correctly the number "
                          "of weights must be strictly larger than the number of objectives");
    }

    if (n_f < 2u) {
        pagmo_throw(
            std::invalid_argument,
            "A fitness size of " + std::to_string(n_f)
                + " was requested to generate decomposed weights. A dimension of at least two must be requested.");
    }

    // Random distributions
    std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)
    std::vector<vector_double> retval;
    if (method == "grid") {
        // find the largest H resulting in a population smaller or equal to NP
        decltype(n_w) H;
        if (n_f == 2u) {
            H = n_w - 1u;
        } else if (n_f == 3u) {
            H = static_cast<decltype(H)>(std::floor(0.5 * (std::sqrt(8. * static_cast<double>(n_w) + 1.) - 3.)));
        } else {
            H = 1u;
            while (binomial_coefficient(H + n_f - 1u, n_f - 1u) <= static_cast<double>(n_w)) {
                ++H;
            }
            H--;
        }
        // We check that NP equals the population size resulting from H
        if (std::abs(static_cast<double>(n_w) - binomial_coefficient(H + n_f - 1u, n_f - 1u)) > 1E-8) {
            std::ostringstream error_message;
            error_message << "Population size of " << std::to_string(n_w) << " is detected, but not supported by the '"
                          << method << "' weight generation method selected. A size of "
                          << binomial_coefficient(H + n_f - 1u, n_f - 1u) << " or "
                          << binomial_coefficient(H + n_f, n_f - 1u) << " is possible.";
            pagmo_throw(std::invalid_argument, error_message.str());
        }
        // We generate the weights
        std::vector<pop_size_t> range(H + 1u);
        std::iota(range.begin(), range.end(), std::vector<pop_size_t>::size_type(0u));
        detail::reksum(retval, range, n_f, H);
        for (decltype(retval.size()) i = 0u; i < retval.size(); ++i) {
            for (decltype(retval[i].size()) j = 0u; j < retval[i].size(); ++j) {
                retval[i][j] /= static_cast<double>(H);
            }
        }
    } else if (method == "low discrepancy") {
        // We first push back the "corners" [1,0,0,...], [0,1,0,...]
        for (decltype(n_f) i = 0u; i < n_f; ++i) {
            retval.push_back(vector_double(n_f, 0.));
            retval[i][i] = 1.;
        }
        // Then we add points on the simplex randomly genrated using Halton low discrepancy sequence
        halton ld_seq{boost::numeric_cast<unsigned>(n_f - 1u), boost::numeric_cast<unsigned>(n_f)};
        for (decltype(n_w) i = n_f; i < n_w; ++i) {
            retval.push_back(sample_from_simplex(ld_seq()));
        }
    } else if (method == "random") {
        // We first push back the "corners" [1,0,0,...], [0,1,0,...]
        for (decltype(n_f) i = 0u; i < n_f; ++i) {
            retval.push_back(vector_double(n_f, 0.));
            retval[i][i] = 1.;
        }
        for (decltype(n_w) i = n_f; i < n_w; ++i) {
            vector_double dummy(n_f - 1u, 0.);
            for (decltype(n_f) j = 0u; j < n_f - 1u; ++j) {
                dummy[j] = drng(r_engine);
            }
            retval.push_back(sample_from_simplex(dummy));
        }
    } else {
        pagmo_throw(std::invalid_argument,
                    "Weight generation method " + method
                        + " is unknown. One of 'grid', 'random' or 'low discrepancy' was expected");
    }
    return retval;
}

// Decomposes a vector of objectives.
PAGMO_DLL_PUBLIC vector_double decompose_objectives(const vector_double &, const vector_double &, const vector_double &,
                                                    const std::string &);

} // namespace pagmo

#endif
