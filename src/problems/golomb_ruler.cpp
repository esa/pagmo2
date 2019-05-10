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
#include <initializer_list>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/golomb_ruler.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

golomb_ruler::golomb_ruler(unsigned order, unsigned upper_bound) : m_order(order), m_upper_bound(upper_bound)
{
    if (order < 2u) {
        pagmo_throw(std::invalid_argument, "Golomb ruler problem must have at least order 2, while "
                                               + std::to_string(order) + " was requested.");
    }
    if (upper_bound < 2u) {
        pagmo_throw(std::invalid_argument,
                    "The upper bound for the maximum distance between consecutive ticks has to be at least 2, while "
                        + std::to_string(upper_bound) + " was requested.");
    }
    // Overflow can occur when evaluating the fitness later if the upper_bound is too large.
    if (upper_bound > std::numeric_limits<unsigned>::max() / (order - 1u)) {
        pagmo_throw(std::overflow_error,
                    "Overflow in Golomb ruler problem, select a smaller maximum distance between consecutive ticks.");
    }
}

/// Fitness computation
/**
 * Computes the fitness for this UDP
 *
 * @param x the decision vector.
 *
 * @return the fitness of \p x.
 */
vector_double golomb_ruler::fitness(const vector_double &x) const
{
    vector_double f(2, 0.);
    // 1 - We compute the ticks (the ruler length will be the last tick since we start from 0)
    vector_double ticks(x.size() + 1, 0.);
    std::partial_sum(x.begin(), x.end(), ticks.begin() + 1u);
    f[0] = ticks.back();
    // 2 - We compute all pairwise distances
    vector_double distances;
    distances.reserve(x.size() * (x.size() - 1) / 2);
    for (decltype(ticks.size()) i = 0; i < ticks.size() - 1; ++i) {
        for (decltype(ticks.size()) j = i + 1; j < ticks.size(); ++j) {
            distances.push_back(ticks[j] - ticks[i]);
        }
    }
    // 3 - We compute how many duplicate distances are there.
    std::sort(distances.begin(), distances.end(), detail::less_than_f<double>);
    f[1] = static_cast<double>(distances.size())
           - static_cast<double>(std::distance(
               distances.begin(), std::unique(distances.begin(), distances.end(), detail::equal_to_f<double>)));
    return f;
}

/// Box-bounds
/**
 * Returns the box-bounds for this UDP.
 *
 * @return the lower and upper bounds for each of the decision vector components
 */
std::pair<vector_double, vector_double> golomb_ruler::get_bounds() const
{
    unsigned prob_dim = m_order - 1u;
    vector_double lb(prob_dim, 1);
    vector_double ub(prob_dim, m_upper_bound);
    return {lb, ub};
}

/// Problem name
/**
 * Returns the problem name.
 *
 * @return a string containing the problem name
 */
std::string golomb_ruler::get_name() const
{
    return "Golomb Ruler (order " + std::to_string(m_order) + ")";
}

/// Object serialization
/**
 * This method will save/load \p this into the archive \p ar.
 *
 * @param ar target archive.
 *
 * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
 */
template <typename Archive>
void golomb_ruler::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_order, m_upper_bound);
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::golomb_ruler)
