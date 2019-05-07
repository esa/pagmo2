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
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Fitness computation
/**
 * Computes the fitness for this UDP
 *
 * @param x the decision vector.
 *
 * @return the fitness of \p x.
 */
vector_double inventory::fitness(const vector_double &x) const
{
    // We seed the random engine
    m_e.seed(m_seed);
    // We construct a uniform distribution from 0 to 1.
    auto drng = std::uniform_real_distribution<double>(0., 1.);
    // We may now start the computations
    const double c = 1.0, b = 1.5,
                 h = 0.1; // c is the cost per unit, b is the backorder penalty cost and h is the holding cost
    double retval = 0;

    for (decltype(m_sample_size) i = 0; i < m_sample_size; ++i) {
        double I = 0;
        for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
            double d = drng(m_e) * 100;
            retval += c * x[j] + b * std::max<double>(d - I - x[j], 0) + h * std::max<double>(I + x[j] - d, 0);
            I = std::max<double>(0, I + x[j] - d);
        }
    }
    return {retval / m_sample_size};
}

/// Box-bounds
/**
 * It returns the box-bounds for this UDP.
 *
 * @return the lower and upper bounds for each of the decision vector components
 */
std::pair<vector_double, vector_double> inventory::get_bounds() const
{
    vector_double lb(m_weeks, 0.);
    vector_double ub(m_weeks, 200.);
    return {lb, ub};
}

/// Extra info
/**
 * @return a string containing extra info on the problem
 */
std::string inventory::get_extra_info() const
{
    std::ostringstream ss;
    ss << "\tWeeks: " << std::to_string(m_weeks) << "\n";
    ss << "\tSample size: " << std::to_string(m_sample_size) << "\n";
    ss << "\tSeed: " << std::to_string(m_seed) << "\n";
    return ss.str();
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
void inventory::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_weeks, m_sample_size, m_e, m_seed);
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::inventory)
