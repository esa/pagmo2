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

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

schwefel::schwefel(unsigned dim) : m_dim(dim)
{
    if (dim < 1u) {
        pagmo_throw(std::invalid_argument,
                    "Schwefel Function must have minimum 1 dimension, " + std::to_string(dim) + " requested");
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
vector_double schwefel::fitness(const vector_double &x) const
{
    vector_double f(1, 0.);
    auto n = x.size();
    for (decltype(n) i = 0u; i < n; i++) {
        f[0] += x[i] * std::sin(std::sqrt(std::abs(x[i])));
    }
    f[0] = 418.9828872724338 * static_cast<double>(n) - f[0];
    return f;
}

/// Box-bounds
/**
 * It returns the box-bounds for this UDP.
 *
 * @return the lower and upper bounds for each of the decision vector components
 */
std::pair<vector_double, vector_double> schwefel::get_bounds() const
{
    vector_double lb(m_dim, -500);
    vector_double ub(m_dim, 500);
    return {lb, ub};
}

/// Optimal solution
/**
 * @return the decision vector corresponding to the best solution for this problem.
 */
vector_double schwefel::best_known() const
{
    return vector_double(m_dim, 420.9687);
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
void schwefel::serialize(Archive &ar, unsigned)
{
    ar &m_dim;
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::schwefel)
