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

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

rosenbrock::rosenbrock(vector_double::size_type dim) : m_dim(dim)
{
    if (dim < 2u) {
        pagmo_throw(std::invalid_argument,
                    "Rosenbrock Function must have minimum 2 dimensions, " + std::to_string(dim) + " requested");
    }
}

/// Fitness computation
/**
 * Computes the fitness for this UDP.
 *
 * @param x the decision vector.
 *
 * @return the fitness of \p x.
 */
vector_double rosenbrock::fitness(const vector_double &x) const
{
    double retval = 0.;
    for (decltype(m_dim) i = 0u; i < m_dim - 1u; ++i) {
        retval += 100. * (x[i] * x[i] - x[i + 1]) * (x[i] * x[i] - x[i + 1]) + (x[i] - 1) * (x[i] - 1);
    }
    return {retval};
}

/// Box-bounds
/**
 * @return the lower (-5.) and upper (10.) bounds for each decision vector component.
 */
std::pair<vector_double, vector_double> rosenbrock::get_bounds() const
{
    return {vector_double(m_dim, -5.), vector_double(m_dim, 10.)};
}

/// Gradient.
/**
 * @param x the input decision vector.
 *
 * @return the gradient of the fitness function in \p x.
 */
vector_double rosenbrock::gradient(const vector_double &x) const
{
    vector_double retval(m_dim);
    retval[0] = -400. * x[0] * (x[1] - x[0] * x[0]) - 2. * (1 - x[0]);
    for (unsigned i = 1; i < m_dim - 1u; ++i) {
        retval[i] = -400. * x[i] * (x[i + 1u] - x[i] * x[i]) - 2. * (1 - x[i]) + 200. * (x[i] - x[i - 1u] * x[i - 1u]);
    }
    retval[m_dim - 1u] = 200. * (x[m_dim - 1u] - x[m_dim - 2u] * x[m_dim - 2u]);
    return retval;
}

/// Optimal solution.
/**
 * @return the decision vector corresponding to the best solution for this problem.
 */
vector_double rosenbrock::best_known() const
{
    return vector_double(m_dim, 1.);
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
void rosenbrock::serialize(Archive &ar, unsigned)
{
    ar &m_dim;
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::rosenbrock)
