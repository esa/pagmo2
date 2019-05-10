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
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/lennard_jones.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

lennard_jones::lennard_jones(unsigned atoms) : m_atoms(atoms)
{
    if (atoms < 3) {
        pagmo_throw(std::invalid_argument, "The number of atoms in a Lennard Jones Clusters problem must be "
                                           "positive and greater than 2, while a number of "
                                               + std::to_string(atoms) + " was detected.");
    }
    if (m_atoms - 2u > std::numeric_limits<unsigned>::max() / 3u) {
        pagmo_throw(std::overflow_error,
                    "Overflow caused by the number of atoms requested: " + std::to_string(m_atoms));
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
vector_double lennard_jones::fitness(const vector_double &x) const
{
    vector_double f(1, 0.);
    // We evaluate the potential
    for (unsigned i = 0u; i < (m_atoms - 1u); ++i) {
        for (unsigned j = (i + 1u); j < m_atoms; ++j) {
            double sixth, dist;
            dist = std::pow(_r(i, 0u, x) - _r(j, 0u, x), 2) + std::pow(_r(i, 1u, x) - _r(j, 1u, x), 2)
                   + std::pow(_r(i, 2u, x) - _r(j, 2u, x), 2); // rij^2
            if (dist == 0.0) {
                f[0] = std::numeric_limits<double>::max();
            } else {
                sixth = std::pow(dist, -3); // rij^-6
                f[0] += (std::pow(sixth, 2) - sixth);
            }
        }
    }
    f[0] = 4 * f[0];
    return f;
}

/// Box-bounds
/**
 * Returns the box-bounds for this UDP.
 *
 * @return the lower and upper bounds for each of the decision vector components
 */
std::pair<vector_double, vector_double> lennard_jones::get_bounds() const
{
    unsigned prob_dim = 3u * m_atoms - 6u;
    vector_double lb(prob_dim, -3.0);
    vector_double ub(prob_dim, 3.0);
    for (unsigned i = 0u; i < 3u * m_atoms - 6u; ++i) {
        if ((i != 0) && (i % 3) == 0) {
            lb[i] = 0.0;
            ub[i] = 6.0;
        }
    }
    return {lb, ub};
}

/// Problem name
/**
 * Returns the problem name.
 *
 * @return a string containing the problem name
 */
std::string lennard_jones::get_name() const
{
    return "Lennard Jones Cluster (" + std::to_string(m_atoms) + " atoms)";
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
void lennard_jones::serialize(Archive &ar, unsigned)
{
    ar &m_atoms;
}

// Helper function that transforms the decision vector x in atoms positions r
double lennard_jones::_r(unsigned atom, unsigned coord, const vector_double &x) const
{
    if (atom == 0u) { // x1,y1,z1 fixed
        return 0.0;
    } else if (atom == 1) {
        if (coord < 2u) { // x2,y2    fixed
            return 0.0;
        } else { // z2 is a variable
            return x[0];
        }
    } else if (atom == 2u) {
        if (coord == 0u) { // x3 fixed
            return 0.0;
        } else { // y3 and z3 are variables
            return x[coord];
        }
    } else {
        return x[3u * (atom - 2u) + coord];
    }
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::lennard_jones)
