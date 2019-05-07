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

#ifndef PAGMO_PROBLEMS_LENNARD_JONES_HPP
#define PAGMO_PROBLEMS_LENNARD_JONES_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// The Lennard Jones Cluster problem.
/**
 *
 * \image html lennard_jones.jpg "Pictures of Lennard-Jones clusters." width=3cm
 *
 * This is a box-constrained continuous single-objecive problem. It represents the minimization
 * of the energy of a cluster of atoms assuming a Lennard-Jones potential between each pair.
 * The complexity for computing the objective function scales with the square of the number of atoms.
 *
 * The decision vector contains [z2, y3, z3, x4, y4, z4, ....] as the cartesian coordinates x1, y1, z1, x2, y2 and x3
 * are fixed to be zero.
 *
 * See: http://doye.chem.ox.ac.uk/jon/structures/LJ.html
 *
 */
class PAGMO_DLL_PUBLIC lennard_jones
{
public:
    /// Constructor from number of atoms
    /**
     * Constructs a Lennard Jones Clusters global optimisation problem
     *
     * @param atoms number of atoms in the cluster.
     *
     * @throw std::invalid_argument if \p atoms is < 3
     */
    lennard_jones(unsigned atoms = 3u);
    // Fitness computation
    vector_double fitness(const vector_double &) const;
    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;
    // Problem name
    std::string get_name() const;
    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Helper function that transforms the decision vector x in atoms positions r
    PAGMO_DLL_LOCAL double _r(unsigned, unsigned, const vector_double &) const;

    // Number of atoms
    unsigned m_atoms;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::lennard_jones)

#endif
