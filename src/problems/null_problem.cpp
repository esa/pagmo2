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
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

null_problem::null_problem(vector_double::size_type nobj, vector_double::size_type nec, vector_double::size_type nic,
                           vector_double::size_type nix)
    : m_nobj(nobj), m_nec(nec), m_nic(nic), m_nix(nix)
{
    if (!nobj) {
        pagmo_throw(std::invalid_argument, "The null problem must have a non-zero number of objectives");
    }
    if (nix > 1u) {
        pagmo_throw(std::invalid_argument, "The null problem must have an integer part strictly smaller than 2");
    }
}

/// Fitness.
/**
 * @return a zero-filled vector of size equal to the number of objectives.
 */
vector_double null_problem::fitness(const vector_double &) const
{
    return vector_double(get_nobj() + get_nec() + get_nic(), 0.);
}

/// Problem bounds.
/**
 * @return the pair <tt>([0.],[1.])</tt>.
 */
std::pair<vector_double, vector_double> null_problem::get_bounds() const
{
    return {{0.}, {1.}};
}

/// Serialization
/**
 * @param ar the target serialization archive.
 */
template <typename Archive>
void null_problem::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_nobj, m_nec, m_nic, m_nix);
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::null_problem)
