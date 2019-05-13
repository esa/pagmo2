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

#ifndef PAGMO_PROBLEMS_NULL_PROBLEM_HPP
#define PAGMO_PROBLEMS_NULL_PROBLEM_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Null problem
/**
 * This problem is used to implement the default constructors of pagmo::problem and of the meta-problems.
 */
struct PAGMO_DLL_PUBLIC null_problem {
    /// Constructor from number of objectives.
    /**
     * @param nobj the desired number of objectives.
     * @param nec the desired number of equality constraints.
     * @param nic the desired number of inequality constraints.
     * @param nix the problem integer dimension.
     *
     * @throws std::invalid_argument if \p nobj is zero.
     */
    null_problem(vector_double::size_type nobj = 1u, vector_double::size_type nec = 0u,
                 vector_double::size_type nic = 0u, vector_double::size_type nix = 0u);
    // Fitness.
    vector_double fitness(const vector_double &) const;
    // Problem bounds.
    std::pair<vector_double, vector_double> get_bounds() const;
    /// Number of objectives.
    /**
     * @return the number of objectives of the problem (as specified upon construction).
     */
    vector_double::size_type get_nobj() const
    {
        return m_nobj;
    }
    /// Number of equality constraints.
    /**
     * @return the number of equality constraints of the problem (as specified upon construction).
     */
    vector_double::size_type get_nec() const
    {
        return m_nec;
    }
    /// Number of inequality constraints.
    /**
     * @return the number of inequality constraints of the problem (as specified upon construction).
     */
    vector_double::size_type get_nic() const
    {
        return m_nic;
    }
    /// Size of the integer part.
    /**
     * @return the size of the integer part (as specified upon construction).
     */
    vector_double::size_type get_nix() const
    {
        return m_nix;
    }
    /// Problem name.
    /**
     * @return <tt>"Null problem"</tt>.
     */
    std::string get_name() const
    {
        return "Null problem";
    }
    // Serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    vector_double::size_type m_nobj;
    vector_double::size_type m_nec;
    vector_double::size_type m_nic;
    vector_double::size_type m_nix;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::null_problem)

#endif
