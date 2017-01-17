/* Copyright 2017 PaGMO development team
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

#ifndef PAGMO_PROBLEM_NULL_HPP
#define PAGMO_PROBLEM_NULL_HPP

#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{

/// Null problem
/**
 * This problem is used to test, develop and provide default values to e.g. meta-problems
 */
struct null_problem {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0., 0., 0.};
    }

    /// Number of objectives (one)
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }

    /// Equality constraint dimension (one)
    vector_double::size_type get_nec() const
    {
        return 1u;
    }

    /// Inequality constraint dimension (one)
    vector_double::size_type get_nic() const
    {
        return 1u;
    }

    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }

    /// Gradients
    vector_double gradient(const vector_double &) const
    {
        return {};
    }

    /// Gradient sparsity
    sparsity_pattern gradient_sparsity() const
    {
        return {};
    }

    /// Hessians
    std::vector<vector_double> hessians(const vector_double &) const
    {
        return {{}, {}, {}};
    }

    /// Hessian sparsity
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{}, {}, {}};
    }

    /// Problem name
    std::string get_name() const
    {
        return "Null problem";
    }

    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tA fictitious problem useful to test, debug and initialize default constructors";
    }

    /// Optimal solution
    vector_double best_known() const
    {
        return {0.};
    }

    /// Serialization
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};
}

PAGMO_REGISTER_PROBLEM(pagmo::null_problem)

#endif
