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

#ifndef PAGMO_PROBLEMS_HOCK_SCHITTKOWSKY_71_HPP
#define PAGMO_PROBLEMS_HOCK_SCHITTKOWSKY_71_HPP

#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Problem No.71 from the Hock Schittkowsky suite
/**
 * Mainly used for testing and debugging during PaGMO development, this
 * struct implements the problem No.71 from the Hock Schittkowsky suite:
 *
 * \f[
 *    \begin{array}{rl}
 *    \mbox{find: } & 1 \le \mathbf x \le 5 \\
 *    \mbox{to minimize: } & x_1x_4(x_1+x_2+x_3) + x_3 \\
 *    \mbox{subject to: } & x_1^2+x_2^2+x_3^2+x_4^2 - 40 = 0 \\
 *                        & 25 - x_1 x_2 x_3 x_4 \le 0
 *    \end{array}
 * \f]
 *
 * See: W. Hock and K. Schittkowski. Test examples for nonlinear programming codes.
 * Lecture Notes in Economics and Mathematical Systems, 187, 1981. doi: 10.1007/978-3-642-48320-2.
 *
 */
struct PAGMO_DLL_PUBLIC hock_schittkowsky_71 {
    // Fitness computation
    vector_double fitness(const vector_double &) const;

    /// Equality constraint dimension
    /**
     *
     * It returns the number of equality constraints
     *
     * @return the number of equality constraints
     */
    vector_double::size_type get_nec() const
    {
        return 1u;
    }

    /// Inequality constraint dimension
    /**
     *
     * It returns the number of inequality constraints
     *
     * @return the number of inequality constraints
     */
    vector_double::size_type get_nic() const
    {
        return 1u;
    }

    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;

    // Gradients
    vector_double gradient(const vector_double &) const;

    // Hessians
    std::vector<vector_double> hessians(const vector_double &) const;

    // Hessians sparsity (only the diagonal elements are non zero)
    std::vector<sparsity_pattern> hessians_sparsity() const;

    /// Problem name
    /**
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Hock Schittkowsky 71";
    }

    /// Extra info
    /**
     * @return a string containing extra info on the problem
     */
    std::string get_extra_info() const
    {
        return "\tProblem number 71 from the Hock-Schittkowsky test suite";
    }

    // Optimal solution
    vector_double best_known() const;

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);
};
} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::hock_schittkowsky_71)

#endif
