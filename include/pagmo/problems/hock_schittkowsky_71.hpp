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

#ifndef PAGMO_PROBLEM_HOCK_SCHITTKOWSKY_71_HPP
#define PAGMO_PROBLEM_HOCK_SCHITTKOWSKY_71_HPP

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/io.hpp>
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
struct hock_schittkowsky_71 {
    /// Fitness computation
    /**
     * Computes the fitness for this UDP
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     */
    vector_double fitness(const vector_double &x) const
    {
        return {
            x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],                   // objfun
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40., // equality con.
            25. - x[0] * x[1] * x[2] * x[3]                              // inequality con.
        };
    }

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

    /// Box-bounds
    /**
     *
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{1., 1., 1., 1.}, {5., 5., 5., 5.}};
    }

    /// Gradients
    /**
     *
     * It returns the fitness gradient for this UDP.
     *
     * The gradient is represented in a sparse form as required by
     * problem::gradient().
     *
     * @param x the decision vector.
     *
     * @return the gradient of the fitness function
     */
    vector_double gradient(const vector_double &x) const
    {
        return {x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1,
                x[0] * (x[0] + x[1] + x[2]),
                2 * x[0],
                2 * x[1],
                2 * x[2],
                2 * x[3],
                -x[1] * x[2] * x[3],
                -x[0] * x[2] * x[3],
                -x[0] * x[1] * x[3],
                -x[0] * x[1] * x[2]};
    }

    /// Hessians
    /**
     *
     * It returns the hessians for this UDP.
     *
     * The hessians are represented in a sparse form as required by
     * problem::hessians().
     *
     * @param x the decision vector.
     *
     * @return the hessians of the fitness function
     */
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        return {{2 * x[3], x[3], x[3], 2 * x[0] + x[1] + x[2], x[0], x[0]},
                {2., 2., 2., 2.},
                {-x[2] * x[3], -x[1] * x[3], -x[0] * x[3], -x[1] * x[2], -x[0] * x[2], -x[0] * x[1]}};
    }

    /// Hessians sparsity (only the diagonal elements are non zero)
    /**
     *
     * It returns the hessian sparisty structure for this UDP.
     *
     * The hessian sparisty is represented in the form required by
     * problem::hessians_sparsity().
     *
     * @return the hessians of the fitness function
     */
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 1}, {3, 2}},
                {{0, 0}, {1, 1}, {2, 2}, {3, 3}},
                {{1, 0}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {3, 2}}};
    }

    /// Problem name
    /**
     *
     *
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Hock Schittkowsky 71";
    }

    /// Extra informations
    /**
     *
     *
     * @return a string containing extra informations on the problem
     */
    std::string get_extra_info() const
    {
        return "\tProblem number 71 from the Hock-Schittkowsky test suite";
    }

    /// Optimal solution
    /**
     * @return the decision vector corresponding to the best solution for this problem.
     */
    vector_double best_known() const
    {
        return {1., 4.74299963, 3.82114998, 1.37940829};
    }

    /// Object serialization
    /**
     * This method is needed by the cereal serialization pipeline
     */
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};
} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::hock_schittkowsky_71)

#endif
