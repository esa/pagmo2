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

#ifndef PAGMO_PROBLEM_LUKSANVLCECK_HPP
#define PAGMO_PROBLEM_LUKSANVLCECK_HPP

#include <cassert>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../detail/constants.hpp"
#include "../exceptions.hpp"
#include "../problem.hpp" // needed for cereal registration macro
#include "../types.hpp"

namespace pagmo
{

/// Test problem from the Luksan and Vlcek
/**
 * Implementation of Example 5.1 in the report from Luksan and Vlcek.
 * Each equality constraint is here considered as two inequalities so that
 * the problem formulation is identical to the one provided in IPOPT examples/ScalableProblems folder.
 *
 * The problem is the Chanied Rosenbrock function with trigonometric-exponential
 * constraints.
 *
 * Its formulation in pagmo can be written as:
 *
 * \f[
 * \begin{array}{rl}
 * \mbox{find:}      & -5 \le \mathbf x_i \le 5, \forall i=1..n\\
 * \mbox{to minimize: } & \sum_{i=1}^{n-1}\left[100\left(x_i^2-x_{i+1}\right)^2 + \left(x_i-1\right)^2\right]\\
 * \mbox{subject to:} & 3x_{k+1}^3+2x_{k+2}-5+\sin(x_{k+1}-x_{k+2}})\sin(x_{k+1}+x_{k+2}})
 * +4x_{k+1}-x_k\exp(x_k-x_{k+1})-3 \le UB, \forall k=1..n-2 \\
 *                    & 3x_{k+1}^3+2x_{k+2}-5+\sin(x_{k+1}-x_{k+2}})\sin(x_{k+1}+x_{k+2}})
 * +4x_{k+1}-x_k\exp(x_k-x_{k+1})-3 \ge LB, \forall k=1..n-2 \\
 * \end{array}
 * \f]
 *
 * See: Lukšan, L., and Jan Vlcek. "Sparse and partially separable test problems for unconstrained and equality
 * constrained optimization." (1999). http://folk.uib.no/ssu029/Pdf_file/Luksan99.ps
 *
 */
struct luksan_vlcek1 {
    /// Constructor from dimension and bounds
    /**
     * Constructs the luksan_vlcek1 problem. Setting \plb = \p ub = 0
     * corresponds to the original formulation
     *
     * @param dim the problem dimensions.
     * @param clb lower bounds for the constraints
     * @param cub upper bounds for the constraints
     *
     * @throw std::invalid_argument if \p dim is < 1
     */
    luksan_vlcek1(unsigned int dim = 1u, double clb = 0., double cub = 0.) : m_dim(dim), m_clb(clb), m_cub(cub)
    {
        if (dim < 3u) {
            pagmo_throw(std::invalid_argument,
                        "luksan_vlcek1 must have minimum 3 dimension, " + std::to_string(dim) + " requested");
        }
        if (clb > cub) {
            pagmo_throw(std::invalid_argument, ,
                        "constraints lower bound " + std::to_string(clb) + "is higher than the upper bound "
                            + std::to_string(cub));
        }
    };
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
        assert(x.size() == m_dim);
        auto n = x.size();
        // 1 objective and 2 * (n-2) inequalities
        vector_double f(1 + 2 * (n - 2), 0.);
        f[0] = 0.;
        for (decltype(n) i = 0u; i < n - 1u; ++i) {
            double a1 = x[i] * x[i] - x[i + 1];
            double a2 = x[i] - 1.;
            f[0] += 100. * a1 * a1 + a2 * a2;
        }
        for (decltype(n) i = 0u; i < n - 2u; ++i) {
            f[2 * i + 1] = (3. * std::pow(x[i + 1], 3.) + 2. * x[i + 2] - 5.
                            + std::sin(x[i + 1] - x[i + 2]) * std::sin(x[i + 1] + x[i + 2]) + 4. * x[i + 1]
                            - x[i] * std::exp(x[i] - x[i + 1]) - 3.)
                           - m_cub[i];
            f[2 * i + 1 + 1] = -(3. * std::pow(x[i + 1], 3.) + 2. * x[i + 2] - 5.
                                 + std::sin(x[i + 1] - x[i + 2]) * std::sin(x[i + 1] + x[i + 2]) + 4. * x[i + 1]
                                 - x[i] * std::exp(x[i] - x[i + 1]) - 3.)
                               + m_clb[i];
        }
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
        vector_double lb(m_dim, -5.);
        vector_double ub(m_dim, 5.);
        return {lb, ub};
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
        return 2 * (m_dim - 2);
    }
    /// Gradients sparsity
    /**
     *
     * It returns the gradent sparisty structure for the Luksan Vlcek 1 problem
     *
     * The gradients sparisty is represented in the form required by
     * problem::gradient_sparsity().
     *
     * @return the gradient sparsity structure of the fitness function
     */
    sparsity_pattern gradient_sparsity() const
    {
        sparsity_pattern retval;
        // The part relative to the objective function is dense
        for (decltype(m_dim) i = 0u; i < m_dim; ++i) {
            retval.push_back({0, i});
        }
        // The part relative to the inequality constraints is sparse as each
        // constraint c_k depends on x_k, x_{k+1} and x_{k+2}
        for (decltype(m_dim) i = 0u; i < m_dim - 2u; ++i) {
            retval.push_back({2 * i + 1, i});
            retval.push_back({2 * i + 1, i + 1});
            retval.push_back({2 * i + 1, i + 2});
            retval.push_back({2 * i + 2, i});
            retval.push_back({2 * i + 2, i + 1});
            retval.push_back({2 * i + 2, i + 2});
        }
    }
    /// Problem name
    /**
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "luksan_vlcek1";
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
    void serialize(Archive &ar)
    {
        ar(m_dim, m_lb, m_ub);
    }
    /// Problem dimensions
    unsigned int m_dim;
    double m_lb;
    double m_ub;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::luksan_vlcek1)

#endif
