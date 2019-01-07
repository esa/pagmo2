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

#ifndef PAGMO_PROBLEM_LUKSANVLCECK_HPP
#define PAGMO_PROBLEM_LUKSANVLCECK_HPP

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp> // needed for cereal registration macro
#include <pagmo/types.hpp>

namespace pagmo
{

/// Test problem from Luksan and Vlcek
/**
 * Implementation of Example 5.1 in the report from Luksan and Vlcek.
 *
 * The problem is the Chained Rosenbrock function with trigonometric-exponential
 * constraints.
 *
 * Its formulation can be written as:
 *
 * \f[
 *   \begin{array}{rl}
 *   \mbox{find:} & -5 \le x_i \le 5, \forall i=1..n \\
 *   \mbox{to minimize: } & \sum_{i=1}^{n-1}\left[100\left(x_i^2-x_{i+1}\right)^2 + \left(x_i-1\right)^2\right] \\
 *   \mbox{subject to:} &
 * 3x_{k+1}^3+2x_{k+2}-5+\sin(x_{k+1}-x_{k+2})\sin(x_{k+1}+x_{k+2}) + \\
 * & +4x_{k+1}-x_k\exp(x_k-x_{k+1})-3 = 0, \forall k=1..n-2
 * \end{array}
 * \f]
 *
 * See: Luksan, L., and Jan Vlcek. "Sparse and partially separable test problems for unconstrained and equality
 * constrained optimization." (1999). http://hdl.handle.net/11104/0123965
 *
 */
struct luksan_vlcek1 {
    /// Constructor from dimension and bounds
    /**
     * Constructs the luksan_vlcek1 problem.
     *
     * @param dim the problem dimensions.
     *
     * @throw std::invalid_argument if \p dim is < 3.
     */
    luksan_vlcek1(unsigned dim = 3u) : m_dim(dim)
    {
        if (dim < 3u) {
            pagmo_throw(std::invalid_argument,
                        "luksan_vlcek1 must have minimum 3 dimension, " + std::to_string(dim) + " requested");
        }
    };
    /// Fitness computation
    /**
     * Computes the fitness for this UDP.
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     */
    vector_double fitness(const vector_double &x) const
    {
        assert(x.size() == m_dim);
        auto n = x.size();
        // 1 objective and (n-2) equalities
        vector_double f(1 + (n - 2), 0.);
        f[0] = 0.;
        for (decltype(n) i = 0u; i < n - 1u; ++i) {
            double a1 = x[i] * x[i] - x[i + 1];
            double a2 = x[i] - 1.;
            f[0] += 100. * a1 * a1 + a2 * a2;
        }
        for (decltype(n) i = 0u; i < n - 2u; ++i) {
            f[i + 1] = (3. * std::pow(x[i + 1], 3.) + 2. * x[i + 2] - 5.
                        + std::sin(x[i + 1] - x[i + 2]) * std::sin(x[i + 1] + x[i + 2]) + 4. * x[i + 1]
                        - x[i] * std::exp(x[i] - x[i + 1]) - 3.);
        }
        return f;
    }
    /// Box-bounds
    /**
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components.
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return std::make_pair(vector_double(m_dim, -5.), vector_double(m_dim, 5.));
    }
    /// Equality constraint dimension
    /**
     * @return the number of equality constraints.
     */
    vector_double::size_type get_nec() const
    {
        return m_dim - 2u;
    }
    /// Gradients
    /**
     * It returns the fitness gradient.
     *
     * The gradient is represented in a sparse form as required by
     * problem::gradient().
     *
     * @param x the decision vector.
     *
     * @return the gradient of the fitness function.
     */
    vector_double gradient(const vector_double &x) const
    {
        assert(x.size() == m_dim);
        auto n = x.size();
        vector_double grad(n + 3 * (n - 2), 0.);
        for (decltype(n) i = 0u; i < n - 1; ++i) {
            grad[i] += 400. * x[i] * (x[i] * x[i] - x[i + 1]) + 2. * (x[i] - 1.);
            grad[i + 1] = -200. * (x[i] * x[i] - x[i + 1]);
        }
        for (decltype(n) i = 0u; i < n - 2; ++i) {
            // x[i]
            grad[n + 3 * i] = -(1. + x[i]) * std::exp(x[i] - x[i + 1]);
            // x[i+1]
            grad[n + 1 + 3 * i] = 9. * x[i + 1] * x[i + 1]
                                  + std::cos(x[i + 1] - x[i + 2]) * std::sin(x[i + 1] + x[i + 2])
                                  + std::sin(x[i + 1] - x[i + 2]) * std::cos(x[i + 1] + x[i + 2]) + 4.
                                  + x[i] * std::exp(x[i] - x[i + 1]);
            // x[i+2]
            grad[n + 2 + 3 * i] = 2. - std::cos(x[i + 1] - x[i + 2]) * std::sin(x[i + 1] + x[i + 2])
                                  + std::sin(x[i + 1] - x[i + 2]) * std::cos(x[i + 1] + x[i + 2]);
        }
        return grad;
    }
    /// Gradients sparsity
    /**
     * It returns the gradent sparisty structure for the Luksan Vlcek 1 problem.
     *
     * The gradients sparisty is represented in the form required by
     * problem::gradient_sparsity().
     *
     * @return the gradient sparsity structure of the fitness function.
     */
    sparsity_pattern gradient_sparsity() const
    {
        sparsity_pattern retval;
        // The part relative to the objective function is dense
        for (decltype(m_dim) i = 0u; i < m_dim; ++i) {
            retval.emplace_back(0, i);
        }
        // The part relative to the equality constraints is sparse as each
        // constraint c_k depends on x_k, x_{k+1} and x_{k+2}
        for (decltype(m_dim) i = 0u; i < m_dim - 2u; ++i) {
            retval.emplace_back(i + 1, i);
            retval.emplace_back(i + 1, i + 1);
            retval.emplace_back(i + 1, i + 2);
        }
        return retval;
    }
    /// Problem name
    /**
     * @return a string containing the problem name.
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
        ar(m_dim);
    }
    /// Problem dimensions.
    unsigned int m_dim;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::luksan_vlcek1)

#endif
