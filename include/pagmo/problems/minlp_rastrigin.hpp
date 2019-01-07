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

#ifndef PAGMO_PROBLEM_MINLP_RASTRIGIN_HPP
#define PAGMO_PROBLEM_MINLP_RASTRIGIN_HPP

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// A MINLP version of the Rastrigin problem
/**
 *
 * \image html rastrigin.png "Two-dimensional Rastrigin function." width=3cm
 *
 * This is a scalable, box-constrained, mixed integer nonlinear programmng (MINLP) problem.
 * The objective function is the generalised n-dimensional Rastrigin function:
 * \f[
 * 	F\left(x_1,\ldots,x_n\right) = 10 \cdot n + \sum_{i=1}^n x_i^2 - 10\cdot\cos\left( 2\pi \cdot x_i \right)
 * \f]
 *
 * where we constraint the last \f$m\f$ components of the decision vector to be integers. The variables are
 * box bounded as follows: \f$\quad x_i \in [-5.12,5.12], \forall i = 1 .. n-m\f$, \f$\quad x_i \in [-10,-5], \forall
 * i = m+1 .. n\f$
 *
 * Gradients (dense) are also provided (also for the integer part) as:
 * \f[
 * 	G_i\left(x_1,\ldots,x_n\right) = 2 x_i + 10 \cdot 2\pi \cdot\sin\left( 2\pi \cdot x_i \right)
 * \f]
 * And Hessians (sparse as only the diagonal is non-zero) are:
 * \f[
 * 	H_{ii}\left(x_1,\ldots,x_n\right) = 2 + 10 \cdot 4\pi^2 \cdot\cos\left( 2\pi \cdot x_i \right)
 * \f]
 */
struct minlp_rastrigin {
    /// Constructor from continuous and integer dimension
    /**
     * Constructs a MINLP Rastrigin problem.
     *
     * @param dim_c the problem continuous dimension.
     * @param dim_i the problem continuous dimension.
     *
     * @throw std::invalid_argument if \p dim_c+ \p dim_i is < 1
     */
    minlp_rastrigin(unsigned dim_c = 1u, unsigned dim_i = 1u) : m_dim_c(dim_c), m_dim_i(dim_i)
    {
        if (dim_c + dim_i < 1u) {
            pagmo_throw(std::invalid_argument, "Minlp Rastrigin Function must have minimum 1 dimension, "
                                                   + std::to_string(dim_c + dim_i) + " requested");
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
        vector_double f(1, 0.);
        const auto omega = 2. * pagmo::detail::pi();
        auto n = x.size();
        for (decltype(n) i = 0u; i < n; ++i) {
            f[0] += x[i] * x[i] - 10. * std::cos(omega * x[i]);
        }
        f[0] += 10. * static_cast<double>(n);
        return f;
    }

    /// Box-bounds
    /**
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(m_dim_c + m_dim_i, -5.12);
        vector_double ub(m_dim_c + m_dim_i, 5.12);
        for (decltype(m_dim_i) i = m_dim_c; i < m_dim_i + m_dim_c; ++i) {
            lb[i] = -10;
            ub[i] = -5;
        }
        return {lb, ub};
    }

    /// Integer dimension
    /**
     * It returns the integer dimension of the problem.
     *
     * @return the integer dimension of the problem.
     */
    vector_double::size_type get_nix() const
    {
        return m_dim_i;
    }
    /// Gradients
    /**
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
        auto n = x.size();
        vector_double g(n);
        const auto omega = 2. * pagmo::detail::pi();
        for (decltype(n) i = 0u; i < n; ++i) {
            g[i] = 2 * x[i] + 10.0 * omega * std::sin(omega * x[i]);
        }
        return g;
    }

    /// Hessians
    /**
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
        auto n = x.size();
        vector_double h(n);
        const auto omega = 2. * pagmo::detail::pi();
        for (decltype(n) i = 0u; i < n; ++i) {
            h[i] = 2 + 10.0 * omega * omega * std::cos(omega * x[i]);
        }
        return {h};
    }

    /// Hessians sparsity (only the diagonal elements are non zero)
    /**
     * It returns the hessian sparisty structure for this UDP.
     *
     * The hessian sparisty is represented in the form required by
     * problem::hessians_sparsity().
     *
     * @return the hessians of the fitness function
     */
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        sparsity_pattern hs;
        auto n = m_dim_c + m_dim_i;
        for (decltype(n) i = 0u; i < n; ++i) {
            hs.push_back({i, i});
        }
        return {hs};
    }
    /// Problem name
    /**
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "MINLP Rastrigin Function";
    }
    /// Extra informations
    /**
     * @return a string containing extra informations on the problem
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        ss << "\tMINLP continuous dimension: " << std::to_string(m_dim_c) << "\n";
        ss << "\tMINLP integer dimension: " << std::to_string(m_dim_i) << "\n";
        return ss.str();
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
        ar(m_dim_c, m_dim_i);
    }

private:
    /// Problem dimensions
    unsigned int m_dim_c;
    unsigned int m_dim_i;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::minlp_rastrigin)

#endif
