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

#ifndef PAGMO_PROBLEM_RASTRIGIN_HPP
#define PAGMO_PROBLEM_RASTRIGIN_HPP

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

/// The Rastrigin problem.
/**
 *
 * \image html rastrigin.png "Two-dimensional Rastrigin function." width=3cm
 *
 * This is a scalable box-constrained continuous single-objective problem.
 * The objective function is the generalised n-dimensional Rastrigin function:
 * \f[
 * 	F\left(x_1,\ldots,x_n\right) = 10 \cdot n + \sum_{i=1}^n x_i^2 - 10\cdot\cos\left( 2\pi \cdot x_i \right), \quad x_i
 * \in \left[ -5.12,5.12 \right].
 * \f]
 *
 * Gradients (dense) are also provided as:
 * \f[
 * 	G_i\left(x_1,\ldots,x_n\right) = 2 x_i + 10 \cdot 2\pi \cdot\sin\left( 2\pi \cdot x_i \right)
 * \f]
 * And Hessians (sparse as only the diagonal is non-zero) are:
 * \f[
 * 	H_{ii}\left(x_1,\ldots,x_n\right) = 2 + 10 \cdot 4\pi^2 \cdot\cos\left( 2\pi \cdot x_i \right)
 * \f]
 * The global minimum is in the origin, where \f$ F\left( 0,\ldots,0 \right) = 0 \f$.
 */
struct rastrigin {
    /// Constructor from dimension
    /**
     * Constructs a Rastrigin problem
     *
     * @param dim the problem dimensions.
     *
     * @throw std::invalid_argument if \p dim is < 1
     */
    rastrigin(unsigned int dim = 1u) : m_dim(dim)
    {
        if (dim < 1u) {
            pagmo_throw(std::invalid_argument,
                        "Rastrigin Function must have minimum 1 dimension, " + std::to_string(dim) + " requested");
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
        vector_double lb(m_dim, -5.12);
        vector_double ub(m_dim, 5.12);
        return {lb, ub};
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
        auto n = m_dim;
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
        return "Rastrigin Function";
    }
    /// Optimal solution
    /**
     * @return the decision vector corresponding to the best solution for this problem.
     */
    vector_double best_known() const
    {
        return vector_double(m_dim, 0.);
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
    /// Problem dimensions
    unsigned int m_dim;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::rastrigin)

#endif
