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

#ifndef PAGMO_PROBLEMS_MINLP_RASTRIGIN_HPP
#define PAGMO_PROBLEMS_MINLP_RASTRIGIN_HPP

#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
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
struct PAGMO_DLL_PUBLIC minlp_rastrigin {
    /// Constructor from continuous and integer dimension
    /**
     * Constructs a MINLP Rastrigin problem.
     *
     * @param dim_c the problem continuous dimension.
     * @param dim_i the problem continuous dimension.
     *
     * @throw std::invalid_argument if \p dim_c+ \p dim_i is < 1
     */
    minlp_rastrigin(unsigned dim_c = 1u, unsigned dim_i = 1u);

    // Fitness computation
    vector_double fitness(const vector_double &) const;

    // Box-bounds
    /**
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const;

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
        return "MINLP Rastrigin Function";
    }

    // Extra info
    std::string get_extra_info() const;

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Problem dimensions
    unsigned m_dim_c;
    unsigned m_dim_i;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::minlp_rastrigin)

#endif
