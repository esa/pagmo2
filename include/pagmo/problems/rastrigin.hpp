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

#ifndef PAGMO_PROBLEMS_RASTRIGIN_HPP
#define PAGMO_PROBLEMS_RASTRIGIN_HPP

#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// The Rastrigin problem.
/**
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
struct PAGMO_DLL_PUBLIC rastrigin {
    /// Constructor from dimension
    /**
     * Constructs a Rastrigin problem
     *
     * @param dim the problem dimensions.
     *
     * @throw std::invalid_argument if \p dim is < 1
     */
    rastrigin(unsigned dim = 1u);

    // Fitness computation
    vector_double fitness(const vector_double &) const;

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
        return "Rastrigin Function";
    }

    // Optimal solution
    vector_double best_known() const;

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

    /// Problem dimensions
    unsigned m_dim;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::rastrigin)

#endif
