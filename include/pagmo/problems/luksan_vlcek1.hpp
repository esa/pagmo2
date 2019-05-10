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

#ifndef PAGMO_PROBLEMS_LUKSAN_VLCECK1_HPP
#define PAGMO_PROBLEMS_LUKSAN_VLCECK1_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
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
struct PAGMO_DLL_PUBLIC luksan_vlcek1 {
    /// Constructor from dimension and bounds
    /**
     * Constructs the luksan_vlcek1 problem.
     *
     * @param dim the problem dimensions.
     *
     * @throw std::invalid_argument if \p dim is < 3.
     */
    luksan_vlcek1(unsigned dim = 3u);

    // Fitness computation
    vector_double fitness(const vector_double &) const;

    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;

    /// Equality constraint dimension
    /**
     * @return the number of equality constraints.
     */
    vector_double::size_type get_nec() const
    {
        return m_dim - 2u;
    }

    // Gradients
    vector_double gradient(const vector_double &) const;

    // Gradients sparsity
    sparsity_pattern gradient_sparsity() const;

    /// Problem name
    /**
     * @return a string containing the problem name.
     */
    std::string get_name() const
    {
        return "luksan_vlcek1";
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

    /// Problem dimensions.
    unsigned m_dim;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::luksan_vlcek1)

#endif
