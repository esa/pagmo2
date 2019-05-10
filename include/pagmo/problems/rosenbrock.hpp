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

#ifndef PAGMO_PROBLEMS_ROSENBROCK_HPP
#define PAGMO_PROBLEMS_ROSENBROCK_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// The Rosenbrock problem.
/**
 * \image html rosenbrock.png "Two-dimensional Rosenbrock function." width=3cm
 *
 * This is a box-constrained continuous single-objective problem.
 * The objective function is the generalised n-dimensional Rosenbrock function:
 * \f[
 *  F\left(x_1,\ldots,x_n\right) =
 *  \sum_{i=1}^{n-1}\left[ 100\left(x_i^2-x_{i+1}\right)^2+\left(x_i-1\right)^2\right], \quad x_i \in \left[ -5,10
 * \right].
 * \f]
 * The global minimum is in \f$x_i=1\f$, where \f$ F\left( 1,\ldots,1 \right) = 0 \f$.
 */
struct PAGMO_DLL_PUBLIC rosenbrock {
    /// Constructor from dimension
    /**
     * @param dim problem dimension.
     *
     * @throw std::invalid_argument if \p dim is less than 2.
     */
    rosenbrock(vector_double::size_type dim = 2u);
    // Fitness computation
    vector_double fitness(const vector_double &) const;

    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;
    /// Problem name
    /**
     * @return a string containing the problem name.
     */
    std::string get_name() const
    {
        return "Multidimensional Rosenbrock Function";
    }
    // Gradient.
    vector_double gradient(const vector_double &) const;
    // Optimal solution.
    vector_double best_known() const;
    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);
    /// Thread safety level.
    /**
     * @return the ``constant`` thread safety level.
     */
    thread_safety get_thread_safety() const
    {
        return thread_safety::constant;
    }
    /// Problem dimensions
    vector_double::size_type m_dim;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::rosenbrock)

#endif
