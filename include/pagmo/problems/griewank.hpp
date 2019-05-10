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

#ifndef PAGMO_PROBLEMS_GRIEWANK_HPP
#define PAGMO_PROBLEMS_GRIEWANK_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// The Griewank problem.
/**
 *
 * \image html griewank.png "Two-dimensional Griewank function." width=3cm
 *
 * This is a scalabale box-constrained continuous single-objective problem.
 * The objective function is the generalised n-dimensional Griewank function:
 * \f[
 * 	F\left(x_1,\ldots,x_n\right) = \sum_{i=1}^n x_i^2 / 4000 - \prod_{i=1}^n\cos\frac{x_i}{\sqrt{i}}, \quad x_i \in
 * \left[ -600,600 \right].
 * \f]
 * The global minimum is in \f$x_i=0\f$, where \f$ F\left( 0,\ldots,0 \right) = 0 \f$.
 */
struct PAGMO_DLL_PUBLIC griewank {
    /// Constructor from dimension
    /**
     * Constructs a Griewank problem
     *
     * @param dim the problem dimensions.
     *
     * @throw std::invalid_argument if \p dim is < 1
     */
    griewank(unsigned dim = 1u);

    // Fitness computation
    vector_double fitness(const vector_double &) const;

    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;

    /// Problem name
    /**
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Griewank Function";
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

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::griewank)

#endif
