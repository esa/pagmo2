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

#ifndef PAGMO_PROBLEM_GRIEWANK_HPP
#define PAGMO_PROBLEM_GRIEWANK_HPP

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp> // needed for cereal registration macro
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
struct griewank {
    /// Constructor from dimension
    /**
     * Constructs a Griewank problem
     *
     * @param dim the problem dimensions.
     *
     * @throw std::invalid_argument if \p dim is < 1
     */
    griewank(unsigned int dim = 1u) : m_dim(dim)
    {
        if (dim < 1u) {
            pagmo_throw(std::invalid_argument,
                        "Griewank Function must have minimum 1 dimension, " + std::to_string(dim) + " requested");
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
        auto n = x.size();
        double fr = 4000.;
        double retval = 0.;
        double p = 1.;

        for (decltype(n) i = 0u; i < n; i++) {
            retval += x[i] * x[i];
        }
        for (decltype(n) i = 0u; i < n; i++) {
            p *= std::cos(x[i] / std::sqrt(static_cast<double>(i) + 1.0));
        }
        f[0] = (retval / fr - p + 1.);
        return f;
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
        vector_double lb(m_dim, -600);
        vector_double ub(m_dim, 600);
        return {lb, ub};
    }
    /// Problem name
    /**
     *
     *
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Griewank Function";
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

PAGMO_REGISTER_PROBLEM(pagmo::griewank)

#endif
