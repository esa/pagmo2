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

#ifndef PAGMO_PROBLEM_ACKLEY_HPP
#define PAGMO_PROBLEM_ACKLEY_HPP

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

/// The Ackley problem.
/**
 *
 * \image html ackley.png "Two-dimensional Ackley function." width=3cm
 *
 * This is a scalable box-constrained continuous single-objective problem.
 * The objective function is the generalised n-dimensional Ackley function:
 * \f[
 * 	F\left(x_1,\ldots,x_n\right) = 20 + e - 20e^{-\frac 15 \sqrt{\frac 1n \sum_{i=1}^n x_i^2}} - e^{\frac 1n
 * \sum_{i=1}^n \cos(2\pi x_i)}, \quad x_i \in \left[ -15,30 \right].
 * \f]
 * The global minimum is in \f$x_i=0\f$, where \f$ F\left( 0,\ldots,0 \right) = 0 \f$.
 */
struct ackley {
    /// Constructor from dimension
    /**
     * Constructs an Ackley problem
     *
     * @param dim the problem dimensions.
     *
     * @throw std::invalid_argument if \p dim is < 1
     */
    ackley(unsigned int dim = 1u) : m_dim(dim)
    {
        if (dim < 1u) {
            pagmo_throw(std::invalid_argument,
                        "Ackley Function must have minimum 1 dimension, " + std::to_string(dim) + " requested");
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
        double omega = 2. * detail::pi();
        double s1 = 0., s2 = 0.;
        double nepero = std::exp(1.0);

        for (decltype(n) i = 0u; i < n; i++) {
            s1 += x[i] * x[i];
            s2 += std::cos(omega * x[i]);
        }
        f[0] = -20 * std::exp(-0.2 * std::sqrt(1.0 / static_cast<double>(n) * s1))
               - std::exp(1.0 / static_cast<double>(n) * s2) + 20 + nepero;
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
        vector_double lb(m_dim, -15);
        vector_double ub(m_dim, 30);
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
        return "Ackley Function";
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

PAGMO_REGISTER_PROBLEM(pagmo::ackley)

#endif
