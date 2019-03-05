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

#ifndef PAGMO_PROBLEM_GOLOMB_RULER_HPP
#define PAGMO_PROBLEM_GOLOMB_RULER_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp> // needed for cereal registration macro
#include <pagmo/types.hpp>

namespace pagmo
{

/// The Golomb Ruler Problem
/**
 *
 * \image html golomb_ruler.png "The optimal Golomb Ruler of order 4" width=3cm
 *
 * In mathematics, a Golomb ruler is a set of marks at integer positions along an imaginary ruler such that no two pairs
 * of marks are the same distance apart. The number of marks on the ruler is its order, and the largest distance between
 * two of its marks is its length. Translation and reflection of a Golomb ruler are considered trivial, so the smallest
 * mark is customarily put at 0 and the next mark at the smaller of its two possible values.  There is no requirement
 * that a Golomb ruler be able to measure all distances up to its length, but if it does, it is called a perfect Golomb
 * ruler.
 *
 * A Golomb ruler is optimal if no shorter Golomb ruler of the same order exists. Creating Golomb rulers is easy,
 * but finding the optimal Golomb ruler (or rulers) for a specified order is computationally very challenging.
 *
 * This UDP represents the problem of finding an optimal Golomb ruler of a given order \f$ n\f$. A maximal distance \f$
 * l_{max}\f$ between consecutive marks is also specified to make the problem representation possible. The resulting
 * optimization problem is an integer programming problem with one equality constraint.
 *
 * In this UDP, the decision vector is \f$ x=[d_1, d_2, d_{n-1}]\f$, where the distances between consecutive ticks are
 * indicated with \f$ d_i\f$. The ticks on the ruler can then be reconstructed as \f$ a_0 = 0\f$, \f$ a_i = sum_{j=1}^i
 * d_i, i=1 .. n-1\f$
 *
 * Its formulation can thus be written as:
 *
 * \f[
 *   \begin{array}{rl}
 *   \mbox{find:} & 1 \le d_i \le l_{max}, \forall i=1..n-1 \\
 *   \mbox{to minimize: } & \sum_i d_i  \\
 *   \mbox{subject to:} & |a_i-a_j| \neq |a_l - a_m|, \forall (\mbox{distinct}) i,j,l,m \in [0, n]
 * \end{array}
 * \f]
 *
 * We transcribe the constraints as one single equality constraint: \f$ c = 0 \f$ where \f$ c\f$ is the count of
 * repeated distances.
 *
 * See: https://en.wikipedia.org/wiki/Golomb_ruler
 *
 */
class golomb_ruler
{
public:
    /// Constructor from ruler order and maximal consecutive ticks distance
    /**
     * Constructs a Golomb Ruler problem.
     *
     * @param order ruler order.
     * @param upper_bound maximum distance between consecutive ticks.
     *
     * @throw std::invalid_argument if \p order is < 2 or \p upper_bound is < 1.
     * @throw std::overflow_error if \p upper_bound is too large.
     */
    golomb_ruler(unsigned order = 3u, unsigned upper_bound = 10) : m_order(order), m_upper_bound(upper_bound)
    {
        if (order < 2u) {
            pagmo_throw(std::invalid_argument, "Golomb ruler problem must have at least order 2, while "
                                                   + std::to_string(order) + " was requested.");
        }
        if (upper_bound < 2u) {
            pagmo_throw(
                std::invalid_argument,
                "The upper bound for the maximum distance between consecutive ticks has to be at least 2, while "
                    + std::to_string(upper_bound) + " was requested.");
        }
        // Overflow can occur when evaluating the fitness later if the upper_bound is too large.
        if (upper_bound > std::numeric_limits<unsigned>::max() / (order - 1u)) {
            pagmo_throw(
                std::overflow_error,
                "Overflow in Golomb ruler problem, select a smaller maximum distance between consecutive ticks.");
        }
    }
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
        vector_double f(2, 0.);
        // 1 - We compute the ticks (the ruler length will be the last tick since we start from 0)
        vector_double ticks(x.size() + 1, 0.);
        std::partial_sum(x.begin(), x.end(), ticks.begin() + 1u);
        f[0] = ticks.back();
        // 2 - We compute all pairwise distances
        vector_double distances;
        distances.reserve(x.size() * (x.size() - 1) / 2);
        for (decltype(ticks.size()) i = 0; i < ticks.size() - 1; ++i) {
            for (decltype(ticks.size()) j = i + 1; j < ticks.size(); ++j) {
                distances.push_back(ticks[j] - ticks[i]);
            }
        }
        // 3 - We compute how many duplicate distances are there.
        std::sort(distances.begin(), distances.end(), detail::less_than_f<double>);
        f[1] = static_cast<double>(distances.size())
               - static_cast<double>(std::distance(
                     distances.begin(), std::unique(distances.begin(), distances.end(), detail::equal_to_f<double>)));
        return f;
    }
    /// Box-bounds
    /**
     * Returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        unsigned prob_dim = m_order - 1u;
        vector_double lb(prob_dim, 1);
        vector_double ub(prob_dim, m_upper_bound);
        return {lb, ub};
    }
    /// Equality constraint dimension
    /**
     * Returns the number of equality constraints
     *
     * @return the number of equality constraints
     */
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    /// Integer dimension
    /**
     * Returns the integer dimension of the problem.
     *
     * @return the integer dimension of the problem.
     */
    vector_double::size_type get_nix() const
    {
        return m_order - 1u;
    }
    /// Problem name
    /**
     * Returns the problem name.
     *
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        return "Golomb Ruler (order " + std::to_string(m_order) + ")";
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
        ar(m_order, m_upper_bound);
    }

private:
    /// Ruler Order.
    unsigned m_order;
    /// Maximum distance between consecutive ticks.
    unsigned m_upper_bound;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::golomb_ruler)

#endif
