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

#ifndef PAGMO_PROBLEMS_ZDT_HPP
#define PAGMO_PROBLEMS_ZDT_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// ZDT problem test suite
/**
 *
 * This widespread test suite was conceived for two-objective problems and takes its name from its
 * authors Zitzler, Deb and Thiele.
 *
 * In their paper the authors propose a set of 6 different scalable problems all originating from a
 * well thought combination of functions allowing, by construction, to measure the distance of
 * any point to the Pareto front while creating interesting problems. They also suggest some
 * dimensions for instantiating the problems, namely \f$m = [30, 30, 30, 10, 11, 10]\f$.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The ZDT5 problem is an integer problem, its fitness is computed rounding all the chromosome values,
 *    so that [1,0,1] or [0.97, 0.23, 0.57] will have the same fitness. Integer relaxation techniques are
 *    thus not appropriate fot this type of fitness.
 *
 * .. seealso::
 *
 *    Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary algorithms:
 *    Empirical results." Evolutionary computation 8.2 (2000): 173-195. doi: 10.1.1.30.5848
 *
 * \endverbatim
 *
 * ZDT1:
 *
 * This is a box-constrained continuous \f$n\f$-dimensional (\f$n\f$>1) multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *  g\left(x\right) = 1 + 9 \left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right) \\
 *  F_1 \left(x\right) = x_1 \\
 *  F_2 \left(x\right) = g(x) \left[ 1 - \sqrt{x_1 / g(x)} \right]  x \in \left[ 0,1 \right].
 * \end{array}
 * \f]
 *
 * ZDT2:
 *
 * This is a box-constrained continuous \f$n\f$-dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 1 + 9 \left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right) \\
 *      F_1 \left(x\right) = x_1 \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - \left(x_1 / g(x)\right)^2 \right]  x \in \left[ 0,1 \right].
 * \end{array}
 * \f]
 *
 * ZDT3:
 *
 * This is a box-constrained continuous \f$n\f$-dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 1 + 9 \left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right) \\
 *      F_1 \left(x\right) = x_1 \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - \sqrt{x_1 / g(x)} - x_1/g(x) \sin(10 \pi x_1) \right]  x \in \left[ 0,1
 * \right].
 * \end{array}
 * \f]
 *
 * ZDT4:
 *
 * This is a box-constrained continuous \f$n\f$-dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 91 + \sum_{i=2}^{n} \left[x_i^2 - 10 \cos \left(4\pi x_i \right) \right] \\
 *      F_1 \left(x\right) = x_1 \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - \sqrt{x_1 / g(x)} \right]  x_1 \in [0,1], x_i \in \left[ -5,5 \right] i=2,
 * \cdots, 10.
 * \end{array}
 * \f]
 *
 * ZDT5
 *
 * This is a box-constrained integer \f$n\f$-dimension multi-objecive problem. The chromosome is
 * a bitstring so that \f$x_i \in \left\{0,1\right\}\f$. Refer to the original paper for the formal definition.
 *
 * ZDT6
 *
 * This is a box-constrained continuous \f$n\f$--dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 1 + 9 \left[\left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right)\right]^{0.25} \\
 *      F_1 \left(x\right) = 1 - \exp(-4 x_1) \sin^6(6 \pi \ x_1) \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - (f_1(x) / g(x))^2  \right]  x \in \left[ 0,1 \right].
 * \end{array}
 * \f]
 */

class PAGMO_DLL_PUBLIC zdt
{
public:
    /** Constructor
     *
     * Will construct one problem from the ZDT test-suite.
     *
     * @param prob_id problem number. Must be in [1, .., 6]
     * @param param problem parameter, representing the problem dimension
     * except for ZDT5 where it represents the number of binary strings
     *
     * @throws std::invalid_argument if \p id is not in [1,..,6]
     * @throws std::invalid_argument if \p param is not at least 2.
     */
    zdt(unsigned prob_id = 1u, unsigned param = 30u);
    // Fitness computation
    vector_double fitness(const vector_double &) const;
    /// Number of objectives
    /**
     * It returns the number of objectives.
     *
     * @return the number of objectives
     */
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }

    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;

    // Integer dimension
    vector_double::size_type get_nix() const;

    // Problem name
    std::string get_name() const;

    // Distance from the Pareto front (of a population)
    double p_distance(const population &) const;
    /// Distance from the Pareto front
    /**
     * Convergence metric for a given decision_vector (0 = on the optimal front)
     *
     * Introduced by Martens and Izzo, this metric is able
     * to measure "a distance" of any point from the pareto front of any ZDT
     * problem analytically without the need to precompute the front.
     *
     * @param x input decision vector
     * @return the p_distance
     *
     * See: Märtens, Marcus, and Dario Izzo. "The asynchronous island model
     * and NSGA-II: study of a new migration operator and its performance."
     * Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013.
     */
    double p_distance(const vector_double &) const;

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL vector_double zdt1_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double zdt2_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double zdt3_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double zdt4_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double zdt5_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double zdt6_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL double zdt123_p_distance(const vector_double &) const;
    PAGMO_DLL_LOCAL double zdt4_p_distance(const vector_double &) const;
    PAGMO_DLL_LOCAL double zdt5_p_distance(const vector_double &) const;
    PAGMO_DLL_LOCAL double zdt6_p_distance(const vector_double &) const;

private:
    // Problem dimensions
    unsigned m_prob_id;
    unsigned m_param;
};
} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::zdt)

#endif
