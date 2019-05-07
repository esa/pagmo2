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

#ifndef PAGMO_PROBLEMS_DTLZ_HPP
#define PAGMO_PROBLEMS_DTLZ_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// DTLZ problem test suite.
/**
 * This widespread test suite was conceived for multiobjective problems with scalable fitness dimensions
 * and takes its name from its authors Deb, Thiele, Laumanns and Zitzler
 *
 * All problems in this test suite are box-constrained continuous n-dimensional multi-objective problems, scalable in
 * fitness dimension. The dimension of the decision space is \f$ k + fdim - 1 \f$, whereas fdim is the number of
 * objectives and k a paramter. Properties of the decision space and the Pareto-front of each problems are as follows:
 *
 * DTLZ1:
 *
 * The optimal pareto front lies on a linear hyperplane \f$ \sum_{m=1}^M f_m = 0.5 \f$ .
 *
 * DTLZ2:
 *
 * The search space is continous, unimodal and the problem is not deceptive.
 *
 * DTLZ3:
 *
 * The search space is continous, unimodal and the problem is not deceptive.
 * It is supposed to be harder to converge towards the optimal pareto front than DTLZ2
 *
 * DTLZ4:
 *
 * The search space contains a dense area of solutions next to the \f$ f_M / f_1\f$ plane.
 *
 * DTLZ5:
 *
 * This problem will test an MOEA's ability to converge to a cruve and will also allow an easier way to visually
 * demonstrate (just by plotting f_M with any other objective function) the performance of an MOEA. Since there is a
 * natural bias for solutions close to this Pareto-optimal curve, this problem may be easy for an algorithmn to solve.
 * Because of its simplicity its recommended to use a higher number of objectives \f$ M \in [5, 10]\f$.
 *
 * DTLZ6:
 *
 * A more difficult version of the DTLZ5 problem: the non-linear distance function g makes it harder to convergence
 * against the pareto optimal curve.
 *
 * DTLZ7:
 *
 * This problem has disconnected Pareto-optimal regions in the search space.
 *
 * See: K. Deb, L. Thiele, M. Laumanns, E. Zitzler, Scalable test problems for evolutionary multiobjective optimization
 */

class PAGMO_DLL_PUBLIC dtlz
{
public:
    /** Constructor
     *
     * Will construct a problem of the DTLZ test-suite.
     *
     * @param prob_id problem id
     * @param dim the problem dimension (size of the decision vector)
     * @param fdim number of objectives
     * @param alpha controls density of solutions (used only by DTLZ4)
     *
     * @throw std::invalid_argument if the prob_id is not in [1 .. 7], if fdim is less than 2 or if fdim or dim_param
     * are larger than an implementation defiend value
     *
     */
    dtlz(unsigned prob_id = 1u, vector_double::size_type dim = 5u, vector_double::size_type fdim = 3u,
         unsigned alpha = 100u);
    // Fitness computation
    vector_double fitness(const vector_double &) const;
    /// Number of objectives
    /**
     *
     * It returns the number of objectives.
     *
     * @return the number of objectives
     */
    vector_double::size_type get_nobj() const
    {
        return m_fdim;
    }
    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;
    // Distance from the Pareto front (of a population)
    double p_distance(const pagmo::population &) const;
    // Distance from the Pareto front
    double p_distance(const vector_double &) const;
    // Problem name
    std::string get_name() const;
    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Convergence metric for a dv (0 = converged to the optimal front)
    PAGMO_DLL_LOCAL double g_func(const vector_double &) const;

    // Implementations of the different g-functions used
    PAGMO_DLL_LOCAL double g13_func(const vector_double &) const;
    PAGMO_DLL_LOCAL double g245_func(const vector_double &) const;
    PAGMO_DLL_LOCAL double g6_func(const vector_double &) const;
    PAGMO_DLL_LOCAL double g7_func(const vector_double &) const;

    // Implementation of the distribution function h
    PAGMO_DLL_LOCAL double h7_func(const vector_double &, double) const;

    // Implementation of the objective functions.
    /* The chomosome: x_1, x_2, ........, x_M-1, x_M, .........., x_M+k
     *											 [------- Vector x_M -------]
     *               x[0], x[1], ... ,x[fdim-2], x[fdim-1], ... , x[fdim+k-1] */
    PAGMO_DLL_LOCAL vector_double f1_objfun_impl(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double f23_objfun_impl(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double f4_objfun_impl(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double f56_objfun_impl(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double f7_objfun_impl(const vector_double &) const;

    // Gives a convergence metric for the population (0 = converged to the optimal front)
    PAGMO_DLL_LOCAL double convergence_metric(const vector_double &) const;

    // Problem dimensions
    unsigned m_prob_id;
    // used only for DTLZ4
    unsigned m_alpha;
    // dimension parameter
    vector_double::size_type m_dim;
    // number of objectives
    vector_double::size_type m_fdim;
};
} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::dtlz)

#endif
