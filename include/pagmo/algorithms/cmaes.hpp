/* Copyright 2017-2020 PaGMO development team

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

#ifndef PAGMO_ALGORITHMS_CMAES_HPP
#define PAGMO_ALGORITHMS_CMAES_HPP

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_EIGEN3)

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/eigen.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{
/// Covariance Matrix Adaptation Evolutionary Strategy
/**
 * \image html cmaes.png "CMA-ES logic." width=3cm
 *
 * CMA-ES is one of the most successful algorithm, classified as an Evolutionary Strategy, for derivative-free global
 * optimization. The version implemented in PaGMO is the "classic" version described in the 2006 paper titled
 * "The CMA evolution strategy: a comparing review.".
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from pagmo::cmaes is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * .. note::
 *
 *    This user-defined algorithm is available only if pagmo was compiled with the ``PAGMO_WITH_EIGEN3`` option
 *    enabled (see the :ref:`installation instructions <install>`).
 *
 * .. note::
 *    Since at each generation all newly generated individuals sampled from the adapted distribution are
 *    reinserted into the population, CMA-ES may not preserve the best individual (not elitist). As a consequence the
 *    plot of the population best fitness may not be perfectly monotonically decreasing.
 *
 * .. seealso::
 *
 *    Hansen, Nikolaus. "The CMA evolution strategy: a comparing review." Towards a new evolutionary computation.
 *    Springer Berlin Heidelberg, 2006. 75-102.
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC cmaes
{
public:
    /// Single data line for the algorithm's log.
    /**
     * A log data line is a tuple consisting of:
     * - the generation number,
     * - the number of function evaluations
     * - the best fitness vector so far,
     * - the population flatness evaluated as the distance between the decisions vector of the best and of the worst
     * individual,
     * - the population flatness evaluated as the distance between the fitness of the best and of the worst individual.
     */
    typedef std::tuple<unsigned, unsigned long long, double, double, double, double> log_line_type;

    /// Log type.
    /**
     * The algorithm log is a collection of cmaes::log_line_type data lines, stored in chronological order
     * during the optimisation if the verbosity of the algorithm is set to a nonzero value
     * (see cmaes::set_verbosity()).
     */
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Constructs cmaes
     *
     * @param gen number of generations.
     * @param cc backward time horizon for the evolution path (by default is automatically assigned)
     * @param cs makes partly up for the small variance loss in case the indicator is zero (by default is
     automatically assigned)
     * @param c1  learning rate for the rank-one update of the covariance matrix (by default is automatically
     assigned)
     * @param cmu learning rate for the rank-\f$\mu\f$  update of the covariance matrix (by default is automatically
     assigned)
     * @param sigma0 initial step-size
     * @param ftol stopping criteria on the x tolerance (default is 1e-6)
     * @param xtol stopping criteria on the f tolerance (default is 1e-6)
     * @param memory when true the adapted parameters are not reset between successive calls to the evolve method
     * @param force_bounds when true the box bounds are enforced. The fitness will never be called outside the bounds
     but the covariance matrix adaptation  mechanism will worsen
     * @param seed seed used by the internal random number generator (default is random)
     * @throws std::invalid_argument if cc, cs, c1 and cmu are not in [0, 1]
     */
    cmaes(unsigned gen = 1, double cc = -1, double cs = -1, double c1 = -1, double cmu = -1, double sigma0 = 0.5,
          double ftol = 1e-6, double xtol = 1e-6, bool memory = false, bool force_bounds = false,
          unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

    // Sets the seed
    void set_seed(unsigned);

    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned get_seed() const
    {
        return m_seed;
    }

    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:      Fevals:          Best:            dx:            df:         sigma:
     * 51           1000    1.15409e-06     0.00205151    3.38618e-05       0.138801
     * 52           1020     3.6735e-07     0.00423372    2.91669e-05        0.13002
     * 53           1040     3.7195e-07    0.000655583    1.04182e-05       0.107739
     * 54           1060    6.26405e-08     0.00181163    3.86002e-06      0.0907474
     * 55           1080    4.09783e-09    0.000714699    3.57819e-06      0.0802022
     * 56           1100    1.77896e-08    4.91136e-05    9.14752e-07       0.075623
     * 57           1120    7.63914e-09    0.000355162    1.10134e-06      0.0750457
     * 58           1140    1.35199e-09    0.000356034    2.65614e-07      0.0622128
     * 59           1160    8.24796e-09    0.000695454    1.14508e-07        0.04993
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness
     * function currently in the population, dx is the norm of the distance to the population mean of
     * the mutant vectors, df is the population flatness evaluated as the distance between the fitness
     * of the best and of the worst individual and sigma is the current step-size
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    }

    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }

    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned get_gen() const
    {
        return m_gen;
    }

    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a cmaes::log_line_type containing: Gen, Fevals, Best, dx, df, sigma
     * as described in cmaes::set_verbosity
     * @return an <tt>std::vector</tt> of cmaes::log_line_type containing the logged values Gen, Fevals, Best, dx, df,
     * sigma
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Eigen stores indexes and sizes as signed types, while PaGMO
    // uses STL containers thus sizes and indexes are unsigned. To
    // make the conversion as painless as possible this template is provided
    // allowing, for example, syntax of the type D(_(i),_(j)) to adress an Eigen matrix
    // when i and j are unsigned
    template <typename I>
    static Eigen::DenseIndex _(I n)
    {
        return static_cast<Eigen::DenseIndex>(n);
    }

    // Data members
    unsigned m_gen;
    double m_cc;
    double m_cs;
    double m_c1;
    double m_cmu;
    double m_sigma0;
    double m_ftol;
    double m_xtol;
    bool m_memory;
    bool m_force_bounds;

    // "Memory" data members (these are adapted during each evolve call and may be remembered if m_memory is true)
    mutable double sigma;
    mutable Eigen::VectorXd mean;
    mutable Eigen::VectorXd variation;
    mutable std::vector<Eigen::VectorXd> newpop;
    mutable Eigen::MatrixXd B;
    mutable Eigen::MatrixXd D;
    mutable Eigen::MatrixXd C;
    mutable Eigen::MatrixXd invsqrtC;
    mutable Eigen::VectorXd pc;
    mutable Eigen::VectorXd ps;
    mutable population::size_type counteval;
    mutable population::size_type eigeneval;

    // "Common" data members
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::cmaes)

#else // PAGMO_WITH_EIGEN3

#error The cmaes.hpp header was included, but pagmo was not compiled with eigen3 support

#endif // PAGMO_WITH_EIGEN3

#endif
