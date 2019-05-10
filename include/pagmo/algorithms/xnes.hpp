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

#ifndef PAGMO_ALGORITHMS_XNES_HPP
#define PAGMO_ALGORITHMS_XNES_HPP

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
/// Exponential Natural Evolution Strategies
/**
 * \image html xnes.png width=3cm
 *
 * Exponential Natural Evolution Strategies is an algorithm closely related to pagmo::cmaes and based
 * on the adaptation of a gaussian sampling distribution via the so-called natural gradient.
 * Like pagmo::cmaes it is based on the idea of sampling new trial vectors from a multivariate distribution
 * and using the new sampled points to update the distribution parameters. Naively this could be done following
 * the gradient of the expected fitness as approximated by a finite number of sampled points. While this idea
 * offers a powerful lead on algorithmic construction it has some major drawbacks that are solved in the so-called
 * Natural Evolution Strategies class of algorithms by adopting, instead, the natural gradient. xNES is one of
 * the most performing variants in this class.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from pagmo::xnes is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * .. note::
 *
 *    This user-defined algorithm is available only if pagmo was compiled with the ``PAGMO_WITH_EIGEN3`` option
 *    enabled (see the :ref:`installation instructions <install>`).
 *
 * .. note::
 *
 *    We introduced one change to the original algorithm in order to simplify its use for the generic user.
 *    The initial covariance matrix depends on the bounds width so that heterogenously scaled variables
 *    are not a problem: the width along the i-th direction will be w_i = sigma_0 * (ub_i - lb_i)
 *
 * .. note::
 *
 *    Since at each generation all newly generated individuals sampled from the adapted distribution are
 *    reinserted into the population, xNES may not preserve the best individual (not elitist).
 *    As a consequence the plot of the population best fitness may not be perfectly monotonically decreasing.
 *
 * .. seealso::
 *
 *    Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., & Schmidhuber, J. (2010, July). Exponential natural
 *    evolution strategies. In Proceedings of the 12th annual conference on Genetic and evolutionary computation (pp.
 *    393-400). ACM.
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC xnes
{
public:
    /// Single data line for the algorithm's log.
    /**
     * A log data line is a tuple consisting of:
     * - the generation number,
     * - the number of function evaluations
     * - the best fitness vector so far,
     * - the population flatness evaluated as the distance between the decisions vector of the best and of the worst
     *   individual,
     * - the population flatness evaluated as the distance between the fitness of the best and of the worst individual.
     */
    typedef std::tuple<unsigned, unsigned long long, double, double, double, double> log_line_type;

    /// Log type.
    /**
     * The algorithm log is a collection of xnes::log_line_type data lines, stored in chronological order
     * during the optimisation if the verbosity of the algorithm is set to a nonzero value
     * (see xnes::set_verbosity()).
     */
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Constructs xnes
     *
     * @param gen number of generations.
     * @param eta_mu learning rate for mean update (if -1 will be automatically selected to be 1)
     * @param eta_sigma learning rate for step-size update (if -1 will be automatically selected)
     * @param eta_b  learning rate for the covariance matrix update (if -1 will be automatically selected)
     * @param sigma0 the initial search width will be sigma0 * (ub - lb) (if -1 will be selected to be 0.5)
     * @param ftol stopping criteria on the x tolerance (default is 1e-6)
     * @param xtol stopping criteria on the f tolerance (default is 1e-6)
     * @param memory when true the distribution parameters are not reset between successive calls to the evolve method
     * @param force_bounds when true the box bounds are enforced. The fitness will never be called outside the
     *        bounds but the covariance matrix adaptation  mechanism will worsen
     * @param seed seed used by the internal random number generator (default is random)

     * @throws std::invalid_argument if eta_mu, eta_sigma, eta_b and sigma0 are not in ]0, 1] or -1
     */
    xnes(unsigned gen = 1, double eta_mu = -1, double eta_sigma = -1, double eta_b = -1, double sigma0 = -1,
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
        return "xNES: Exponential Natural Evolution Strategies";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a xnes::log_line_type containing: Gen, Fevals, Best, dx, df, sigma
     * as described in xnes::set_verbosity
     * @return an <tt>std::vector</tt> of xnes::log_line_type containing the logged values Gen, Fevals, Best, dx, df,
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

    // "Real" data members
    unsigned m_gen;
    double m_eta_mu;
    double m_eta_sigma;
    double m_eta_b;
    double m_sigma0;
    double m_ftol;
    double m_xtol;
    bool m_memory;
    bool m_force_bounds;

    // "Memory" data members (these are adapted during each evolve call and may be remembered if m_memory is true)
    mutable double sigma;
    mutable Eigen::VectorXd mean;
    mutable Eigen::MatrixXd A;

    // "Common" data members
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::xnes)

#else // PAGMO_WITH_EIGEN3

#error The xnes.hpp header was included, but pagmo was not compiled with eigen3 support

#endif // PAGMO_WITH_EIGEN3

#endif
