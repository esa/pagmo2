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

#ifndef PAGMO_ALGORITHMS_CMAES_HPP
#define PAGMO_ALGORITHMS_CMAES_HPP

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_EIGEN3)

#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/detail/eigen.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>

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
class cmaes
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
    typedef std::tuple<unsigned int, unsigned long long, double, double, double, double> log_line_type;

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
    cmaes(unsigned int gen = 1, double cc = -1, double cs = -1, double c1 = -1, double cmu = -1, double sigma0 = 0.5,
          double ftol = 1e-6, double xtol = 1e-6, bool memory = false, bool force_bounds = false,
          unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_cc(cc), m_cs(cs), m_c1(c1), m_cmu(cmu), m_sigma0(sigma0), m_ftol(ftol), m_xtol(xtol),
          m_memory(memory), m_force_bounds(force_bounds), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
        if (((cc < 0.) || (cc > 1.)) && !(cc == -1)) {
            pagmo_throw(std::invalid_argument,
                        "cc must be in [0,1] or -1 if its value has to be initialized automatically, a value of "
                            + std::to_string(cc) + " was detected");
        }
        if (((cs < 0.) || (cs > 1.)) && !(cs == -1)) {
            pagmo_throw(std::invalid_argument,
                        "cs needs to be in [0,1] or -1 if its value has to be initialized automatically, a value of "
                            + std::to_string(cs) + " was detected");
        }
        if (((c1 < 0.) || (c1 > 1.)) && !(c1 == -1)) {
            pagmo_throw(std::invalid_argument,
                        "c1 needs to be in [0,1] or -1 if its value has to be initialized automatically, a value of "
                            + std::to_string(c1) + " was detected");
        }
        if (((cmu < 0.) || (cmu > 1.)) && !(cmu == -1)) {
            pagmo_throw(std::invalid_argument,
                        "cmu needs to be in [0,1] or -1 if its value has to be initialized automatically, a value of "
                            + std::to_string(cmu) + " was detected");
        }

        // Initialize explicitly the algorithm memory
        sigma = m_sigma0;
        mean = Eigen::VectorXd::Zero(1);
        variation = Eigen::VectorXd::Zero(1);
        newpop = std::vector<Eigen::VectorXd>{};
        B = Eigen::MatrixXd::Identity(1, 1);
        D = Eigen::MatrixXd::Identity(1, 1);
        C = Eigen::MatrixXd::Identity(1, 1);
        invsqrtC = Eigen::MatrixXd::Identity(1, 1);
        pc = Eigen::VectorXd::Zero(1);
        ps = Eigen::VectorXd::Zero(1);
        counteval = 0u;
        eigeneval = 0u;
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for a maximum number of generations, until one of
     * tolerances set on the population flatness (x_tol, f_tol) are met.
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained
     * @throws std::invalid_argument if the problem is unbounded
     * @throws std::invalid_argument if the population size is not at least 5
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed.
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto lam = pop.size();
        auto mu = lam / 2u;
        auto prob_f_dimension = prob.get_nf();
        auto fevals0 = prob.get_fevals(); // discount for the already made fevals
        auto count = 1u;                  // regulates the screen output

        // PREAMBLE--------------------------------------------------
        // Checks on the problem type
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob_f_dimension != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (lam < 5u) {
            pagmo_throw(std::invalid_argument, get_name() + " needs at least 5 individuals in the population, "
                                                   + std::to_string(lam) + " detected");
        }
        for (auto num : lb) {
            if (!std::isfinite(num)) {
                pagmo_throw(std::invalid_argument, "A " + std::to_string(num) + " is detected in the lower bounds, "
                                                       + this->get_name() + " cannot deal with it.");
            }
        }
        for (auto num : ub) {
            if (!std::isfinite(num)) {
                pagmo_throw(std::invalid_argument, "A " + std::to_string(num) + " is detected in the upper bounds, "
                                                       + this->get_name() + " cannot deal with it.");
            }
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // -----------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Initializing the random number generators
        std::uniform_real_distribution<double> randomly_distributed_number(0., 1.); // to generate a number in [0, 1)
        std::normal_distribution<double> normally_distributed_number(
            0., 1.); // to generate a normally distributed number        // Setting coefficients for Selection
        Eigen::VectorXd weights(_(mu));
        for (decltype(weights.rows()) i = 0; i < weights.rows(); ++i) {
            weights(i) = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i) + 1.);
        }
        weights /= weights.sum();                            // weights for the weighted recombination
        double mueff = 1. / (weights.transpose() * weights); // variance-effectiveness of sum w_i x_i

        // Setting coefficients for Adaptation automatically or to user defined data
        double cc(m_cc), cs(m_cs), c1(m_c1), cmu(m_cmu);
        double N = static_cast<double>(dim);
        if (cc == -1) {
            cc = (4. + mueff / N) / (N + 4. + 2. * mueff / N); // t-const for cumulation for C
        }
        if (cs == -1) {
            cs = (mueff + 2.) / (N + mueff + 5.); // t-const for cumulation for sigma control
        }
        if (c1 == -1) {
            c1 = 2. / ((N + 1.3) * (N + 1.3) + mueff); // learning rate for rank-one update of C
        }
        if (cmu == -1) {
            cmu = 2. * (mueff - 2. + 1. / mueff) / ((N + 2.) * (N + 2.) + mueff); // and for rank-mu update
        }

        double damps
            = 1. + 2. * std::max(0., std::sqrt((mueff - 1.) / (N + 1.)) - 1.) + cs; // damping coefficient for sigma
        double chiN
            = std::sqrt(N) * (1. - 1. / (4. * N) + 1. / (21. * N * N)); // expectation of ||N(0,I)|| == norm(randn(N,1))

        // Some buffers
        Eigen::VectorXd meanold = Eigen::VectorXd::Zero(_(dim));
        Eigen::MatrixXd Dinv = Eigen::MatrixXd::Identity(_(dim), _(dim));
        Eigen::MatrixXd Cold = Eigen::MatrixXd::Identity(_(dim), _(dim));
        Eigen::VectorXd tmp = Eigen::VectorXd::Zero(_(dim));
        std::vector<Eigen::VectorXd> elite(mu, tmp);
        vector_double dumb(dim, 0.);

        // If the algorithm is called for the first time on this problem dimension / pop size or if m_memory is false we
        // erease the memory of past calls
        if ((newpop.size() != lam) || (static_cast<unsigned int>(newpop[0].rows()) != dim) || (m_memory == false)) {
            sigma = m_sigma0;
            mean.resize(_(dim));
            auto idx_b = pop.best_idx();
            for (decltype(dim) i = 0u; i < dim; ++i) {
                mean(_(i)) = pop.get_x()[idx_b][i];
            }
            newpop = std::vector<Eigen::VectorXd>(lam, tmp);
            variation.resize(_(dim));

            // We define the starting B,D,C
            B = Eigen::MatrixXd::Identity(_(dim), _(dim)); // B defines the coordinate system
            D = Eigen::MatrixXd::Identity(_(dim), _(dim));
            // diagonal D defines the scaling. By default this is the witdh of the box bounds.
            // If this is too small... then 1e-6 is used
            for (decltype(dim) j = 0u; j < dim; ++j) {
                D(_(j), _(j)) = std::max((ub[j] - lb[j]), 1e-6);
            }
            C = Eigen::MatrixXd::Identity(_(dim), _(dim)); // covariance matrix C
            C = D * D;
            invsqrtC = Eigen::MatrixXd::Identity(_(dim), _(dim)); // inverse of sqrt(C)
            for (decltype(dim) j = 0; j < dim; ++j) {
                invsqrtC(_(j), _(j)) = 1. / D(_(j), _(j));
            }
            pc = Eigen::VectorXd::Zero(_(dim));
            ps = Eigen::VectorXd::Zero(_(dim));
            counteval = 0u;
            eigeneval = 0u;
        }

        if (m_verbosity > 0u) {
            std::cout << "CMAES 4 PaGMO: " << std::endl;
            std::cout << "mu: " << mu << " - lambda: " << lam << " - mueff: " << mueff << " - N: " << N << std::endl;
            std::cout << "cc: " << cc << " - cs: " << cs << " - c1: " << c1 << " - cmu: " << cmu
                      << " - sigma: " << sigma << " - damps: " << damps << " - chiN: " << chiN << std::endl;
        }

        // ----------------------------------------------//
        // HERE WE START THE JUICE OF THE ALGORITHM      //
        // ----------------------------------------------//
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_(dim));
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // 1 - We generate and evaluate lam new individuals
            for (decltype(lam) i = 0u; i < lam; ++i) {
                // 1a - we create a randomly normal distributed vector
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    tmp(_(j)) = normally_distributed_number(m_e);
                }
                // 1b - and store its transformed value in the newpop
                newpop[i] = mean + (sigma * B * D * tmp);
            }

            // 1bis - Check the exit conditions and logs
            // Exit condition on xtol
            {
                if ((sigma * B * D * tmp).norm() < m_xtol) {
                    if (m_verbosity > 0u) {
                        std::cout << "Exit condition -- xtol < " << m_xtol << std::endl;
                    }
                    return pop;
                }
                // Exit condition on ftol
                auto idx_b = pop.best_idx();
                auto idx_w = pop.worst_idx();
                double delta_f = std::abs(pop.get_f()[idx_b][0] - pop.get_f()[idx_w][0]);
                if (delta_f < m_ftol) {
                    if (m_verbosity) {
                        std::cout << "Exit condition -- ftol < " << m_ftol << std::endl;
                    }
                    return pop;
                }
            }

            // 1bis - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    // The population flattness in chromosome
                    auto dx = (sigma * B * D * tmp).norm();
                    // The population flattness in fitness
                    auto idx_b = pop.best_idx();
                    auto idx_w = pop.worst_idx();
                    auto df = std::abs(pop.get_f()[idx_b][0] - pop.get_f()[idx_w][0]);
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                              "Best:", std::setw(15), "dx:", std::setw(15), "df:", std::setw(15), "sigma:", '\n');
                    }
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                          pop.get_f()[idx_b][0], std::setw(15), dx, std::setw(15), df, std::setw(15), sigma, '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(gen, prob.get_fevals() - fevals0, pop.get_f()[idx_b][0], dx, df, sigma);
                }
            }
            // 2 - We fix the bounds.
            // Note that this screws up the whole covariance matrix machinery and worsen
            // performances considerably.
            if (m_force_bounds) {
                for (decltype(lam) i = 0u; i < lam; ++i) {
                    for (decltype(dim) j = 0u; j < dim; ++j) {
                        if (newpop[i](_(j)) < lb[j]) {
                            newpop[i](_(j)) = lb[j];
                        } else if (newpop[i](_(j)) > ub[j]) {
                            newpop[i](_(j)) = ub[j];
                        }
                    }
                }
            }
            // 3 - We Evaluate the new population (if the problem is stochastic change seed first)
            if (prob.is_stochastic()) {
                // change the problem seed. This is done via the population_set_seed method as prob.set_seed
                // is forbidden being prob a const ref.
                pop.get_problem().set_seed(std::uniform_int_distribution<unsigned int>()(m_e));
            }
            // Reinsertion
            for (decltype(lam) i = 0u; i < lam; ++i) {
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    dumb[j] = newpop[i](_(j));
                }
                pop.set_x(i, dumb);
            }
            counteval += lam;
            // 4 - We extract the elite from this generation.
            std::vector<population::size_type> best_idx(lam);
            std::iota(best_idx.begin(), best_idx.end(), population::size_type(0));
            std::sort(best_idx.begin(), best_idx.end(), [&pop](population::size_type idx1, population::size_type idx2) {
                return detail::less_than_f(pop.get_f()[idx1][0], pop.get_f()[idx2][0]);
            });
            best_idx.resize(mu); // not needed?
            for (decltype(mu) i = 0u; i < mu; ++i) {
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    elite[i](_(j)) = pop.get_x()[best_idx[i]][j];
                }
            }
            // 5 - Compute the new mean of the elite storing the old one
            meanold = mean;
            mean = elite[0] * weights(0);
            for (decltype(mu) i = 1u; i < mu; ++i) {
                mean += elite[i] * weights(_(i));
            }
            // 6 - Update evolution paths
            ps = (1. - cs) * ps + std::sqrt(cs * (2. - cs) * mueff) * invsqrtC * (mean - meanold) / sigma;
            double hsig = 0.;
            hsig = (ps.squaredNorm() / N
                    / (1. - std::pow((1. - cs), (2. * static_cast<double>(counteval) / static_cast<double>(lam)))))
                   < (2. + 4. / (N + 1.));
            pc = (1. - cc) * pc + hsig * std::sqrt(cc * (2. - cc) * mueff) * (mean - meanold) / sigma;
            // 7 - Adapt Covariance Matrix
            Cold = C;
            C = (elite[0] - meanold) * (elite[0] - meanold).transpose() * weights(0);
            for (decltype(mu) i = 1u; i < mu; ++i) {
                C += (elite[i] - meanold) * (elite[i] - meanold).transpose() * weights(_(i));
            }
            C /= sigma * sigma;
            C = (1. - c1 - cmu) * Cold + cmu * C + c1 * ((pc * pc.transpose()) + (1. - hsig) * cc * (2. - cc) * Cold);
            // 8 - Adapt sigma
            sigma *= std::exp(std::min(0.6, (cs / damps) * (ps.norm() / chiN - 1.)));
            // 9 - Perform eigen-decomposition of C
            if (static_cast<double>(counteval - eigeneval)
                > (static_cast<double>(lam) / (c1 + cmu) / N / 10.)) { // achieve O(N^2)
                eigeneval = counteval;
                C = (C + C.transpose()) / 2.; // enforce symmetry
                es.compute(C);                // eigen decomposition
                if (es.info() == Eigen::Success) {
                    B = es.eigenvectors();
                    D = es.eigenvalues().asDiagonal();
                    for (decltype(dim) j = 0u; j < dim; ++j) {
                        D(_(j), _(j)) = std::sqrt(std::max(1e-20, D(_(j), _(j)))); // D contains standard deviations now
                    }
                    for (decltype(dim) j = 0u; j < dim; ++j) {
                        Dinv(_(j), _(j)) = 1. / D(_(j), _(j));
                    }
                    invsqrtC = B * Dinv * B.transpose();
                } // if eigendecomposition fails just skip it and keep pevious successful one.
            }
        } // end of generation loop
        if (m_verbosity) {
            std::cout << "Exit condition -- generations = " << m_gen << std::endl;
        }
        return pop;
    }
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned int seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    };
    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned int get_seed() const
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
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }
    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned int get_gen() const
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
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tGenerations: ", m_gen);
        stream(ss, "\n\tcc: ");
        if (m_cc == -1)
            stream(ss, "auto");
        else
            stream(ss, m_cc);
        stream(ss, "\n\tcs: ");
        if (m_cs == -1)
            stream(ss, "auto");
        else
            stream(ss, m_cs);
        stream(ss, "\n\tc1: ");
        if (m_c1 == -1)
            stream(ss, "auto");
        else
            stream(ss, m_c1);
        stream(ss, "\n\tcmu: ");
        if (m_cmu == -1)
            stream(ss, "auto");
        else
            stream(ss, m_cmu);
        stream(ss, "\n\tsigma0: ", m_sigma0);
        stream(ss, "\n\tStopping xtol: ", m_xtol);
        stream(ss, "\n\tStopping ftol: ", m_ftol);
        stream(ss, "\n\tMemory: ", m_memory);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        stream(ss, "\n\tForce bounds: ", m_force_bounds);
        stream(ss, "\n\tSeed: ", m_seed);
        return ss.str();
    }
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
        ar(m_gen, m_cc, m_cs, m_c1, m_cmu, m_sigma0, m_ftol, m_xtol, m_memory, m_force_bounds, sigma, mean, variation,
           newpop, B, D, C, invsqrtC, pc, ps, counteval, eigeneval, m_e, m_seed, m_verbosity, m_log);
    }

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
    unsigned int m_gen;
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
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::cmaes)

#else // PAGMO_WITH_EIGEN3

#error The cmaes.hpp header was included, but pagmo was not compiled with eigen3 support

#endif // PAGMO_WITH_EIGEN3

#endif
