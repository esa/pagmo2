/* Copyright 2017 PaGMO development team

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

#ifndef PAGMO_ALGORITHMS_SIMULATED_ANNEALING_HPP
#define PAGMO_ALGORITHMS_SIMULATED_ANNEALING_HPP

#include <cmath> //std::is_finite
#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../rng.hpp"

namespace pagmo
{

/// Simulated Annealing, Corana's version with adaptive neighbourhood.
/**
 * \image html Hill_Climbing_with_Simulated_Annealing.gif
 *
 * This version of the simulated annealing algorithm is, essentially, an iterative random search
 * procedure with adaptive moves along the coordinate directions. It permits uphill moves under
 * the control of metropolis criterion, in the hope to avoid the first local minima encountered.
 *
 * The implementation provided here allows to obtain a reannealing procedure via subsequent calls
 * to the pagmo::simulated_annealing::evolve() method.
 *
 *
 * **NOTE** When selecting the starting and final temperature values it helps to think about the tempertaure
 * as the deterioration in the objective function value that still has a 37% chance of being accepted.
 *
 * **NOTE** The algorithm does not work for multi-objective problems, stochastic problems nor for
 * constrained problems
 *
 * **NOTE** At each call of the evolve method the number of function evaluations is guaranteed to be less
 * than \p max_iter as when a point is produced out of the bounds that iteration is skipped
 *
 * @see Corana, A., Marchesi, M., Martini, C., & Ridella, S. (1987). Minimizing multimodal
 * functions of continuous variables with the “simulated annealing” algorithm Corrigenda
 * for this article is available here. ACM Transactions on Mathematical Software (TOMS), 13(3), 262-280.
 * http://people.sc.fsu.edu/~inavon/5420a/corana.pdf
 */
class simulated_annealing
{
public:
    /// Single entry of the log (fevals, best, current, avg_range, temperature)
    typedef std::tuple<unsigned int, double, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs simulated_annealing
     *
     * @param Ts starting temperature
     * @param Tf final temperature
     * @param n_T_adj number of temperature adjustments in the annealing schedule
     * @param n_range_adj number of adjustments of the search range performed at a constant temperature
     * @param bin_size number of mutations that are used to compute the acceptance rate
     * @param start_range starting range for mutating the decision vector
     * @param seed seed used by the internal random number generator
     */
    simulated_annealing(double Ts = 10., double Tf = .1, unsigned int n_T_adj = 10u, unsigned int n_range_adj = 1u,
                        unsigned int bin_size = 20u, double start_range = 1.,
                        unsigned int seed = pagmo::random_device::next())
        : m_Ts(Ts), m_Tf(Tf), m_n_T_adj(n_T_adj), m_n_range_adj(n_range_adj), m_bin_size(bin_size),
          m_start_range(start_range), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
        if (Ts <= 0. || !std::isfinite(Ts)) {
            pagmo_throw(std::invalid_argument, "The starting temperature must be finite and positive, while a value of "
                                                   + std::to_string(Ts) + " was detected.");
        }
        if (Tf <= 0. || !std::isfinite(Tf)) {
            pagmo_throw(std::invalid_argument, "The final temperature must be finite and positive, while a value of "
                                                   + std::to_string(Tf) + " was detected.");
        }
        if (start_range <= 0. || start_range > 1.) {
            pagmo_throw(std::invalid_argument, "The start range must be in (0,1], while a value of "
                                                   + std::to_string(start_range) + " was detected.");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained
     */
    population evolve(population pop) const
    {
        // We store some useful properties
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();             // not const as used type for counters
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto fevals0 = prob.get_fevals(); // disount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.get_nf() != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        if (pop.size() < 1u) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 1 individual in the population, "
                                                   + std::to_string(pop.size()) + " detected");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)

        // Starting point is the best individual
        auto best_idx = pop.best_idx();
        const auto &x0 = pop.get_x()[best_idx];
        const auto &fit0 = pop.get_f()[best_idx];
        // Determines the coefficient to decrease the temperature
        const double Tcoeff = std::pow(m_Tf / m_Ts, 1.0 / static_cast<double>(m_n_T_adj));
        // Stores the current and new points
        auto xNEW = x0;
        auto xOLD = x0;
        auto best_x = x0;
        auto fNEW = fit0;
        auto fOLD = fit0;
        auto best_f = fit0;

        // Stores the adaptive ranges for each component
        vector_double step(dim, m_start_range);

        // Stores the number of accepted points for each component
        std::vector<int> acp(dim, 0u);
        double ratio = 0., currentT = m_Ts, probab = 0.;

        // Main SA loops
        for (decltype(m_n_T_adj) jter = 0u; jter < m_n_T_adj; ++jter) {
            for (decltype(m_n_range_adj) mter = 0u; mter < m_n_range_adj; ++mter) {
                for (decltype(m_bin_size) kter = 0u; kter < m_bin_size; ++kter) {
                    auto nter = std::uniform_int_distribution<vector_double::size_type>(0u, dim - 1u)(m_e);
                    for (decltype(dim) numb = 0u; numb < dim; ++numb) {
                        nter = (nter + 1u) % dim;
                        // We modify the current point by mutating its nter component within
                        // a step that we will later adapt
                        xNEW[nter]
                            = xOLD[nter]
                              + std::uniform_real_distribution<>(-1, 1)(m_e) * step[nter] * (ub[nter] - lb[nter]);

                        // If new solution produced is infeasible ignore it
                        if ((xNEW[nter] > ub[nter]) || (xNEW[nter] < lb[nter])) {
                            xNEW[nter] = xOLD[nter];
                            continue;
                        }
                        // And we valuate the objective function for the new point
                        fNEW = prob.fitness(xNEW);

                        // We decide wether to accept or discard the point
                        if (fNEW[0] <= fOLD[0]) {
                            // accept
                            xOLD[nter] = xNEW[nter];
                            fOLD = fNEW;
                            acp[nter]++; // Increase the number of accepted values
                            best_f = fNEW;
                            best_x = xNEW;
                        } else {
                            // test it with Boltzmann to decide the acceptance
                            probab = std::exp(-std::abs(fOLD[0] - fNEW[0]) / currentT);
                            // we compare prob with a random probability.
                            if (probab > drng(m_e)) {
                                xOLD[nter] = xNEW[nter];
                                fOLD = fNEW;
                                acp[nter]++; // Increase the number of accepted values
                            } else {
                                xNEW[nter] = xOLD[nter];
                            }
                        } // end if
                    }     // end for(nter = 0; ...
                }         // end for(kter = 0; ...
                // adjust the step (adaptively)
                for (decltype(dim) iter = 0u; iter < dim; ++iter) {
                    ratio = static_cast<double>(acp[iter]) / static_cast<double>(m_bin_size);
                    acp[iter] = 0u; // reset the counter
                    if (ratio > .6) {
                        // too many acceptances, increase the step by a factor 3 maximum
                        step[iter] = step[iter] * (1. + 2. * (ratio - .6) / .4);
                    } else {
                        if (ratio < .4) {
                            // too few acceptance, decrease the step by a factor 3 maximum
                            step[iter] = step[iter] / (1. + 2. * ((.4 - ratio) / .4));
                        };
                    };
                    // And if it becomes too large, reset it to its initial value
                    if (step[iter] > m_start_range) step[iter] = m_start_range;
                }
                // We log to screen
                if (m_verbosity > 0u) {
                    // Prints a log line after each range adjustment
                    // 1 - Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15), "Current:",
                              std::setw(15), "Range:", std::setw(15), "Temperature:", '\n');
                    }
                    auto avg_range = std::accumulate( step.begin(), step.end(), 0.) / step.size();
                    // 2 - Print
                    print(std::setw(7), prob.get_fevals() - fevals0, std::setw(15), best_f[0], std::setw(15), fOLD[0],
                          std::setw(15), avg_range, std::setw(15), currentT);
                    ++count;
                    std::cout << std::endl; // we flush here as we want the user to read in real time ...
                    // Logs
                    m_log.push_back(log_line_type(prob.get_fevals() - fevals0, best_f[0], fOLD[0], avg_range, currentT));
                }
            }
            // Cooling schedule
            currentT *= Tcoeff;
        }
        if (fOLD[0] <= fit0[0]) {
            pop.set_xf(best_idx, xOLD, fOLD);
        }
        return pop;
    };
    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - 1: will only print and log when the population is improved
     * - >1: will print and log one line each \p level generations.
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:        Fevals:          Best:   Improvement:     Mutations:
     * 632           3797        1464.31        51.0203              1
     * 633           3803        1463.23        13.4503              1
     * 635           3815        1562.02        31.0434              3
     * 667           4007         1481.6        24.1889              1
     * 668           4013        1487.34        73.2677              2
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness
     * function currently in the population, Improvement is the improvement made by the las mutation and Mutations
     * is the number of mutated componnets of the decision vector
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
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned int seed)
    {
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
    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "Simulated Annealing (Corana's)";
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
        stream(ss, "\tStarting temperature: ", m_Ts);
        stream(ss, "\n\tFinal temperature: ", m_Tf);
        stream(ss, "\n\tNumber of temperature adjustments: ", m_n_T_adj);
        stream(ss, "\n\tNumber of range adjustments: ", m_n_range_adj);
        stream(ss, "\n\tBin size: ", m_bin_size);
        stream(ss, "\n\tStarting range: ", m_start_range);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        return ss.str();
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt> std::vector </tt> is a sea::log_line_type containing: Gen, Fevals, Best, Improvement, Mutations as described
     * in sea::set_verbosity
     * @return an <tt> std::vector </tt> of sea::log_line_type containing the logged values Gen, Fevals, Best,
     * Improvement, Mutations
     */
    // const log_type &get_log() const
    //{
    //    return m_log;
    //}
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
        ar(m_Ts, m_Tf, m_n_T_adj, m_n_range_adj, m_bin_size, m_start_range, m_e, m_seed, m_verbosity, m_log);
    }

private:
    // Starting temperature
    double m_Ts;
    // Final temperature
    double m_Tf;
    // Number of temperature adjustments during the annealing procedure
    unsigned int m_n_T_adj;
    // Number of range adjustments performed at each temperature
    unsigned int m_n_range_adj;
    // Number of mutation trials to evaluate the acceptance rate
    unsigned int m_bin_size;
    // Starting neighbourhood size
    double m_start_range;

    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::simulated_annealing)

#endif
