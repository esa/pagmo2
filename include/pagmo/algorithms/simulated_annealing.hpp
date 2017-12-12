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

#include <algorithm> //std::accumulate
#include <cmath>     //std::is_finite
#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

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
 * \verbatim embed:rst:leading-asterisk
 *
 * .. warning::
 *
 *    The algorithm is not suitable for multi-objective problems, nor for
 *    constrained or stochastic optimization
 *
 * .. note::
 *
 *    When selecting the starting and final temperature values it helps to think about the tempertaure
 *    as the deterioration in the objective function value that still has a 37% chance of being accepted.
 *
 * .. note::
 *
 *    At each call of the evolve method the number of fitness evaluations will be
 *    `n_T_adj` * `n_range_adj` * `bin_size` times the problem dimension
 *
 * .. seealso::
 *
 *    Corana, A., Marchesi, M., Martini, C., & Ridella, S. (1987). Minimizing multimodal
 *    functions of continuous variables with the “simulated annealing” algorithm Corrigenda
 *    for this article is available here. ACM Transactions on Mathematical Software (TOMS), 13(3), 262-280.
 *
 * .. seealso::
 *
 *    http://people.sc.fsu.edu/~inavon/5420a/corana.pdf
 *
 * \endverbatim
 */
class simulated_annealing : public not_population_based
{
public:
    /// Single entry of the log (fevals, best, current, avg_range, temperature)
    typedef std::tuple<unsigned long long, double, double, double, double> log_line_type;
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
     *
     * @throws std::invalid_argument if \p Ts or \p Tf are not finite and positive, \p start_range is not in (0,1],
     * \p n_T_adj or n_range_adj \p are not strictly positive
     * @throws if \p Tf > \p Ts
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
        if (Tf > Ts) {
            pagmo_throw(std::invalid_argument,
                        "The final temperature must be smaller than the initial temperature, while a value of "
                            + std::to_string(Tf) + " >= " + std::to_string(Ts) + " was detected.");
        }
        if (start_range <= 0. || start_range > 1.) {
            pagmo_throw(std::invalid_argument, "The start range must be in (0,1], while a value of "
                                                   + std::to_string(start_range) + " was detected.");
        }
        if (n_T_adj == 0u) {
            pagmo_throw(std::invalid_argument,
                        "The number of temperature adjustments must be strictly positive, while a value of "
                            + std::to_string(n_T_adj) + " was detected.");
        }
        if (n_range_adj == 0u) {
            pagmo_throw(std::invalid_argument,
                        "The number of range adjustments must be strictly positive, while a value of "
                            + std::to_string(n_range_adj) + " was detected.");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective, constrained or stochastic
     * @throws std::invalid_argument if the population size is < 1u
     */
    population evolve(population pop) const
    {
        // We store some useful properties
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
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
        if (!pop.size()) {
            pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)

        // We init the starting point
        auto sel_xf = select_individual(pop);
        vector_double x0(std::move(sel_xf.first)), fit0(std::move(sel_xf.second));
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
                // 1 - Annealing
                for (decltype(m_bin_size) kter = 0u; kter < m_bin_size; ++kter) {
                    auto nter = std::uniform_int_distribution<vector_double::size_type>(0u, dim - 1u)(m_e);
                    for (decltype(dim) numb = 0u; numb < dim; ++numb) {
                        nter = (nter + 1u) % dim;
                        // We modify the current point by mutating its nter component within the adaptive step
                        auto width = step[nter] * (ub[nter] - lb[nter]);
                        xNEW[nter] = std::uniform_real_distribution<>(std::max(xOLD[nter] - width, lb[nter]),
                                                                      std::min(xOLD[nter] + width, ub[nter]))(m_e);
                        // And we valuate the objective function for the new point
                        fNEW = prob.fitness(xNEW);
                        // We decide wether to accept or discard the point
                        if (fNEW[0] <= fOLD[0]) {
                            // accept
                            xOLD[nter] = xNEW[nter];
                            fOLD = fNEW;
                            acp[nter]++; // Increase the number of accepted values
                            // We update the best
                            if (fNEW[0] <= best_f[0]) {
                                best_f = fNEW;
                                best_x = xNEW;
                            }
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
                        }
                        // 2 - We log to screen
                        if (m_verbosity > 0u) {
                            // Prints a log line every m_verbosity fitness evaluations
                            auto fevals_count = prob.get_fevals() - fevals0;
                            if (fevals_count >= (count - 1u) * m_verbosity) {
                                // 1 - Every 50 lines print the column names
                                if (count % 50u == 1u) {
                                    print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15),
                                          "Current:", std::setw(15), "Mean range:", std::setw(15), "Temperature:",
                                          '\n');
                                }
                                auto avg_range
                                    = std::accumulate(step.begin(), step.end(), 0.) / static_cast<double>(step.size());
                                // 2 - Print
                                print(std::setw(7), fevals_count, std::setw(15), best_f[0], std::setw(15), fOLD[0],
                                      std::setw(15), avg_range, std::setw(15), currentT);
                                ++count;
                                std::cout << std::endl; // we flush here as we want the user to read in real time ...
                                // Logs
                                m_log.emplace_back(fevals_count, best_f[0], fOLD[0], avg_range, currentT);
                            }
                        }
                    } // end for(nter = 0; ...
                }     // end for(kter = 0; ...
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
            }
            // Cooling schedule
            currentT *= Tcoeff;
        }
        // We update the decision vector in pop, but only if things have improved
        if (best_f[0] <= fit0[0]) {
            replace_individual(pop, best_x, best_f);
        }
        return pop;
    };
    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >=1: will print and log one line at minimum every \p level fitness evaluations.
     *
     * Example (verbosity 5000):
     * @code{.unparsed}
     * Fevals:          Best:       Current:    Mean range:   Temperature:
     *  ...
     *  45035      0.0700823       0.135928     0.00116657      0.0199526
     *  50035      0.0215442      0.0261641    0.000770297           0.01
     *  55035     0.00551839      0.0124842    0.000559839     0.00501187
     *  60035     0.00284761     0.00703856    0.000314098     0.00251189
     *  65035     0.00264808      0.0114764    0.000314642     0.00125893
     *  70035      0.0011007     0.00293813    0.000167859    0.000630957
     *  75035    0.000435798     0.00184352    0.000126954    0.000316228
     *  80035    0.000287984    0.000825294    8.91823e-05    0.000158489
     *  85035     9.5885e-05    0.000330647    6.49981e-05    7.94328e-05
     *  90035     4.7986e-05    0.000148512    4.24692e-05    3.98107e-05
     *  95035    2.43633e-05    2.43633e-05    2.90025e-05    1.99526e-05
     * @endcode
     *
     * Fevals is the number of function evaluation used, Best is the best fitness
     * function found, Current is the last fitness sampled, Mean range is the Mean
     * search range across the decision vector components, Temperature is the current temperature.
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
    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "SA: Simulated Annealing (Corana's)";
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
     * <tt>std::vector</tt> is a simulated_annealing::log_line_type containing: Fevals, Best, Current, Mean range
     * Temperature as described in simulated_annealing::set_verbosity
     * @return an <tt>std::vector</tt> of simulated_annealing::log_line_type containing the logged values Gen, Fevals,
     * Best, Improvement, Mutations
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
     * @throws unspecified any exception thrown by the serialization of the UDA and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<not_population_based>(this), m_Ts, m_Tf, m_n_T_adj, m_n_range_adj, m_bin_size,
           m_start_range, m_e, m_seed, m_verbosity, m_log);
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
    // Deleting the methods load save public in base as to avoid conflict with serialize
    template <typename Archive>
    void load(Archive &ar) = delete;
    template <typename Archive>
    void save(Archive &ar) const = delete;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::simulated_annealing)

#endif
