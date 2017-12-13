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

#ifndef PAGMO_ALGORITHMS_SEA_HPP
#define PAGMO_ALGORITHMS_SEA_HPP

#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

/// (N+1)-ES Simple Evolutionary Algorithm
/**
 * \image html sea.png
 *
 * Evolutionary strategies date back to the mid 1960s when P. Bienert,
 * I. Rechenberg, and H.-P. Schwefel at the Technical University of Berlin, Germany,
 * developed the first bionics-inspired schemes for evolving optimal shapes of
 * minimal drag bodies in a wind tunnel using Darwin's evolution principle.
 *
 * This c++ class represents the simplest evolutionary strategy, where a
 * population of \f$ \lambda \f$ individuals at each generation produces one offspring
 * by mutating its best individual uniformly at random within the bounds. Should the
 * offspring be better than the worst individual in the population it will substitute it.
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
 *    The mutation is uniform within box-bounds. Hence, unbounded problems will produce undefined
 *    behaviours.
 *
 * .. seealso::
 *
 *    Oliveto, Pietro S., Jun He, and Xin Yao. "Time complexity of evolutionary algorithms for
 *    combinatorial optimization: A decade of results." International Journal of Automation and Computing
 *    4.3 (2007): 281-293.
 *
 * .. seealso::
 *
 *    http://www.scholarpedia.org/article/Evolution_strategies
 *
 * \endverbatim
 */
class sea
{
public:
    /// Single entry of the log (gen, fevals, best, improvement, mutations)
    typedef std::tuple<unsigned int, unsigned long long, double, double, vector_double::size_type> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs sea
     *
     * @param gen Number of generations to consider. Each generation will compute the objective function once
     * @param seed seed used by the internal random number generator
     */
    sea(unsigned int gen = 1u, unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
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
                                              // allowed
        const auto dim = prob.get_nx();       // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto fevals0 = prob.get_fevals(); // disount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.get_nf() != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        if (!pop.size()) {
            pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Main loop
        // 1 - Compute the best and worst individual (index)
        auto best_idx = pop.best_idx();
        auto worst_idx = pop.worst_idx();
        std::uniform_real_distribution<double> drng(0., 1.); // [0,1]

        for (unsigned int i = 1u; i <= m_gen; ++i) {
            if (prob.is_stochastic()) {
                pop.get_problem().set_seed(std::uniform_int_distribution<unsigned int>()(m_e));
                // re-evaluate the whole population w.r.t. the new seed
                for (decltype(pop.size()) j = 0u; j < pop.size(); ++j) {
                    pop.set_xf(j, pop.get_x()[j], prob.fitness(pop.get_x()[j]));
                }
            }

            vector_double offspring = pop.get_x()[best_idx];
            // 2 - Mutate the components (at least one) of the best
            vector_double::size_type mut = 0u;
            while (!mut) {
                for (vector_double::size_type j = 0u; j < dim; ++j) { // for each decision vector component
                    if (drng(m_e) < 1.0 / static_cast<double>(dim)) {
                        offspring[j] = uniform_real_from_range(lb[j], ub[j], m_e);
                        ++mut;
                    }
                }
            }
            // 3 - Insert the offspring into the population if better
            auto offspring_f = prob.fitness(offspring);
            auto improvement = pop.get_f()[worst_idx][0] - offspring_f[0];
            if (improvement >= 0.) {
                pop.set_xf(worst_idx, offspring, offspring_f);
                if (pop.get_f()[best_idx][0] - offspring_f[0] >= 0.) {
                    best_idx = worst_idx;
                }
                worst_idx = pop.worst_idx();
                // Logs and prints (verbosity mode 1: a line is added everytime the population is improved by the
                // offspring)
                if (m_verbosity == 1u && improvement > 0.) {
                    // Prints on screen
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:",
                              std::setw(15), "Improvement:", std::setw(15), "Mutations:", '\n');
                    }
                    print(std::setw(7), i, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                          pop.get_f()[best_idx][0], std::setw(15), improvement, std::setw(15), mut, '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(i, prob.get_fevals() - fevals0, pop.get_f()[best_idx][0], improvement, mut);
                }
            }
            // 4 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 1u) {
                // Every m_verbosity generations print a log line
                if (i % m_verbosity == 1u) {
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:",
                              std::setw(15), "Improvement:", std::setw(15), "Mutations:", '\n');
                    }
                    print(std::setw(7), i, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                          pop.get_f()[best_idx][0], std::setw(15), improvement, std::setw(15), mut, '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(i, prob.get_fevals() - fevals0, pop.get_f()[best_idx][0], improvement, mut);
                }
            }
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
     * function currently in the population, Improvement is the improvement made by the last mutation and Mutations
     * is the number of mutated components of the decision vector
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
        return "SEA: (N+1)-EA Simple Evolutionary Algorithm";
    }
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) + "\n\tVerbosity: " + std::to_string(m_verbosity)
               + "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a sea::log_line_type containing: Gen, Fevals, Best, Improvement, Mutations as described
     * in sea::set_verbosity
     * @return an <tt>std::vector</tt> of sea::log_line_type containing the logged values Gen, Fevals, Best,
     * Improvement, Mutations
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
        ar(m_gen, m_e, m_seed, m_verbosity, m_log);
    }

private:
    unsigned int m_gen;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::sea)

#endif
