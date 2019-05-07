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

#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

sea::sea(unsigned gen, unsigned seed) : m_gen(gen), m_e(seed), m_seed(seed), m_verbosity(0u), m_log() {}

/// Algorithm evolve method
/**
 * @param pop population to be evolved
 * @return evolved population
 * @throws std::invalid_argument if the problem is multi-objective or constrained
 */
population sea::evolve(population pop) const
{
    // We store some useful properties
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    const auto dim = prob.get_nx();       // This getter does not return a const reference but a copy
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto fevals0 = prob.get_fevals(); // disount for the already made fevals
    unsigned count = 1u;              // regulates the screen output

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

    for (unsigned i = 1u; i <= m_gen; ++i) {
        if (prob.is_stochastic()) {
            pop.get_problem().set_seed(std::uniform_int_distribution<unsigned>()(m_e));
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
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Improvement:", std::setw(15), "Mutations:", '\n');
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
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Improvement:", std::setw(15), "Mutations:", '\n');
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
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void sea::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Extra info
/**
 * One of the optional methods of any user-defined algorithm (UDA).
 *
 * @return a string containing extra info on the algorithm
 */
std::string sea::get_extra_info() const
{
    return "\tGenerations: " + std::to_string(m_gen) + "\n\tVerbosity: " + std::to_string(m_verbosity)
           + "\n\tSeed: " + std::to_string(m_seed);
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
void sea::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_e, m_seed, m_verbosity, m_log);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::sea)
