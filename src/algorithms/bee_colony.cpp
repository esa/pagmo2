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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/bee_colony.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

bee_colony::bee_colony(unsigned gen, unsigned limit, unsigned seed)
    : m_gen(gen), m_limit(limit), m_e(seed), m_seed(seed), m_verbosity(0u)
{
    if (limit == 0u) {
        pagmo_throw(std::invalid_argument, "The limit must be greater than 0.");
    }
}

/// Algorithm evolve method
/**
 * Evolves the population for a maximum number of generations
 *
 * @param pop population to be evolved
 * @return evolved population
 * @throws std::invalid_argument if the problem is multi-objective or constrained or stochastic
 * @throws std::invalid_argument if the population size is smaller than 2
 */
population bee_colony::evolve(population pop) const
{
    const auto &prob = pop.get_problem();
    auto dim = prob.get_nx();
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto NP = pop.size();
    auto fevals0 = prob.get_fevals(); // fevals already made
    auto count = 1u;                  // regulates the screen output
    // PREAMBLE-------------------------------------------------------------------------------------------------
    // Check whether the problem/population are suitable for bee_colony
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Constraints detected in " + prob.get_name() + " instance. " + get_name()
                                               + " cannot deal with them");
    }
    if (prob.get_nf() != 1u) {
        pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic. " + get_name() + " cannot deal with it");
    }
    if (NP < 2u) {
        pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 2 individuals in the population, "
                                               + std::to_string(NP) + " detected");
    }
    // Get out if there is nothing to do.
    if (m_gen == 0u) {
        return pop;
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    // Some vectors used during evolution are declared.
    vector_double newsol(dim); // contains the mutated candidate
    auto X = pop.get_x();
    auto fit = pop.get_f();
    std::vector<unsigned> trial(NP, 0u);
    std::uniform_real_distribution<double> phirng(-1., 1.); // to generate a number in [-1, 1)
    std::uniform_real_distribution<double> rrng(0., 1.);    // to generate a number in [0, 1)
    std::uniform_int_distribution<vector_double::size_type> comprng(
        0u, dim - 1u); // to generate a random index for the component to mutate
    std::uniform_int_distribution<vector_double::size_type> dvrng(
        0u, NP - 2u); // to generate a random index for the second decision vector

    for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
        // 1 - Employed bees phase
        std::vector<unsigned>::size_type mi = 0u;
        for (decltype(NP) i = 1u; i < NP; ++i) {
            if (trial[i] > trial[mi]) {
                mi = i;
            }
        }
        bool scout = false;
        if (trial[mi] >= m_limit) {
            scout = true;
        }
        for (decltype(NP) i = 0u; i < NP; ++i) {
            if (trial[i] < m_limit || i != mi) {
                newsol = X[i];
                // selects a random component of the decision vector
                auto comp2change = comprng(m_e);
                // selects a random decision vector in the population other than the current
                auto rdv = dvrng(m_e);
                if (rdv >= i) {
                    ++rdv;
                }
                // mutate new solution
                newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[rdv][comp2change]);
                // if the generated parameter value is out of boundaries, shift it into the boundaries
                if (newsol[comp2change] < lb[comp2change]) {
                    newsol[comp2change] = lb[comp2change];
                }
                if (newsol[comp2change] > ub[comp2change]) {
                    newsol[comp2change] = ub[comp2change];
                }
                // if the new solution is better than the old one replace it and reset its trial counter
                auto newfitness = prob.fitness(newsol);
                if (newfitness[0] < fit[i][0]) {
                    fit[i][0] = newfitness[0];
                    X[i][comp2change] = newsol[comp2change];
                    pop.set_xf(i, newsol, newfitness);
                    trial[i] = 0;
                } else {
                    ++trial[i];
                }
            }
        }
        // 2 - Scout bee phase
        if (scout) {
            for (auto j = 0u; j < dim; ++j) {
                X[mi][j] = uniform_real_from_range(lb[j], ub[j], m_e);
            }
            pop.set_x(mi, X[mi]); // this causes a fitness evaluation
            trial[mi] = 0;
        }
        // 3 - Onlooker bee phase
        // compute probabilities
        vector_double p(NP);
        auto sump = 0.;
        for (decltype(NP) i = 0u; i < NP; ++i) {
            if (fit[i][0] >= 0.) {
                p[i] = 1. / (1. + fit[i][0]);
            } else {
                p[i] = 1. - fit[i][0];
            }
            sump += p[i];
        }
        for (decltype(NP) i = 0u; i < NP; ++i) {
            p[i] /= sump;
        }
        vector_double::size_type s = 0u;
        decltype(NP) t = 0u;
        // for each onlooker bee
        while (t < NP) {
            // probabilistic selection of a food source
            auto r = rrng(m_e);
            if (r < p[s]) {
                ++t;
                newsol = X[s];
                // selects a random component of the decision vector
                auto comp2change = comprng(m_e);
                // selects a random decision vector in the population other than the current
                auto rdv = dvrng(m_e);
                if (rdv >= s) {
                    ++rdv;
                }
                // mutate new solution
                newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[rdv][comp2change]);
                // if the generated parameter value is out of boundaries, shift it into the boundaries
                if (newsol[comp2change] < lb[comp2change]) {
                    newsol[comp2change] = lb[comp2change];
                }
                if (newsol[comp2change] > ub[comp2change]) {
                    newsol[comp2change] = ub[comp2change];
                }
                // if the new solution is better than the old one replace it and reset its trial counter
                auto newfitness = prob.fitness(newsol);
                if (newfitness[0] < fit[s][0]) {
                    fit[s][0] = newfitness[0];
                    X[s][comp2change] = newsol[comp2change];
                    pop.set_xf(s, newsol, newfitness);
                    trial[s] = 0;
                } else {
                    ++trial[s];
                }
            }
            s = (s + 1) % NP;
        }
        // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                auto best_idx = pop.best_idx();
                // Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Current Best:\n");
                }
                print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15), pop.champion_f()[0],
                      std::setw(15), pop.get_f()[best_idx][0], '\n');
                ++count;
                // Logs
                m_log.emplace_back(gen, prob.get_fevals() - fevals0, pop.champion_f()[0], pop.get_f()[best_idx][0]);
            }
        }
    }
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void bee_colony::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Extra info
/**
 * @return a string containing extra info on the algorithm
 */
std::string bee_colony::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tMaximum number of generations: ", m_gen);
    stream(ss, "\n\tLimit: ", m_limit);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
    stream(ss, "\n\tSeed: ", m_seed);
    return ss.str();
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
void bee_colony::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_limit, m_e, m_seed, m_verbosity, m_log);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::bee_colony)
