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

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/genetic_operators.hpp>
#include <pagmo/utils/multi_objective.hpp>

// NOTE: apparently this must be included *after*
// the other serialization headers.
#include <boost/serialization/optional.hpp>

namespace pagmo
{

nsga2::nsga2(unsigned gen, double cr, double eta_c, double m, double eta_m, unsigned seed)
    : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_m(m), m_eta_m(eta_m), m_e(seed), m_seed(seed), m_verbosity(0u)
{
    if (cr >= 1. || cr < 0.) {
        pagmo_throw(std::invalid_argument, "The crossover probability must be in the [0,1[ range, while a value of "
                                               + std::to_string(cr) + " was detected");
    }
    if (m < 0. || m > 1.) {
        pagmo_throw(std::invalid_argument, "The mutation probability must be in the [0,1] range, while a value of "
                                               + std::to_string(cr) + " was detected");
    }
    if (eta_c < 1. || eta_c > 100.) {
        pagmo_throw(std::invalid_argument, "The distribution index for crossover must be in [1, 100], while a value of "
                                               + std::to_string(eta_c) + " was detected");
    }
    if (eta_m < 1. || eta_m > 100.) {
        pagmo_throw(std::invalid_argument, "The distribution index for mutation must be in [1, 100], while a value of "
                                               + std::to_string(eta_m) + " was detected");
    }
}

/// Algorithm evolve method
/**
 * Evolves the population for the requested number of generations.
 *
 * @param pop population to be evolved
 * @return evolved population
 * @throw std::invalid_argument if pop.get_problem() is stochastic, single objective or has non linear constraints.
 * If \p int_dim is larger than the problem dimension. If the population size is smaller than 5 or not a multiple of
 * 4.
 */
population nsga2::evolve(population pop) const
{
    // We store some useful variables
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    const auto bounds = pop.get_problem().get_bounds();
    auto dim_i = pop.get_problem().get_nix(); // integer dimension
    auto NP = pop.size();

    auto fevals0 = prob.get_fevals(); // discount for the fevals already made
    unsigned count = 1u;              // regulates the screen output

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this
    // particular algorithm.
    if (detail::some_bound_is_equal(prob)) {
        pagmo_throw(std::invalid_argument,
                    get_name()
                        + " cannot work on problems having a lower bound equal to an upper bound. Check your bounds.");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them.");
    }
    if (prob.get_nf() < 2u) {
        pagmo_throw(std::invalid_argument, "This is a multiobjective algortihm, while number of objectives detected in "
                                               + prob.get_name() + " is " + std::to_string(prob.get_nf()));
    }
    if (NP < 5u || (NP % 4 != 0u)) {
        pagmo_throw(std::invalid_argument,
                    "for NSGA-II at least 5 individuals in the population are needed and the "
                    "population size must be a multiple of 4. Detected input population size is: "
                        + std::to_string(NP));
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    // Declarations
    std::vector<vector_double::size_type> best_idx(NP), shuffle1(NP), shuffle2(NP);
    vector_double::size_type parent1_idx, parent2_idx;
    std::pair<vector_double, vector_double> children;

    std::iota(shuffle1.begin(), shuffle1.end(), vector_double::size_type(0));
    std::iota(shuffle2.begin(), shuffle2.end(), vector_double::size_type(0));

    // Main NSGA-II loop
    for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++) {
        // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                // We compute the ideal point
                vector_double ideal_point = ideal(pop.get_f());
                // Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                    for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                        if (i >= 5u) {
                            print(std::setw(15), "... :");
                            break;
                        }
                        print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                    }
                    print('\n');
                }
                print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0);
                for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                    if (i >= 5u) {
                        break;
                    }
                    print(std::setw(15), ideal_point[i]);
                }
                print('\n');
                ++count;
                // Logs
                m_log.emplace_back(gen, prob.get_fevals() - fevals0, ideal_point);
            }
        }

        // At each generation we make a copy of the population into popnew
        population popnew(pop);

        // We create some pseudo-random permutation of the poulation indexes
        std::shuffle(shuffle1.begin(), shuffle1.end(), m_e);
        std::shuffle(shuffle2.begin(), shuffle2.end(), m_e);

        // 1 - We compute crowding distance and non dominated rank for the current population
        auto fnds_res = fast_non_dominated_sorting(pop.get_f());
        auto ndf = std::get<0>(fnds_res); // non dominated fronts [[0,3,2],[1,5,6],[4],...]
        vector_double pop_cd(NP);         // crowding distances of the whole population
        auto ndr = std::get<3>(fnds_res); // non domination rank [0,1,0,0,2,1,1, ... ]
        for (const auto &front_idxs : ndf) {
            if (front_idxs.size() == 1u) { // handles the case where the front has collapsed to one point
                pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
            } else {
                if (front_idxs.size() == 2u) { // handles the case where the front has collapsed to one point
                    pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
                    pop_cd[front_idxs[1]] = std::numeric_limits<double>::infinity();
                } else {
                    std::vector<vector_double> front;
                    for (auto idx : front_idxs) {
                        front.push_back(pop.get_f()[idx]);
                    }
                    auto cd = crowding_distance(front);
                    for (decltype(cd.size()) i = 0u; i < cd.size(); ++i) {
                        pop_cd[front_idxs[i]] = cd[i];
                    }
                }
            }
        }

        // 3 - We then loop thorugh all individuals with increment 4 to select two pairs of parents that will
        // each create 2 new offspring
        if (m_bfe) {
            // bfe is available:
            auto n_obj = prob.get_nobj();
            std::vector<vector_double> poptemp;
            std::vector<unsigned long> fidtemp;
            for (decltype(NP) i = 0u; i < NP; i += 4) {
                // We create two offsprings using the shuffled list 1
                parent1_idx = detail::mo_tournament_selection_impl(shuffle1[i], shuffle1[i + 1], ndr, pop_cd, m_e);
                parent2_idx = detail::mo_tournament_selection_impl(shuffle1[i + 2], shuffle1[i + 3], ndr, pop_cd, m_e);
                children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                      m_cr, m_eta_c, m_e);
                detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_m, m_eta_m, m_e);
                detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_m, m_eta_m, m_e);

                poptemp.push_back(children.first);
                poptemp.push_back(children.second);

                // We repeat with the shuffled list 2
                parent1_idx = detail::mo_tournament_selection_impl(shuffle2[i], shuffle2[i + 1], ndr, pop_cd, m_e);
                parent2_idx = detail::mo_tournament_selection_impl(shuffle2[i + 2], shuffle2[i + 3], ndr, pop_cd, m_e);
                children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                      m_cr, m_eta_c, m_e);
                detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_m, m_eta_m, m_e);
                detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_m, m_eta_m, m_e);
                // we use prob to evaluate the fitness so
                // that its feval counter is correctly updated
                poptemp.push_back(children.first);
                poptemp.push_back(children.second);

            } // poptemp now contains 2NP individuals

            vector_double genes(NP * poptemp[0].size());
            decltype(genes.size()) pos = 0u;
            for (population::size_type i = 0; i < NP; ++i) {
                // I compute the fitness for each new individual which was generated in the
                // tournament_selection for loop
                for (decltype(poptemp[i].size()) ii = 0u; ii < poptemp[i].size(); ++ii) {
                    genes[pos] = poptemp[i][ii];
                    ++pos;
                }
            }
            // array - now contains 2NP new individuals
            // run bfe and populate popnew.
            auto fitnesses = (*m_bfe)(prob, genes);

            // at this point:
            // genes     is an ordered list of child inputs (not used again)
            // poptemp   is a structured list of children   (no fitneeses)
            // fitnesses is an ordered list of fitneeses
            for (decltype(poptemp.size()) i = 0; i < poptemp.size(); i++) {
                // slice up the fitnesses into a chunks of length n_obj
                auto start_pos = fitnesses.begin() + static_cast<std::vector<double>::difference_type>(i * n_obj);
                auto end_pos = fitnesses.begin() + static_cast<std::vector<double>::difference_type>((i + 1) * n_obj);
                std::vector<double> f1(start_pos, end_pos);
                popnew.push_back(poptemp[i], f1);
            }
        } else {
            // bfe not available:
            for (decltype(NP) i = 0u; i < NP; i += 4) {
                // We create two offsprings using the shuffled list 1
                parent1_idx = detail::mo_tournament_selection_impl(shuffle1[i], shuffle1[i + 1], ndr, pop_cd, m_e);
                parent2_idx = detail::mo_tournament_selection_impl(shuffle1[i + 2], shuffle1[i + 3], ndr, pop_cd, m_e);
                children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                      m_cr, m_eta_c, m_e);
                detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_m, m_eta_m, m_e);
                detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_m, m_eta_m, m_e);
                // we use prob to evaluate the fitness so
                // that its feval counter is correctly updated
                auto f1 = prob.fitness(children.first);
                auto f2 = prob.fitness(children.second);
                popnew.push_back(children.first, f1);
                popnew.push_back(children.second, f2);

                // We repeat with the shuffled list 2
                parent1_idx = detail::mo_tournament_selection_impl(shuffle2[i], shuffle2[i + 1], ndr, pop_cd, m_e);
                parent2_idx = detail::mo_tournament_selection_impl(shuffle2[i + 2], shuffle2[i + 3], ndr, pop_cd, m_e);
                children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                      m_cr, m_eta_c, m_e);
                detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_m, m_eta_m, m_e);
                detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_m, m_eta_m, m_e);
                // we use prob to evaluate the fitness so
                // that its feval counter is correctly updated
                f1 = prob.fitness(children.first);
                f2 = prob.fitness(children.second);
                popnew.push_back(children.first, f1);
                popnew.push_back(children.second, f2);
            } // popnew now contains 2NP individuals
        }
        // This method returns the sorted N best individuals in the population according to the crowded comparison
        // operator
        best_idx = select_best_N_mo(popnew.get_f(), NP);
        // We insert into the population
        for (population::size_type i = 0; i < NP; ++i) {
            pop.set_xf(i, popnew.get_x()[best_idx[i]], popnew.get_f()[best_idx[i]]);
        }
    } // end of main NSGAII loop
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void nsga2::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Sets the batch function evaluation scheme
/**
 * @param b batch function evaluation object
 */
void nsga2::set_bfe(const bfe &b)
{
    m_bfe = b;
}

/// Extra info
/**
 * Returns extra information on the algorithm.
 *
 * @return an <tt> std::string </tt> containing extra info on the algorithm
 */
std::string nsga2::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tCrossover probability: ", m_cr);
    stream(ss, "\n\tDistribution index for crossover: ", m_eta_c);
    stream(ss, "\n\tMutation probability: ", m_m);
    stream(ss, "\n\tDistribution index for mutation: ", m_eta_m);
    stream(ss, "\n\tSeed: ", m_seed);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
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
void nsga2::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_cr, m_eta_c, m_m, m_eta_m, m_e, m_seed, m_verbosity, m_log, m_bfe);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::nsga2)
