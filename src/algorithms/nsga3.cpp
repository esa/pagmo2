/* Copyright 2017-2021 PaGMO development team

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

/*
 *  Implements the NSGA-III multi-objective evolutionary algorithm
 *  as described in http://dx.doi.org/10.1109/TEVC.2013.2281535
 *
 *  Paul Slavin <paul.slavin@manchester.ac.uk>
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/genetic_operators.hpp>
#include <pagmo/utils/multi_objective.hpp> // fast_non_dominated_sorting

// NOTE: apparently this must be included *after*
// the other serialization headers.
#include <boost/serialization/optional.hpp>

namespace pagmo
{

nsga3::nsga3(unsigned gen, double cr, double eta_c, double mut, double eta_mut, std::size_t divisions,
             std::size_t divisions_inner, bool random_mating, unsigned seed, bool use_memory)
    : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_mut(mut), m_eta_mut(eta_mut), m_divisions(divisions),
      m_divisions_inner(divisions_inner), m_random_mating(random_mating), m_seed(seed), m_use_memory(use_memory),
      m_reng(seed), m_verbosity(0u)
{
    /*  Validate the ctor args. The tests against NaN are explicit: every comparison
     *  involving a NaN is false, so a range check alone would let one through.
     */
    if (!std::isfinite(cr) || cr < 0.0 || cr > 1.0) {
        pagmo_throw(std::invalid_argument, "The crossover probability must be in the range [0, 1], while a value of "
                                               + std::to_string(cr) + " was detected");
    }
    if (!std::isfinite(mut) || mut < 0.0 || mut > 1.0) {
        pagmo_throw(std::invalid_argument, "The mutation probability must be in the range [0, 1], while a value of "
                                               + std::to_string(mut) + " was detected");
    }
    if (!std::isfinite(eta_c) || eta_c < 1.0 || eta_c > 100.0) {
        pagmo_throw(std::invalid_argument, "The distribution index for crossover must be in the range [1, 100], "
                                           "while a value of "
                                               + std::to_string(eta_c) + " was detected");
    }
    if (!std::isfinite(eta_mut) || eta_mut < 1.0 || eta_mut > 100.0) {
        pagmo_throw(std::invalid_argument, "The distribution index for mutation must be in [1, 100], "
                                           "while a value of "
                                               + std::to_string(eta_mut) + " was detected");
    }
    // See Deb & Jain, Section V, Table I
    if (divisions < 1u) {
        pagmo_throw(std::invalid_argument, "Invalid <divisions> argument: " + std::to_string(divisions)
                                               + ". "
                                                 "Number of reference direction divisions per objective must be "
                                                 "positive");
    }
    /*  The inner layer exists to keep the direction count manageable when the outer
     *  one alone would explode, so a finer inner layer than outer is always a
     *  mistake. Every configuration of Table I has divisions_inner < divisions.
     */
    if (divisions_inner > divisions) {
        pagmo_throw(std::invalid_argument,
                    "The number of divisions of the inner layer of reference directions must not exceed that of the "
                    "outer layer, while an inner value of "
                        + std::to_string(divisions_inner) + " was detected for an outer value of "
                        + std::to_string(divisions));
    }
}

/// Algorithm evolve method
/**
 * Evolves the population for the requested number of generations.
 *
 * @param pop population to be evolved
 * @return evolved population
 *
 * @throws std::invalid_argument if the problem is stochastic, constrained, single
 * objective or has equal lower and upper bounds; if the population size is smaller
 * than 5, is not a multiple of 4, or is smaller than the number of reference
 * directions; or if a configured batch fitness evaluator returns a fitness vector of
 * unexpected size.
 * @throws unspecified any exception thrown by the reference direction construction,
 * in particular if the requested number of directions is too large to be built.
 */
population nsga3::evolve(population pop) const
{
    const auto &prob = pop.get_problem();
    const auto bounds = prob.get_bounds();
    const auto fevals0 = prob.get_fevals();
    const auto dim_i = prob.get_nix();
    const auto dim = prob.get_nx();
    const auto NP = pop.size();
    unsigned count = 1u; // regulates the screen output

    /* Verify problem characteristics:
     *  - Has multiple objectives
     *  - Is not stochastic
     *  - Has unequal bounds
     *  - No non-linear constraints
     *  - "Appropriate" population size and factors; NP >= num reference directions
     */
    if (detail::some_bound_is_equal(prob)) {
        pagmo_throw(std::invalid_argument,
                    "Lower and upper bounds are equal, " + get_name() + " requires these to be different");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument, get_name() + " algorithm cannot operate on stochastic problems.");
    }
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non-linear constraints detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them.");
    }
    if (prob.get_nf() < 2u) {
        pagmo_throw(std::invalid_argument, "This is a multiobjective algorithm, while number of objectives detected in "
                                               + prob.get_name() + " is " + std::to_string(prob.get_nf()));
    }
    if (NP < 5u || (NP % 4 != 0u)) {
        pagmo_throw(std::invalid_argument, "NSGA-III requires a population greater than 5 and which is divisible by 4."
                                           "Detected input population size is: "
                                               + std::to_string(NP));
    }

    /*  The reference directions do not change during the evolution, so they are
     *  built once here rather than once per generation. Every generation works on a
     *  reset copy of this template.
     */
    const auto directions = detail::generate_reference_directions(prob.get_nobj(), m_divisions, m_divisions_inner);
    /*  Deb & Jain size the population as the smallest multiple of four which is not
     *  smaller than the number of reference directions; their Table I uses a
     *  population of exactly 156 for the 156 directions of the eight-objective case,
     *  so equality is permitted here.
     */
    if (NP < directions.size()) {
        pagmo_throw(std::invalid_argument,
                    "Population size must not be smaller than the number of reference "
                    "directions. NP = "
                        + std::to_string(NP) + " while " + std::to_string(m_divisions) + " outer and "
                        + std::to_string(m_divisions_inner) + " inner divisions for " + std::to_string(prob.get_nobj())
                        + " objectives give a total of " + std::to_string(directions.size()) + " directions.");
    }

    // No throws, all valid: we clear the logs
    m_log.clear();

    std::vector<vector_double::size_type> shuffle1(NP), shuffle2(NP);
    vector_double::size_type parent1_idx, parent2_idx;
    std::pair<vector_double, vector_double> children;
    std::vector<vector_double> offspring;
    offspring.reserve(NP);

    for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++) {
        // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                // We compute the ideal point
                vector_double p_ideal = ideal(pop.get_f());
                // Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                    for (decltype(p_ideal.size()) i = 0u; i < p_ideal.size(); ++i) {
                        if (i >= 5u) {
                            print(std::setw(15), "... :");
                            break;
                        }
                        print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                    }
                    print('\n');
                }
                print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0);
                for (decltype(p_ideal.size()) i = 0u; i < p_ideal.size(); ++i) {
                    if (i >= 5u) {
                        break;
                    }
                    print(std::setw(15), p_ideal[i]);
                }
                print('\n');
                ++count;
                // Logs
                m_log.emplace_back(gen, prob.get_fevals() - fevals0, p_ideal);
            }
        }

        // At each generation we make a copy of the population into popnew
        population popnew(pop);

        /*  The permutations are rebuilt from the identity at the start of every
         *  generation rather than shuffled cumulatively, so that a single evolution
         *  of m_gen generations is identical to m_gen successive evolutions of one
         *  generation each.
         */
        std::iota(shuffle1.begin(), shuffle1.end(), vector_double::size_type(0));
        std::shuffle(shuffle1.begin(), shuffle1.end(), m_reng);
        if (!m_random_mating) {
            std::iota(shuffle2.begin(), shuffle2.end(), vector_double::size_type(0));
            std::shuffle(shuffle2.begin(), shuffle2.end(), m_reng);
        }

        /*  1. Generate offspring population Q_t
         *  2. R = P_t U Q_t
         *  3. P_t+1 = selection(R)
         */
        offspring.clear();

        if (m_random_mating) {
            /*  Deb & Jain Section IV-F: no explicit selection operator is applied,
             *  the parents being picked at random. A random permutation of the
             *  population, mated in consecutive pairs, is a uniformly random pairing
             *  in which every individual is a parent exactly once.
             */
            for (population::size_type i = 0u; i < NP; i += 2u) {
                children = detail::sbx_crossover_impl(pop.get_x()[shuffle1[i]], pop.get_x()[shuffle1[i + 1u]], bounds,
                                                      dim_i, m_cr, m_eta_c, m_reng);
                detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_mut, m_eta_mut, m_reng);
                detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_mut, m_eta_mut, m_reng);
                offspring.push_back(std::move(children.first));
                offspring.push_back(std::move(children.second));
            }
        } else {
            /*  The pagmo convention established by nsga2: a binary tournament on the
             *  non-domination rank and the crowding distance. This is a deliberate
             *  deviation from Deb & Jain, and is documented as such.
             */
            auto fnds_res = fast_non_dominated_sorting(pop.get_f());
            const auto &ndf = std::get<0>(fnds_res); // non dominated fronts [[0,3,2],[1,5,6],[4],...]
            const auto &ndr = std::get<3>(fnds_res); // non domination rank [0,1,0,0,2,1,1, ... ]
            vector_double pop_cd(NP);                // crowding distances of the whole population
            for (const auto &front_idxs : ndf) {
                if (front_idxs.size() < 3u) { // crowding distance is undefined for one or two points
                    for (auto idx : front_idxs) {
                        pop_cd[idx] = std::numeric_limits<double>::infinity();
                    }
                } else {
                    std::vector<vector_double> front;
                    front.reserve(front_idxs.size());
                    for (auto idx : front_idxs) {
                        front.push_back(pop.get_f()[idx]);
                    }
                    auto cd = crowding_distance(front);
                    for (decltype(cd.size()) i = 0u; i < cd.size(); ++i) {
                        pop_cd[front_idxs[i]] = cd[i];
                    }
                }
            }

            for (population::size_type i = 0u; i < NP; i += 4u) {
                // We create two offsprings using the shuffled list 1
                parent1_idx = detail::mo_tournament_selection_impl(shuffle1[i], shuffle1[i + 1u], ndr, pop_cd, m_reng);
                parent2_idx
                    = detail::mo_tournament_selection_impl(shuffle1[i + 2u], shuffle1[i + 3u], ndr, pop_cd, m_reng);
                children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                      m_cr, m_eta_c, m_reng);
                detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_mut, m_eta_mut, m_reng);
                detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_mut, m_eta_mut, m_reng);
                offspring.push_back(std::move(children.first));
                offspring.push_back(std::move(children.second));

                // Repeat with the shuffled list 2
                parent1_idx = detail::mo_tournament_selection_impl(shuffle2[i], shuffle2[i + 1u], ndr, pop_cd, m_reng);
                parent2_idx
                    = detail::mo_tournament_selection_impl(shuffle2[i + 2u], shuffle2[i + 3u], ndr, pop_cd, m_reng);
                children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                      m_cr, m_eta_c, m_reng);
                detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_mut, m_eta_mut, m_reng);
                detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_mut, m_eta_mut, m_reng);
                offspring.push_back(std::move(children.first));
                offspring.push_back(std::move(children.second));
            }
        }

        /*  The whole offspring population is generated before any of it is
         *  evaluated, so that the two evaluation paths below consume the random
         *  engine identically and reach the same decisions. Both perform exactly
         *  |Q_t| fitness evaluations.
         */
        if (m_bfe) {
            const auto n_obj = prob.get_nobj();
            vector_double genes(offspring.size() * dim);
            decltype(genes.size()) pos = 0u;
            for (const auto &child : offspring) {
                for (decltype(child.size()) j = 0u; j < child.size(); ++j) {
                    genes[pos] = child[j];
                    ++pos;
                }
            }
            auto fitnesses = (*m_bfe)(prob, genes);
            if (fitnesses.size() != offspring.size() * n_obj) {
                pagmo_throw(std::invalid_argument, "The batch fitness evaluator of " + get_name()
                                                       + " returned a vector of " + std::to_string(fitnesses.size())
                                                       + " values, while " + std::to_string(offspring.size() * n_obj)
                                                       + " were expected for " + std::to_string(offspring.size())
                                                       + " individuals of " + std::to_string(n_obj)
                                                       + " objectives each");
            }
            for (decltype(offspring.size()) i = 0u; i < offspring.size(); ++i) {
                // Slice the flat fitness vector into chunks of length n_obj
                const auto start_pos = fitnesses.begin() + static_cast<vector_double::difference_type>(i * n_obj);
                const auto end_pos = fitnesses.begin() + static_cast<vector_double::difference_type>((i + 1u) * n_obj);
                popnew.push_back(offspring[i], vector_double(start_pos, end_pos));
            }
        } else {
            for (const auto &child : offspring) {
                // Evaluation via prob ensures the feval counter is correctly updated
                popnew.push_back(child, prob.fitness(child));
            }
        } // popnew now contains |P_t| + |Q_t| = 2NP individuals

        /*  Select NP individuals for the next generation. A null memory pointer means
         *  the corresponding quantity is recomputed from scratch every generation
         *  instead of being retained across them.
         */
        const std::vector<pop_size_t> pop_next
            = detail::nsga3_selection(popnew.get_f(), NP, directions, m_use_memory ? &m_memory.v_ideal : nullptr,
                                      m_use_memory ? &m_memory.v_extreme : nullptr, m_reng);
        for (population::size_type i = 0u; i < NP; i++) {
            pop.set_xf(i, popnew.get_x()[pop_next[i]], popnew.get_f()[pop_next[i]]);
        }
    }
    return pop;
}

/// Sets the batch function evaluation scheme
/**
 * @param b batch function evaluation object
 */
void nsga3::set_bfe(const bfe &b)
{
    m_bfe = b;
}

/// Extra info
/**
 * Returns extra information on the algorithm.
 *
 * @return an <tt> std::string </tt> containing extra info on the algorithm
 */
std::string nsga3::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tCrossover probability: ", m_cr);
    stream(ss, "\n\tDistribution index for crossover: ", m_eta_c);
    stream(ss, "\n\tMutation probability: ", m_mut);
    stream(ss, "\n\tDistribution index for mutation: ", m_eta_mut);
    stream(ss, "\n\tReference direction divisions: ", m_divisions);
    stream(ss, "\n\tReference direction inner divisions: ", m_divisions_inner);
    stream(ss, "\n\tRandom mating: ", m_random_mating);
    stream(ss, "\n\tInter-generational memory: ", m_use_memory);
    stream(ss, "\n\tSeed: ", m_seed);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
    return ss.str();
}

// Object serialization
template <typename Archive>
void nsga3::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_cr, m_eta_c, m_mut, m_eta_mut, m_divisions, m_divisions_inner, m_random_mating, m_seed,
                    m_use_memory, m_memory, m_reng, m_verbosity, m_log, m_bfe);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::nsga3)
