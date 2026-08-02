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
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/genetic_operators.hpp>
#include <pagmo/utils/multi_objective.hpp>  // fast_non_dominated_sorting
#include <pagmo/s11n.hpp>

#include <boost/serialization/optional.hpp>


namespace pagmo{

nsga3::nsga3(unsigned gen, double cr, double eta_c, double mut, double eta_mut,
             size_t divisions, unsigned seed, bool use_memory)
        : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_mut(mut), m_eta_mut(eta_mut),
          m_divisions(divisions), m_seed(seed), m_use_memory(use_memory), m_reng(seed){
    // Validate ctor args
    if(cr < 0.0 || cr > 1.0){
        pagmo_throw(std::invalid_argument, "The crossover probability must be in the range [0, 1], while a value of "
                                           + std::to_string(cr) + " was detected");
    }
    if(mut < 0.0 || mut > 1.0){
        pagmo_throw(std::invalid_argument, "The mutation probability must be in the range [0, 1], while a value of "
                                           + std::to_string(mut) + " was detected");
    }
    if(eta_c < 1.0 || eta_c > 100.0){
        pagmo_throw(std::invalid_argument, "The distribution index for crossover must be in the range [1, 100], "
                                           "while a value of " + std::to_string(eta_c) + " was detected");
    }
    if(eta_mut < 1.0 || eta_mut > 100.0){
        pagmo_throw(std::invalid_argument, "The distribution index for mutation must be in [1, 100], "
                                           "while a value of " + std::to_string(eta_mut) + " was detected");
    }
    // See Deb. Section V, Table I
    if(divisions < 1){
        pagmo_throw(std::invalid_argument, "Invalid <divisions> argument: " + std::to_string(divisions) + ". "
                                           "Number of reference point divisions per objective must be positive");
    }
}


population nsga3::evolve(population pop) const{
    const auto &prob = pop.get_problem();
    const auto bounds = prob.get_bounds();
    const auto fevals0 = prob.get_fevals();
    auto dim_i = prob.get_nix();
    auto NP = pop.size();

    /* Verify problem characteristics:
     *  - Has multiple objectives
     *  - Is not stochastic
     *  - Has unequal bounds
     *  - No non-linear constraints
     *  - "Appropriate" population size and factors; NP >= num reference directions
     */
    if (detail::some_bound_is_equal(prob)) {
        pagmo_throw(std::invalid_argument, "Lower and upper bounds are equal, " + get_name() +
                    " requires these to be different");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    get_name() + " algorithm cannot operate on stochastic problems.");
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
        pagmo_throw(std::invalid_argument,
                    "NSGA-III requires a population greater than 5 and which is divisible by 4."
                    "Detected input population size is: " + std::to_string(NP));
    }
    size_t num_rps = detail::n_choose_k(prob.get_nf() + m_divisions - 1, m_divisions);
    if(NP <= num_rps){
        pagmo_throw(std::invalid_argument,
                    "Population size must exceed number of reference points. NP = "
                    + std::to_string(NP) + " while " + std::to_string(m_divisions) + " divisions for "
                    "reference points gives a total of " + std::to_string(num_rps) + " points.");
    }

    m_log.clear();

    std::vector<vector_double::size_type> shuffle1(NP), shuffle2(NP);
    vector_double::size_type parent1_idx, parent2_idx;
    std::pair<vector_double, vector_double> children;
    size_t count{1u};

    // Initialise population indices
    std::iota(shuffle1.begin(), shuffle1.end(), vector_double::size_type(0));
    std::iota(shuffle2.begin(), shuffle2.end(), vector_double::size_type(0));

    for(decltype(m_gen)gen = 1u; gen <= m_gen; gen++){
        // Copy existing population
        population popnew(pop);

        // Permute population indices
        std::shuffle(shuffle1.begin(), shuffle1.end(), m_reng);
        std::shuffle(shuffle2.begin(), shuffle2.end(), m_reng);

        /*  1. Generate offspring population Q_t
         *  2. R = P_t U Q_t
         *  3. P_t+1 = selection(R)
         */

        if(m_verbosity > 0u){
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

        /*  Mating selection. Deb & Jain leave the parents of NSGA-III chosen at
         *  random, which Seada & Deb later identify as a weakness. We follow the
         *  pagmo convention established by nsga2 instead and hold a binary
         *  tournament on the non-domination rank and the crowding distance.
         */
        auto fnds_res = fast_non_dominated_sorting(pop.get_f());
        auto ndf = std::get<0>(fnds_res);  // non dominated fronts [[0,3,2],[1,5,6],[4],...]
        auto ndr = std::get<3>(fnds_res);  // non domination rank [0,1,0,0,2,1,1, ... ]
        vector_double pop_cd(NP);          // crowding distances of the whole population
        for (const auto &front_idxs : ndf) {
            if (front_idxs.size() < 3u) {  // crowding distance is undefined for one or two points
                for (auto idx : front_idxs) {
                    pop_cd[idx] = std::numeric_limits<double>::infinity();
                }
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

        // Offspring generation
        for (decltype(NP) i = 0; i < NP; i += 4) {
            // We create two offsprings using the shuffled list 1
            parent1_idx = detail::mo_tournament_selection_impl(shuffle1[i], shuffle1[i + 1], ndr, pop_cd, m_reng);
            parent2_idx = detail::mo_tournament_selection_impl(shuffle1[i + 2], shuffle1[i + 3], ndr, pop_cd, m_reng);
            children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                  m_cr, m_eta_c, m_reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_mut, m_eta_mut, m_reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_mut, m_eta_mut, m_reng);
            // Evaluation via prob ensures feval counter is correctly updated
            auto f1 = prob.fitness(children.first);
            auto f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);

            // Repeat with the shuffled list 2
            parent1_idx = detail::mo_tournament_selection_impl(shuffle2[i], shuffle2[i + 1], ndr, pop_cd, m_reng);
            parent2_idx = detail::mo_tournament_selection_impl(shuffle2[i + 2], shuffle2[i + 3], ndr, pop_cd, m_reng);
            children = detail::sbx_crossover_impl(pop.get_x()[parent1_idx], pop.get_x()[parent2_idx], bounds, dim_i,
                                                  m_cr, m_eta_c, m_reng);
            detail::polynomial_mutation_impl(children.first, bounds, dim_i, m_mut, m_eta_mut, m_reng);
            detail::polynomial_mutation_impl(children.second, bounds, dim_i, m_mut, m_eta_mut, m_reng);
            f1 = prob.fitness(children.first);
            f2 = prob.fitness(children.second);
            popnew.push_back(children.first, f1);
            popnew.push_back(children.second, f2);
        } // popnew now contains |P_t|+|R| = 2NP individuals

        /*  Select NP individuals for next generation. A null memory pointer means
         *  the corresponding quantity is recomputed from scratch every generation
         *  instead of being retained across them.
         */
        std::vector<size_t> pop_next = detail::nsga3_selection(popnew.get_f(), NP, m_divisions,
                                                               m_use_memory ? &m_memory.v_ideal : nullptr,
                                                               m_use_memory ? &m_memory.v_extreme : nullptr,
                                                               m_reng);
        for(population::size_type i = 0; i<NP; i++){
            pop.set_xf(i, popnew.get_x()[pop_next[i]], popnew.get_f()[pop_next[i]]);
        }
    }
    return pop;
}

/// Extra info
/**
 * Returns extra information on the algorithm.
 *
 * @return an <tt> std::string </tt> containing extra info on the algorithm
 */
std::string nsga3::get_extra_info() const{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tCrossover probability: ", m_cr);
    stream(ss, "\n\tDistribution index for crossover: ", m_eta_c);
    stream(ss, "\n\tMutation probability: ", m_mut);
    stream(ss, "\n\tDistribution index for mutation: ", m_eta_mut);
    stream(ss, "\n\tReference point divisions: ", m_divisions);
    stream(ss, "\n\tInter-generational memory: ", m_use_memory);
    stream(ss, "\n\tSeed: ", m_seed);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
    return ss.str();
}

// Object serialization
template <typename Archive>
void nsga3::serialize(Archive &ar, unsigned int) {
    detail::archive(ar, m_gen, m_cr, m_eta_c, m_mut, m_eta_mut, m_divisions, m_seed,
                    m_use_memory, m_memory, m_reng, m_verbosity, m_log);
}

}  // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::nsga3)
