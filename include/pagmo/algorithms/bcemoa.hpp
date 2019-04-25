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


#ifndef PAGMO_ALGORITHMS_BCEMOA_HPP
#define PAGMO_ALGORITHMS_BCEMOA_HPP

#include <algorithm> // std::shuffle, std::transform
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>
#include <pagmo/algorithm.hpp> // needed for the cereal macro
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>         // uniform_real_from_range, some_bound_is_equal
#include <pagmo/utils/multi_objective.hpp> // crowding_distance, etc..
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/preference.hpp> 
//#include <mynsga2.hpp>
namespace pagmo
{

class bcemoa//: nsga2
{
	algorithm m_algo;
	unsigned int m_gen1;
	unsigned int m_geni;
	double m_cr;
	double m_eta_c;
	double m_m;
	double m_eta_m;
	mutable detail::random_engine_type m_e;
	unsigned int m_seed;
	unsigned int m_verbosity;
	mutable log_type m_log;
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned int, unsigned long long, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs the BCEMOA algorithm.
     *
     * @param[in] gen1 Number of generations to evolve.
	 * @param[in] geni Number of generations to evolve between each interaction.
     * @param[in] cr Crossover probability.
     * @param[in] eta_c Distribution index for crossover.
     * @param[in] m Mutation probability.
     * @param[in] eta_m Distribution index for mutation.
     * @param seed seed used by the internal random number generator (default is random)
	 * @param MaxIteration  maximum number of iterations for optimization algorithm
	 * @param MaxInteraction maximum number of interactions with DM and preference learning
     * @throws std::invalid_argument if \p cr is not \f$ \in [0,1[\f$, \p m is not \f$ \in [0,1]\f$, \p eta_c is not in
     * [1,100[ or \p eta_m is not in [1,100[.
     */
    bcemoa(unsigned gen1 = 1u, unsigned geni = 1u, double cr = 0.95, double eta_c = 10., double m = 0.01, double eta_m = 50.,
          unsigned seed = pagmo::random_device::next())
        : m_gen1(gen1), m_geni(geni), m_cr(cr), m_eta_c(eta_c), m_m(m), m_eta_m(eta_m), m_e(seed), m_seed(seed), m_verbosity(0u),
		m_log()
    {
        algorithm m_algo{ nsga2(gen1, cr, eta_c, m, eta_m, seed) };//M: AS THIS IS BCEMOA CLASS WE CAN USE THE ORIGINAL NSGA2

        // MANUEL: The machine DM must be initialized here (the preferences)
        
        // TODO other parameters of BCEMOA
		}

    // MANUEL: This code is not correctly indented (are you using tabs instead of spaces?)
    		/// Algorithm evolve method (juice implementation of the algorithm)
		/**
		 *
		 * Evolves the population for the requested number of generations.
		 *
		 * @param pop population to be evolved
		 * @return evolved population
		 * @throw std::invalid_argument if pop.get_problem() is stochastic, single objective or has non linear constraints.
		 * If \p int_dim is larger than the problem dimension. If the population size is smaller than 5 or not a multiple of
		 * 4.
		 */
		population evolve(population pop) const
		{
                    // We store some useful variables
                    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
												  // allowed
                    auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
                    auto NP = pop.size();

                    auto fevals0 = prob.get_fevals(); // discount for the fevals already made

                    pop = m_algo.evolve(pop);

                    // TODO: Implement the rest of bcemoa

                    return pop;
                }
                    
                    //here I should add the other evolve guided by preference function

		population evolvePref(population pop) const
		{
                    // MANUEL: This 
			// We store some useful variables
			const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
												  // allowed
			auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
			
			auto NP = pop.size();

			auto fevals0 = prob.get_fevals(); // discount for the fevals already made
			unsigned int count = 1u;          // regulates the screen output

			
			
			// Declarations
			std::vector<vector_double::size_type> best_idx(NP), shuffle1(NP), shuffle2(NP);
			vector_double::size_type parent1_idx, parent2_idx;
			vector_double child1(dim), child2(dim);

			std::iota(shuffle1.begin(), shuffle1.end(), 0u);
			std::iota(shuffle2.begin(), shuffle2.end(), 0u);
			linearPreference pref();
			// Main NSGA-II loop
			for (decltype(m_geni) gen = 1u; gen <= m_geni; gen++) {
				// At each generation we make a copy of the population into popnew
				population popnew(pop);

				// We create some pseudo-random permutation of the poulation indexes
				std::shuffle(shuffle1.begin(), shuffle1.end(), m_e);
				std::shuffle(shuffle2.begin(), shuffle2.end(), m_e);

				// 1 - We compute crowding distance and non dominated rank for the current population
				auto fnds_res = fast_non_dominated_sorting(pop.get_f());
				auto ndf = std::get<0>(fnds_res); // non dominated fronts [[0,3,2],[1,5,6],[4],...]
				vector_double pop_pv(NP);         // defining an array for inputing prefrence values
				auto ndr = std::get<3>(fnds_res); // non domination rank [0,1,0,0,2,1,1, ... ]
				vector_double pop_pv = pref.utility(&pop);
				vector_double rank = linearpreference.rank(&pop_pv);
				//for (const auto &front_idxs : ndf) {
				//			std::vector<vector_double> front;
				//			for (auto idx : front_idxs) {
				//				front.push_back(pop.get_f()[idx]);
				//			}
				//			//auto pv = linear(front);
				//			for (decltype(front_idxs.size()) i = 0u; i < front_idxs.size(); ++i) {
				//				pop_pv[front_idxs[i]] = linearpreference.utility(front_idxs[i]);
				//			}

				//}

				// 3 - We then loop thorugh all individuals with increment 4 to select two pairs of parents that will
				// each create 2 new offspring
				for (decltype(NP) i = 0u; i < NP; i += 4) {
					// We create two offsprings using the shuffled list 1
					parent1_idx = tournament_selection(shuffle1[i], shuffle1[i + 1], ndr, pop_pv);
					parent2_idx = tournament_selection(shuffle1[i + 2], shuffle1[i + 3], ndr, pop_pv);
					crossover(child1, child2, parent1_idx, parent2_idx, pop);
					mutate(child1, pop);
					mutate(child2, pop);
					// we use prob to evaluate the fitness so
					// that its feval counter is correctly updated
					auto f1 = prob.fitness(child1);
					auto f2 = prob.fitness(child2);
					popnew.push_back(child1, f1);
					popnew.push_back(child2, f2);

					// We repeat with the shuffled list 2
					parent1_idx = tournament_selection(shuffle2[i], shuffle2[i + 1], ndr, pop_pv);
					parent2_idx = tournament_selection(shuffle2[i + 2], shuffle2[i + 3], ndr, pop_pv);
					crossover(child1, child2, parent1_idx, parent2_idx, pop);
					mutate(child1, pop);
					mutate(child2, pop);
					// we use prob to evaluate the fitness so
					// that its feval counter is correctly updated
					f1 = prob.fitness(child1);
					f2 = prob.fitness(child2);
					popnew.push_back(child1, f1);
					popnew.push_back(child2, f2);
				} // popnew now contains 2NP individuals

				// This method returns the sorted N best individuals in the population according to the crowded comparison
				// operator
				best_idx = select_best_N_mo(popnew.get_f(), NP);
				// We insert into the population
				for (population::size_type i = 0; i < NP; ++i) {
					pop.set_xf(i, popnew.get_x()[best_idx[i]], popnew.get_f()[best_idx[i]]);
				}
			} // end of main NSGAII loop
			//at the end of each geni iterations a log files is printed
			// We compute the ideal point
			vector_double ideal_point = ideal(pop.get_f());
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

							print('\n');
			// Logs
			m_log.emplace_back(gen, prob.get_fevals() - fevals0, ideal_point);
			return pop;
		}
		/// Sets the seed
    


};

 // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::bcemoa)

#endif
