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
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/maco.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>
#include <pagmo/utils/hv_algos/hv_bf_approx.hpp>
#include <pagmo/utils/hv_algos/hv_bf_fpras.hpp>
#include <pagmo/utils/hv_algos/hv_hv2d.hpp>
#include <pagmo/utils/hv_algos/hv_hv3d.hpp>
#include <pagmo/utils/hv_algos/hv_hvwfg.hpp>
#include <pagmo/utils/hypervolume.hpp>
#include <pagmo/utils/multi_objective.hpp>

// NOTE: apparently this must be included *after*
// the other serialization headers.
#include <boost/serialization/optional.hpp>

namespace pagmo
{

maco::maco(unsigned gen, unsigned ker, double q, unsigned threshold, unsigned n_gen_mark, unsigned evalstop,
           double focus, bool memory, unsigned seed)
    : m_gen(gen), m_focus(focus), m_ker(ker), m_evalstop(evalstop), m_e(seed), m_seed(seed), m_verbosity(0u), m_log(),
      m_threshold(threshold), m_q(q), m_n_gen_mark(n_gen_mark), m_memory(memory), m_counter(0u), m_sol_archive(),
      m_n_evalstop(0u), m_gen_mark(1u), m_pop()
{
    if (focus < 0.) {
        pagmo_throw(std::invalid_argument,
                    "The focus parameter must be >=0  while a value of " + std::to_string(focus) + " was detected");
    }
    if ((threshold < 1 || threshold > gen) && gen != 0 && memory == false) {
        pagmo_throw(std::invalid_argument,
                    "If memory is inactive, the threshold parameter must be either in [1,m_gen] while a value of "
                        + std::to_string(threshold) + " was detected");
    }
    if (threshold < 1 && gen != 0 && memory == true) {
        pagmo_throw(std::invalid_argument, "If memory is active, the threshold parameter must be >=1 while a value of "
                                               + std::to_string(threshold) + " was detected");
    }
}

// Algorithm evolve method
population maco::evolve(population pop) const
{
    // If the memory is active, we increase the counter:
    if (m_memory == true) {
        ++m_counter;
    }
    // If memory is active, I store the 'true' population inside m_pop:
    if (m_counter == 1 && m_memory == true) {
        m_pop = pop;
    }
    // We store some useful variables:
    const auto &prob = pop.get_problem();
    auto n_x = prob.get_nx();
    auto pop_size = pop.size();
    unsigned count_screen = 1u;       // regulates the screen output
    auto fevals0 = prob.get_fevals(); // discount for the already made fevals
    auto n_f = prob.get_nf();         // n_f=prob.get_nobj()+prob.get_nec()+prob.get_nic()

    // Other useful variables are stored:
    std::vector<vector_double> sol_archive(m_ker, vector_double(n_x + n_f, 1));
    vector_double omega(m_ker);
    vector_double prob_cumulative(m_ker);
    std::uniform_real_distribution<> dist(0, 1);
    std::normal_distribution<> gauss{0., 1.};
    // I create the vector of vectors where I will store all the new ants which will be generated:
    std::vector<vector_double> new_ants(pop_size, vector_double(n_x, 1));
    vector_double ant(n_x);
    vector_double sigma(n_x);
    vector_double fitness(n_f);
    std::vector<vector_double> merged_fit(pop_size + m_ker, vector_double(n_f, 1));
    std::vector<vector_double> merged_dvs(pop_size + m_ker, vector_double(n_x, 1));
    std::vector<vector_double> sol_archive_fit(m_ker, vector_double(n_f, 1));
    // I retrieve the decision and fitness vectors:
    std::vector<vector_double> dvs(pop_size, vector_double(n_x, 1));
    std::vector<vector_double> fit(pop_size, vector_double(n_f, 1));

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this
    // particular algorithm.
    if (!pop_size) {
        pagmo_throw(std::invalid_argument, get_name() + " cannot work on an empty population");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    if (m_gen == 0u) {
        return pop;
    }
    // I verify that the solution archive is smaller or equal than the population size
    if (m_ker > pop_size) {
        pagmo_throw(std::invalid_argument,
                    get_name() + " cannot work with a solution archive bigger than the population size");
    }
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them.");
    }
    if (prob.get_nf() < 2u) {
        pagmo_throw(std::invalid_argument, "This is a multiobjective algortihm, while number of objectives detected in "
                                               + prob.get_name() + " is " + std::to_string(prob.get_nf()));
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    // Main ACO loop over generations:
    for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
        population popold(pop);
        // In case memory is active, we store handle the population in two variables (m_pop and pop):
        if (m_memory == false) {
            dvs = pop.get_x();
            fit = pop.get_f();
        } else {
            dvs = m_pop.get_x();
            fit = m_pop.get_f();
        }
        // In case memory is active, m_sol_archive is used for keeping track of the sol_archive throughout the different
        // iterations:
        if (m_memory == true && m_counter > 1) {
            sol_archive = m_sol_archive;
        }

        // I store the sol_archive fitness values together with the fitness of the current population
        //(except for the very first generation, in which they would be the same)
        if ((m_counter == 1 && m_memory == true) || (gen == 1 && m_memory == false)) {
            auto fnds = fast_non_dominated_sorting(fit);
            auto ndf = std::get<0>(fnds);
            vector_double::size_type i_arch = 0;
            unsigned front = 0u;
            for (const auto &front_idxs : ndf) {
                if (i_arch < m_ker) {
                    // We can now go through the individuals within each front and store them in the archive, according
                    // to
                    // their hypervolume values:
                    std::vector<vector_double> list_of_fit(front_idxs.size(), vector_double(n_f, 1));
                    std::vector<vector_double> list_of_dvs(front_idxs.size(), vector_double(n_x, 1));
                    vector_double::size_type i_ndf = 0;
                    for (auto idx : front_idxs) {
                        // Here I store the fitness vector corresponding to each front:
                        list_of_fit[i_ndf] = fit[idx];
                        list_of_dvs[i_ndf] = dvs[idx];
                        ++i_ndf;
                    }
                    // I set-up the hypervolume computation by passing the list of fitnesses:
                    hypervolume hv = hypervolume(list_of_fit, true);
                    // I compute the reference point by offsetting it of 0.1 to ensure strict domination:
                    auto ref_point = hv.refpoint(0.1);
                    // I can now compute the hypervolume values:
                    auto contrib = hv.contributions(ref_point);
                    // I sort the individuals in the ndf according to their hypervolume values (bigger first, lower
                    // after)
                    std::vector<decltype(contrib.size())> sort_list(contrib.size());
                    // I now have to order the vectors in a list
                    std::iota(std::begin(sort_list), std::end(sort_list), decltype(contrib.size())(0));
                    // I sort them by placing the biggest hypervolume contributor first:
                    std::sort(sort_list.begin(), sort_list.end(),
                              [&contrib](decltype(contrib.size()) idx1, decltype(contrib.size()) idx2) {
                                  return detail::greater_than_f(contrib[idx1], contrib[idx2]);
                              });

                    // I can now place the sorted individuals in the sol_archive:
                    vector_double::size_type i_hv = 0;
                    for (decltype(contrib.size()) i = 0u; i < contrib.size() && i_arch < m_ker; ++i) {
                        for (decltype(n_x) i_nx = 0u; i_nx < n_x; ++i_nx) {
                            sol_archive[i_arch][i_nx] = list_of_dvs[sort_list[i_hv]][i_nx];
                        }
                        for (decltype(n_f) i_nf = 0u; i_nf < n_f; ++i_nf) {
                            sol_archive[i_arch][n_x + i_nf] = list_of_fit[sort_list[i_hv]][i_nf];
                            sol_archive_fit[i_arch][i_nf] = sol_archive[i_arch][n_x + i_nf];
                        }
                        ++i_hv;
                        ++i_arch;
                    }
                    // If, in the first front, there are more pareto points than the ones allowed to store
                    // in the archive, then we make sure that the extremities are included
                    if (i_arch >= m_ker && front == 0) {
                        vector_double id_pt = ideal(list_of_fit);
                        std::vector<vector_double> border_fits(n_f, vector_double(n_f, 1));
                        std::vector<vector_double> border_points(n_f, vector_double(n_x, 1));
                        vector_double::size_type elem = 0;
                        for (decltype(n_f) i_f = 0; i_f < n_f; ++i_f) {
                            bool flag = true;
                            for (decltype(list_of_fit.size()) i_pop = 0; i_pop < list_of_fit.size() && flag == true;
                                 ++i_pop) {
                                if (list_of_fit[i_pop][i_f] == id_pt[i_f]) {
                                    border_points[elem] = list_of_dvs[i_pop];
                                    border_fits[elem] = list_of_fit[i_pop];
                                    flag = false;
                                    ++elem;
                                }
                            }
                        }
                        for (decltype(n_f) i_f = 0; i_f < n_f && i_f < m_ker; ++i_f) {
                            for (decltype(n_x) i_nx = 0u; i_nx < n_x; ++i_nx) {
                                sol_archive[m_ker - 1 - i_f][i_nx] = border_points[i_f][i_nx];
                            }
                            for (decltype(n_f) i_nf = 0u; i_nf < n_f; ++i_nf) {
                                sol_archive[m_ker - 1 - i_f][n_x + i_nf] = border_fits[i_f][i_nf];
                                sol_archive_fit[m_ker - 1 - i_f][i_nf] = sol_archive[i_f][n_x + i_nf];
                            }
                        }
                    }
                }
                ++front;
            }
            if (m_memory == true) {
                m_sol_archive = sol_archive;
            }
        } else {
            for (decltype(pop_size) j = 0u; j < m_ker + pop_size; ++j) {
                if (j < m_ker) {
                    for (decltype(n_f) i = 0u; i < n_f; ++i) {
                        merged_fit[j][i] = sol_archive[j][n_x + i];
                        sol_archive_fit[j][i] = sol_archive[j][n_x + i];
                    }
                    for (decltype(n_x) i = 0u; i < n_x; ++i) {
                        merged_dvs[j][i] = sol_archive[j][i];
                    }
                } else {
                    for (decltype(n_f) i = 0u; i < n_f; ++i) {
                        merged_fit[j][i] = fit[j - m_ker][i];
                    }
                    for (decltype(n_x) i = 0u; i < n_x; ++i) {
                        merged_dvs[j][i] = dvs[j - m_ker][i];
                    }
                }
            }
        }

        // If the following two points are exactly the same, the ideal point has not been changed from previous
        // generation,
        // and we thus increase the evalstop counter:
        vector_double ideal_point_sol_arch = ideal(sol_archive_fit);
        vector_double ideal_point;
        if ((m_counter == 1 && m_memory == true) || (gen == 1 && m_memory == false)) {
            ideal_point = ideal_point_sol_arch;
        } else {
            ideal_point = ideal(merged_fit);
        }
        bool check = false;
        for (decltype(n_f) i = 0u; i < n_f && check == false; ++i) {
            if (ideal_point_sol_arch[i] != ideal_point[i]) {
                check = true;
            }
        }
        if (check == true) {
            ++m_n_evalstop;
        } else {
            m_n_evalstop = 0u;
        }
        // We increase the gen_mark parameter if there are improvements, or if there have not been improvements in the
        // past two generations or more (this parameter reduces the standard deviation pheromone value, thus causing a
        // greedier search in the domain, around the best individuals)
        if (m_n_evalstop == 0u || m_n_evalstop > 2) {
            ++m_gen_mark;
        }
        // We now restart the gen_mark value if it reaches the user defined threshold:
        if (m_gen_mark > m_n_gen_mark) {
            m_gen_mark = 1;
        }
        // We check whether the maximum number of function improvements in the ideal point has been exceeded:
        if (m_evalstop != 0 && m_n_evalstop >= m_evalstop) {
            if (m_verbosity > 0) {
                std::cout << "max number of evalstop exceeded" << std::endl;
            }
            return pop;
        }

        // 1 - compute ndf and hypervolume values, and update and sort solutions in the sol_archive based on that
        if ((m_counter > 1 && m_memory == true) || (gen > 1 && m_memory == false)) {
            // This returns a std::tuple containing: -the non dominated fronts, -the domination list, -the domination
            // count, -the non domination rank
            auto fnds = fast_non_dominated_sorting(merged_fit);
            auto ndf = std::get<0>(fnds);
            // We now loop through the ndf tuple
            vector_double::size_type i_arch = 0;
            unsigned front = 0u;

            for (const auto &front_idxs : ndf) {
                if (i_arch < m_ker) {
                    // We can now go through the individuals within each front and store them in the archive, according
                    // to
                    // their hypervolume values:
                    std::vector<vector_double> list_of_fit(front_idxs.size(), vector_double(n_f, 1));
                    std::vector<vector_double> list_of_dvs(front_idxs.size(), vector_double(n_x, 1));
                    vector_double::size_type i_ndf = 0;
                    for (auto idx : front_idxs) {
                        // Here I store the fitness vector corresponding to each front:
                        list_of_fit[i_ndf] = merged_fit[idx];
                        list_of_dvs[i_ndf] = merged_dvs[idx];
                        ++i_ndf;
                    }
                    // I set-up the hypervolume computation by passing the list of fitnesses:
                    hypervolume hv = hypervolume(list_of_fit, true);
                    // I compute the reference point by offsetting it of 0.01 to ensure strict domination:
                    auto ref_point = hv.refpoint(0.01);
                    // I can now compute the hypervolume values:
                    auto contrib = hv.contributions(ref_point);

                    // I sort the individuals in the ndf according to their hypervolume values (bigger first, lower
                    // after)
                    std::vector<decltype(contrib.size())> sort_list(contrib.size());
                    // I now have to order the vectors in a list
                    std::iota(std::begin(sort_list), std::end(sort_list), decltype(contrib.size())(0));
                    // I sort them by placing the biggest hypervolume contributor first:
                    std::sort(sort_list.begin(), sort_list.end(),
                              [&contrib](decltype(contrib.size()) idx1, decltype(contrib.size()) idx2) {
                                  return detail::greater_than_f(contrib[idx1], contrib[idx2]);
                              });

                    // I can now place the sorted individuals in the sol_archive:
                    vector_double::size_type i_hv = 0;
                    for (decltype(contrib.size()) i = 0u; i < contrib.size() && i_arch < m_ker; ++i) {
                        for (decltype(n_x) i_nx = 0u; i_nx < n_x; ++i_nx) {
                            sol_archive[i_arch][i_nx] = list_of_dvs[sort_list[i_hv]][i_nx];
                        }
                        for (decltype(n_f) i_nf = 0u; i_nf < n_f; ++i_nf) {
                            sol_archive[i_arch][n_x + i_nf] = list_of_fit[sort_list[i_hv]][i_nf];
                        }
                        ++i_hv;
                        ++i_arch;
                    }
                    // If, in the first front, there are more pareto points than the ones allowed to store
                    // in the archive, then we make sure that the extremities are included
                    if (i_arch >= m_ker && front == 0) {
                        vector_double id_pt = ideal(list_of_fit);
                        std::vector<vector_double> border_fits(n_f, vector_double(n_f, 1));
                        std::vector<vector_double> border_points(n_f, vector_double(n_x, 1));
                        vector_double::size_type elem = 0;
                        for (decltype(n_f) i_f = 0; i_f < n_f; ++i_f) {
                            bool flag = true;
                            for (decltype(list_of_fit.size()) i_pop = 0; i_pop < list_of_fit.size() && flag == true;
                                 ++i_pop) {
                                if (list_of_fit[i_pop][i_f] == id_pt[i_f]) {
                                    border_points[elem] = list_of_dvs[i_pop];
                                    border_fits[elem] = list_of_fit[i_pop];
                                    flag = false;
                                    ++elem;
                                }
                            }
                        }
                        for (decltype(n_f) i_f = 0; i_f < n_f && i_f < m_ker; ++i_f) {
                            for (decltype(n_x) i_nx = 0u; i_nx < n_x; ++i_nx) {
                                sol_archive[m_ker - 1 - i_f][i_nx] = border_points[i_f][i_nx];
                            }
                            for (decltype(n_f) i_nf = 0u; i_nf < n_f; ++i_nf) {
                                sol_archive[m_ker - 1 - i_f][n_x + i_nf] = border_fits[i_f][i_nf];
                                sol_archive_fit[m_ker - 1 - i_f][i_nf] = sol_archive[i_f][n_x + i_nf];
                            }
                        }
                    }
                }
                ++front;
            }
            if (m_memory == true) {
                m_sol_archive = sol_archive;
            }
        }

        // 2 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if ((gen >= 1 && m_verbosity > 0u && m_memory == false)
            || (m_verbosity > 0u && m_memory == true && m_counter >= 1)) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                if (m_memory == false) {
                    // Every 50 lines print the column names
                    if (count_screen % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                        for (decltype(ideal_point_sol_arch.size()) i = 0u; i < ideal_point_sol_arch.size(); ++i) {
                            if (i >= 5u) {
                                print(std::setw(15), "... :");
                                break;
                            }
                            print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                        }
                        print('\n');
                    }
                } else if ((m_memory == true && m_counter == 1) || (m_memory == true && m_counter % 50u == 1u)) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                    for (decltype(ideal_point_sol_arch.size()) i = 0u; i < ideal_point_sol_arch.size(); ++i) {
                        if (i >= 5u) {
                            print(std::setw(15), "... :");
                            break;
                        }
                        print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                    }
                    print('\n');
                }
                if (m_memory == false) {
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0);

                } else {
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals());
                }
                for (decltype(ideal_point_sol_arch.size()) i = 0u; i < ideal_point_sol_arch.size(); ++i) {
                    if (i >= 5u) {
                        break;
                    }
                    print(std::setw(15), ideal_point_sol_arch[i]);
                }
                print('\n');
                ++count_screen;
                // Logs
                if (m_memory == false) {
                    m_log.emplace_back(gen, prob.get_fevals() - fevals0, ideal_point_sol_arch);

                } else {
                    m_log.emplace_back(gen, prob.get_fevals(), ideal_point_sol_arch);
                }
            }
        }

        // 3 - compute pheromone values
        pheromone_computation(gen, prob_cumulative, omega, sigma, pop, sol_archive);

        // 4 - use pheromone values to generate new ants (i.e., individuals)
        generate_new_ants(popold, dist, gauss, prob_cumulative, sigma, new_ants, sol_archive);

        // In case bfe is available, we parallelize the fitness computation
        if (m_bfe) {
            // bfe is available:
            vector_double ants(pop_size * new_ants[0].size());
            decltype(ants.size()) pos = 0u;

            for (population::size_type i = 0; i < pop_size; ++i) {
                for (decltype(new_ants[i].size()) ii = 0u; ii < new_ants[i].size(); ++ii) {
                    ants[pos] = new_ants[i][ii];
                    ++pos;
                }
            }
            // I compute the fitness for each new individual which was generated in the generated_new_ants(..)
            // function
            auto fitnesses = (*m_bfe)(prob, ants);
            decltype(ant.size()) pos_dim = 0u;
            decltype(fitness.size()) pos_fit = 0u;

            for (population::size_type i = 0; i < pop_size; ++i) {

                for (decltype(n_x) ii_dim = 0u; ii_dim < n_x; ++ii_dim) {
                    ant[ii_dim] = ants[pos_dim];
                    ++pos_dim;
                }

                for (decltype(n_f) ii_f = 0u; ii_f < n_f; ++ii_f) {
                    fitness[ii_f] = fitnesses[pos_fit];
                    ++pos_fit;
                }
                // I set the individuals for the next generation
                pop.set_xf(i, ant, fitness);
            }

        } else {
            // bfe not available:
            for (population::size_type i = 0; i < pop_size; ++i) {
                // I compute the fitness for each new individual which was generated in the generated_new_ants(..)
                // function
                for (decltype(n_x) ii = 0u; ii < n_x; ++ii) {
                    ant[ii] = new_ants[i][ii];
                }

                auto ftns = prob.fitness(ant);
                // I set the individuals for the next generation
                pop.set_xf(i, ant, ftns);
                // I set the individuals for the next generation
            }
        }
        if (m_memory == true) {
            m_pop = pop;
        }

    } // end of main ACO loop

    for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
        for (decltype(n_x) ii = 0u; ii < n_x; ++ii) {
            ant[ii] = sol_archive[i][ii];
        }
        vector_double ftns(n_f);
        for (decltype(n_f) ii = 0u; ii < n_f; ++ii) {
            ftns[ii] = sol_archive[i][ii + n_x];
        }
        pop.set_xf(i, ant, ftns);
    }
    return pop;
}

// Sets the seed
void maco::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

// Sets the batch function evaluation scheme
void maco::set_bfe(const bfe &b)
{
    m_bfe = b;
}

// Extra info
std::string maco::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tFocus parameter: ", m_focus);
    stream(ss, "\n\tKernel size: ", m_ker);
    stream(ss, "\n\tEvaluation stopping criterion: ", m_evalstop);
    stream(ss, "\n\tConvergence speed parameter: ", m_q);
    stream(ss, "\n\tThreshold parameter: ", m_threshold);
    stream(ss, "\n\tStandard deviations convergence speed parameter: ", m_n_gen_mark);
    stream(ss, "\n\tMemory parameter: ", m_memory);
    stream(ss, "\n\tPseudo-random number generator (Marsenne Twister 19937): ", m_e);
    stream(ss, "\n\tSeed: ", m_seed);
    stream(ss, "\n\tVerbosity: ", m_verbosity);

    return ss.str();
}

// Object serialization
template <typename Archive>
void maco::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_focus, m_ker, m_evalstop, m_e, m_seed, m_verbosity, m_log, m_threshold, m_q,
                    m_n_gen_mark, m_memory, m_counter, m_sol_archive, m_n_evalstop, m_gen_mark, m_bfe);
}

// Function which computes the pheromone values (useful for generating offspring)
void maco::pheromone_computation(const unsigned gen, vector_double &prob_cumulative, vector_double &omega_vec,
                                 vector_double &sigma_vec, const population &popul,
                                 std::vector<vector_double> &sol_archive) const
{

    const auto &prob = popul.get_problem();
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto n_con = prob.get_ncx();
    auto n_tot = prob.get_nx();

    // Here we define the weights:
    if (m_memory == false) {
        if (gen == 1 || gen == m_threshold) {
            if (gen == m_threshold) {
                m_q = 0.01;
            }

            double omega_new;
            double sum_omega = 0;

            for (decltype(m_ker) l = 1; l <= m_ker; ++l) {
                omega_new = 1.0 / (m_q * m_ker * std::sqrt(2 * boost::math::constants::pi<double>()))
                            * std::exp(-std::pow(l - 1.0, 2) / (2.0 * std::pow(m_q, 2) * std::pow(m_ker, 2)));
                omega_vec[l - 1] = omega_new;
                sum_omega += omega_new;
            }

            for (decltype(m_ker) k = 0u; k < m_ker; ++k) {
                double cumulative = 0.0;
                for (decltype(m_ker) j = 0u; j <= k; ++j) {
                    cumulative += omega_vec[j] / sum_omega;
                }
                prob_cumulative[k] = cumulative;
            }
        }
    } else {
        if (m_counter == m_threshold) {
            m_q = 0.01;
        }

        double omega_new;
        double sum_omega = 0;

        for (decltype(m_ker) l = 1; l <= m_ker; ++l) {
            omega_new = 1.0 / (m_q * m_ker * std::sqrt(2 * boost::math::constants::pi<double>()))
                        * std::exp(-std::pow(l - 1.0, 2) / (2.0 * std::pow(m_q, 2) * std::pow(m_ker, 2)));
            omega_vec[l - 1] = omega_new;
            sum_omega += omega_new;
        }

        for (decltype(m_ker) k = 0u; k < m_ker; ++k) {
            double cumulative = 0;
            for (decltype(m_ker) j = 0u; j <= k; ++j) {
                cumulative += omega_vec[j] / sum_omega;
            }
            prob_cumulative[k] = cumulative;
        }
    }

    // We now compute the standard deviations (sigma):
    for (decltype(n_tot) h = 0; h < n_tot; ++h) {

        // I declare and define D_min and D_max:
        // at first I define D_min, D_max using the first two individuals stored in the sol_archive
        double d_min = std::abs(sol_archive[0][h] - sol_archive[1][h]);

        double d_max = std::abs(sol_archive[0][h] - sol_archive[1][h]);

        // I loop over the various individuals of the variable:
        for (decltype(m_ker) count = 0; count < m_ker - 1; ++count) {

            // I confront each individual with the following ones (until all the comparisons are executed):
            for (decltype(m_ker) k = count + 1; k < m_ker; ++k) {
                // I update d_min
                if (std::abs(sol_archive[count][h] - sol_archive[k][h]) < d_min) {
                    d_min = std::abs(sol_archive[count][h] - sol_archive[k][h]);
                }
                // I update d_max
                if (std::abs(sol_archive[count][h] - sol_archive[k][h]) > d_max) {
                    d_max = std::abs(sol_archive[count][h] - sol_archive[k][h]);
                }
            }
        }

        // In case a value for the focus parameter (different than zero) is passed, this limits the maximum
        // value of the standard deviation
        if (m_focus != 0. && ((d_max - d_min) / gen > (ub[h] - lb[h]) / m_focus) && m_memory == false) {
            sigma_vec[h] = (ub[h] - lb[h]) / m_focus;

        } else if (m_focus != 0. && ((d_max - d_min) / m_counter > (ub[h] - lb[h]) / m_focus) && m_memory == true) {
            sigma_vec[h] = (ub[h] - lb[h]) / m_focus;

        } else {
            if (h < n_con) {
                // continuous variables case:
                sigma_vec[h] = (d_max - d_min) / m_gen_mark;

            } else {
                // integer variables case:
                sigma_vec[h] = std::max(std::max((d_max - d_min) / m_gen_mark, 1.0 / m_gen_mark),
                                        (1.0 - 1.0 / (std::sqrt(n_tot - n_con))));
            }
        }
    }
}

// Function which generates new individuals (i.e., ants)
void maco::generate_new_ants(const population &pop, std::uniform_real_distribution<> dist,
                             std::normal_distribution<double> gauss_pdf, vector_double prob_cumulative,
                             vector_double sigma, std::vector<vector_double> &dvs_new,
                             std::vector<vector_double> &sol_archive) const
{

    const auto &prob = pop.get_problem();
    auto pop_size = pop.size();
    auto n_tot = prob.get_nx();
    auto n_con = prob.get_ncx();
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    vector_double fitness_old;
    vector_double fitness_new;

    // I hereby generate the new ants based on a multi-kernel gaussian probability density function. In particular,
    // I select one of the pdfs that make up the multi-kernel, by using the probability stored in the
    // prob_cumulative vector. A multi-kernel pdf is a weighted sum of several gaussian pdf.

    for (decltype(pop_size) j = 0u; j < pop_size; ++j) {
        vector_double dvs_new_j(n_tot);

        double number = dist(m_e);
        double g_h = 0.0;
        decltype(sol_archive.size()) k_omega = 0u;

        if (number <= prob_cumulative[0]) {
            k_omega = 0;

        } else if (number > prob_cumulative[m_ker - 2]) {
            k_omega = m_ker - 1;

        } else {
            for (decltype(m_ker) k = 1u; k < m_ker - 1; ++k) {
                if (number > prob_cumulative[k - 1] && number <= prob_cumulative[k]) {
                    k_omega = k;
                }
            }
        }
        for (decltype(n_tot) h = 0u; h < n_tot; ++h) {
            g_h = sol_archive[k_omega][h] + sigma[h] * gauss_pdf(m_e);

            if (g_h < lb[h] || g_h > ub[h]) {

                while ((g_h < lb[h] || g_h > ub[h])) {
                    g_h = sol_archive[k_omega][h] + sigma[h] * gauss_pdf(m_e);
                }
            }
            if (h >= n_con) {
                dvs_new_j[h] = std::round(g_h);
            } else {
                dvs_new_j[h] = g_h;
            }
        }

        dvs_new[j] = dvs_new_j;
    }
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::maco)
