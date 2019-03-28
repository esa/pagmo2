<<<<<<< HEAD
/* Copyright 2017-2018 PaGMO development team
=======
/* Copyright 2017 PaGMO development team
>>>>>>> origin/master
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

#ifndef PAGMO_ALGORITHMS_GACO_HPP
#define PAGMO_ALGORITHMS_GACO_HPP

#include <algorithm> // std::shuffle, std::transform
<<<<<<< HEAD
#include <boost/math/constants/constants.hpp>
=======
>>>>>>> origin/master
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <sstream> //std::osstringstream
#include <string>
#include <tuple>
#include <utility> //std::swap
#include <vector>

#include <pagmo/algorithm.hpp> // needed for the cereal macro
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/rng.hpp>
<<<<<<< HEAD
#include <pagmo/utils/constrained.hpp>
=======
>>>>>>> origin/master
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/generic.hpp>         // uniform_real_from_range
#include <pagmo/utils/multi_objective.hpp> // crowding_distance, etc..

namespace pagmo
{
/// Extended ACO
/**
<<<<<<< HEAD
=======
 * \image html ACO.jpg "The ACO flowchart" width=3cm [TO BE ADDED]
>>>>>>> origin/master
 * ACO is inspired by the natural mechanism with which real ant colonies forage food.
 * This algorithm has shown promising results in many trajectory optimization problems.
 * The first appearance of the algorithm happened in Dr. Marco Dorigo's thesis, in 1992.
 * ACO generates future generations of ants by using the a multi-kernel gaussian distribution
 * based on three parameters (i.e., pheromone values) which are computed depending on the quality
 * of each previous solution. The solutions are ranked through an oracle penalty method.
 *
 *
<<<<<<< HEAD
 * The version implemented in pagmo can be applied to box-bounded multiple-objective optimization and its implementation
 * is an extension of Schlueter's originally proposed ACO algorithm.
=======
 * The version implemented in pagmo can be applied to box-bounded multiple-objective optimization.
>>>>>>> origin/master
 *
 * See:  M. Schlueter, et al. (2009). Extended ant colony optimization for non-convex
 * mixed integer non-linear programming. Computers & Operations Research.
 */
<<<<<<< HEAD
class gaco
{
public:
    /// Single entry of the log (gen, m_fevals, best_fit, m_ker, m_oracle, dx, dp)
    typedef std::tuple<unsigned, unsigned, double, unsigned, double, double, double> log_line_type;
=======
class g_aco
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, double, double, double, double, double, double> log_line_type;
>>>>>>> origin/master
    /// The log
    typedef std::vector<log_line_type> log_type;

    /**
     * Constructs the ACO user defined algorithm for single and multi-objective optimization.
     *
     * @param[in] gen Generations: number of generations to evolve.
<<<<<<< HEAD
     * @param[in] ker Kernel: number of solutions stored in the solution archive.
     * @param[in] q Convergence speed parameter: this parameter is useful for managing the convergence speed towards the
     * found minima (the smaller the faster).
     * @param[in] oracle Oracle parameter: this is the oracle parameter used in the penalty method.
     * @param[in] acc Accuracy parameter: for maintaining a minimum penalty function's values distances.
     * @param[in] threshold Threshold parameter: when the generations reach the threshold then q is set to
     * 0.01 automatically.
     * @param[in] n_gen_mark Standard deviations convergence speed parameter: this parameters determines the convergence
     * speed of the standard deviations values.
     * @param[in] impstop Improvement stopping criterion: if a positive integer is assigned here, the algorithm will
     * count the runs without improvements, if this number will exceed impstop value, the algorithm will be stopped.
     * @param[in] evalstop Evaluation stopping criterion: same as previous one, but with function evaluations.
     * @param[in] focus Focus parameter: this parameter makes the search for the optimum greedier and more focused on
     * local improvements (the higher the greedier). If the value is very high, the search is more focused around the
     * current best solutions.
     * @param[in] paretomax Max number of non-dominated solutions: this regulates the max number of Pareto points to be
     * stored.
     * @param[in] epsilon Pareto precision parameter: the smaller this parameter, the higher the chances to introduce a
     * new solution in the Pareto front.
     * @param[in] memory Memory parameter: if true, memory is activated in the algorithm for multiple calls
     * @param seed seed used by the internal random number generator (default is random).
     * @throws std::invalid_argument if \p acc is not \f$ >=0 \f$, \p impstop is not a
     * positive integer, \p evalstop is not a positive integer, \p focus is not \f$ >=0 \f$,
     * \p ker is not a positive integer, \p oracle is not positive, \p paretomax is not a positive integer, \p
     * threshold is not \f$ \in [1,gen] \f$ when \f$memory=false\f$ and  \f$gen!=0\f$, \p threshold is not \f$ >=1 \f$
     * when \f$memory=true\f$ and \f$gen!=0\f$, \p epsilon is not \f$ \in [0,1[\f$, \p q is not \f$ >=0 \f$
     */

    gaco(unsigned gen = 100u, unsigned ker = 63u, double q = 1.0, double oracle = 0., double acc = 0.01,
         unsigned threshold = 1u, unsigned n_gen_mark = 7u, unsigned impstop = 100000u, unsigned evalstop = 100000u,
         double focus = 0., unsigned paretomax = 10u, double epsilon = 0.9, bool memory = false,
         unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_acc(acc), m_impstop(impstop), m_evalstop(evalstop), m_focus(focus), m_ker(ker),
          m_oracle(oracle), m_paretomax(paretomax), m_epsilon(epsilon), m_e(seed), m_seed(seed), m_verbosity(0u),
          m_log(), m_res(), m_threshold(threshold), m_q(q), m_n_gen_mark(n_gen_mark), m_memory(memory), m_counter(0u),
          m_n_evalstop(1u), m_n_impstop(1u), m_gen_mark(1u), m_fevals(0u)
=======
     * @param[in] acc Accuracy parameter: for inequality and equality constraints .
     * @param[in] fstop Objective stopping criterion: when the objective value reaches this value, the algorithm is
     * stopped [for multi-objective, this applies to the first obj. only].
     * @param[in] impstop Improvement stopping criterion: if a positive integer is assigned here, the algorithm will
     * count the runs without improvements, if this number will exceed IMPSTOP value, the algorithm will be stopped.
     * @param[in] evalstop Evaluation stopping criterion: same as previous one, but with function evaluations
     * @param[in] focus Focus parameter: this parameter makes the search for the optimum greedier and more focused on
     * local improvements (the higher the greedier). If the value is very high, the search is more focused around the
     * current best solutions
     * @param[in] ker Kernel: number of solutions stored in the solution archive
     * @param[in] oracle Oracle parameter: this is the oracle parameter used in the penalty method
     * @param[in] paretomax Max number of non-dominated solutions: this regulates the max number of Pareto points to be
     * stored
     * @param[in] q This parameter is useful for managing the convergence speed towards the found minima (the smaller
     * the faster)
     * @param[in] threshold This parameter is coupled with q: when the generations reach the threshold then q is set to
     * 0.01 automatically
     * @param[in] omega_strategy This parameter determines how to compute the weights for the gaussian pdf (it can be
     * done in two different ways)
     * @param[in] n_gen_mark This parameters determines the convergence speed of the standard deviations values
     * @param[in] epsilon Pareto precision: the smaller this parameter, the higher the chances to introduce a new
     * solution in the Pareto front
     * @param seed seed used by the internal random number generator (default is random)
     * @throws std::invalid_argument if \p acc is not \f$ >=0 \f$, \p impstop is not a
     * positive integer, \p evalstop is not a positive integer, \p focus is not \f$ >=0 \f$, \p ants is not a positive
     * integer, \p ker is not a positive integer, \p oracle is not positive, \p paretomax is not a positive integer, \p
     * threshold is not \f$ \in [1,gen] \f$, \p omega_strategy is not \f$ ==1 or ==2 \f$, \p epsilon is not \f$ \in
     * [0,1[\f$, \p q is not \f$ >=0 \f$
     */

    g_aco(unsigned gen = 100u, unsigned ker = 63, double acc = 0, double fstop = 0.0000001, unsigned impstop = 10000,
          unsigned evalstop = 10000, double focus = 0., double oracle = 0., unsigned paretomax = 10, double q = 1.0,
          unsigned threshold = 1000, unsigned omega_strategy = 2, unsigned n_gen_mark = 6, double epsilon = 0.9,
          unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_acc(acc), m_fstop(fstop), m_impstop(impstop), m_evalstop(evalstop), m_focus(focus), m_ker(ker),
          m_oracle(oracle), m_paretomax(paretomax), m_epsilon(epsilon), m_e(seed), m_seed(seed), m_verbosity(0u),
          m_log(), m_res(), m_threshold(threshold), m_q(q), m_omega_strategy(omega_strategy), m_n_gen_mark(n_gen_mark)
>>>>>>> origin/master
    {
        if (acc < 0.) {
            pagmo_throw(std::invalid_argument, "The accuracy parameter must be >=0, while a value of "
                                                   + std::to_string(acc) + " was detected");
        }
        if (focus < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The focus parameter must be >=0  while a value of " + std::to_string(focus) + " was detected");
        }
<<<<<<< HEAD
=======
        if (oracle < 0.) {
            pagmo_throw(std::invalid_argument, "The oracle parameter must be >=0, while a value of "
                                                   + std::to_string(oracle) + " was detected");
        }
>>>>>>> origin/master

        if (epsilon >= 1. || epsilon < 0.) {
            pagmo_throw(std::invalid_argument, "The Pareto precision parameter must be in [0, 1[, while a value of "
                                                   + std::to_string(epsilon) + " was detected");
        }
<<<<<<< HEAD
        if ((threshold < 1 || threshold > gen) && gen != 0 && memory == false) {
            pagmo_throw(std::invalid_argument,
                        "If memory is inactive, the threshold parameter must be either in [1,m_gen] while a value of "
                            + std::to_string(threshold) + " was detected");
        }
        if (threshold < 1 && gen != 0 && memory == true) {
            pagmo_throw(std::invalid_argument,
                        "If memory is active, the threshold parameter must be >=1 while a value of "
                            + std::to_string(threshold) + " was detected");
=======

        if (omega_strategy != 1 && omega_strategy != 2) {
            pagmo_throw(std::invalid_argument, "The omega strategy parameter must be either 1 or 2 while a value of "
                                                   + std::to_string(omega_strategy) + " was detected");
        }
        if (threshold < 1 || threshold > gen) {
            pagmo_throw(std::invalid_argument, "The threshold parameter must be either in [1,m_gen] while a value of "
                                                   + std::to_string(threshold) + " was detected");
        }
        if (q < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The threshold parameter must be >=0 while a value of " + std::to_string(q) + " was detected");
>>>>>>> origin/master
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for the requested number of generations.
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throw std::invalid_argument if pop.get_problem() is stochastic.
     */

    population evolve(population pop) const
    {
<<<<<<< HEAD
        // If the memory is active, we increase the counter:
        if (m_memory == true) {
            ++m_counter;
        }

        // We store some useful variables:
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto dim = prob.get_nx();   // This getter does not return a const reference but a copy of the number of
                                    // continuous variables
=======
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto dim = prob.get_nx(); // This getter does not return a const reference but a copy of the number of variables
>>>>>>> origin/master
        auto pop_size = pop.size(); // Population size
        unsigned count_screen = 1u; // regulates the screen output

        // Note that the number of equality and inequality constraints has to be set up manually in the problem
<<<<<<< HEAD
        // definition, otherwise PaGMO assumes that there aren't any.
        auto n_obj = prob.get_nobj();
        auto n_ec = prob.get_nec();
        auto n_ic = prob.get_nic();
        auto n_f = prob.get_nf(); // n_f=prob.get_nobj()+prob.get_nec()+prob.get_nic()

        // Other useful variables are stored:
        std::vector<vector_double> sol_archive(m_ker, vector_double(1 + dim + n_f, 1));
        vector_double omega(m_ker);
        vector_double prob_cumulative(m_ker);
        std::uniform_real_distribution<> dist(0, 1);
        std::normal_distribution<double> gauss{0., 1.};
=======
        // definition, otherwise PaGMO assumes that there aren't any
        auto n_obj = prob.get_nobj();
        auto n_ec = prob.get_nec();
        auto n_ic = prob.get_nic();
        auto n_all = prob.get_nf(); // n_all=prob.get_nobj()+prob.get_nec()+prob.get_nic()

        // We define the variables which will count the runs without improvements in both the best solution of the
        // solution archive and in the the solution archive as a whole:
        unsigned n_impstop = 1;
        unsigned n_evalstop = 1;
        // We declare the variable which will count the number of function evaluations:
        unsigned fevals = 0;

        // Other useful variables are stored:
        unsigned gen_mark = 1;
        std::vector<vector_double> sol_archive(m_ker, vector_double(1 + dim + n_all, 1));
        vector_double omega;
        vector_double prob_cumulative;
>>>>>>> origin/master

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
<<<<<<< HEAD
        if (prob.get_nix() != 0u) {
            pagmo_throw(std::invalid_argument, "Integer variables detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
=======
>>>>>>> origin/master
        if (m_gen == 0u) {
            return pop;
        }
        // I verify that the solution archive is smaller or equal than the population size
        if (m_ker > pop_size) {
            pagmo_throw(std::invalid_argument,
                        get_name() + " cannot work with a solution archive bigger than the population size");
        }
<<<<<<< HEAD
        if (n_obj != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }

=======
>>>>>>> origin/master
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Main ACO loop over generations:
<<<<<<< HEAD
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
=======
        for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++) {

>>>>>>> origin/master
            // At each generation we make a copy of the population into popnew
            population popnew(pop);

            // In the case the algorithm is multi-objective, a decomposition strategy is applied:
            if (prob.get_nobj() > 1u) {
                // THIS PART HAS NOT BEEN DEFINED YET
            }
            // I otherwise proceed with a single-objective algorithm:
            else {
<<<<<<< HEAD
                auto dvs = pop.get_x(); // note that pop.get_x()[n][k] goes through the different individuals of the
=======

                auto X = pop.get_x();   // note that pop.get_x()[n][k] goes through the different individuals of the
>>>>>>> origin/master
                                        // population (index n) and the number of variables (index k) the number of
                                        // variables can be easily be deduced from counting the bounds.
                auto fit = pop.get_f(); // The following returns a vector of vectors in which objectives, equality and
                                        // inequality constraints are concatenated,for each individual

                // I check whether the maximum number of function evaluations or improvements has been exceeded:
<<<<<<< HEAD
                if (m_impstop != 0 && m_n_impstop >= m_impstop) {
=======
                if (m_impstop != 0 && n_impstop >= m_impstop) {
>>>>>>> origin/master
                    std::cout << "max number of impstop exceeded" << std::endl;
                    return pop;
                }

<<<<<<< HEAD
                if (m_evalstop != 0 && m_n_evalstop >= m_evalstop) {
=======
                if (m_evalstop != 0 && n_evalstop >= m_evalstop) {
>>>>>>> origin/master
                    std::cout << "max number of evalstop exceeded" << std::endl;
                    return pop;
                }

                // 1 - compute penalty functions

                // I declare some useful variables:
<<<<<<< HEAD
                vector_double penalties(pop_size);

                for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
                    // Penalty computation is here executed:
                    penalties[i] = penalty_computation(fit[i], pop, n_obj, n_ec, n_ic);
=======
                vector_double penalties;

                for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
                    // I first verify whether there is a solution that is smaller or equal the fstop parameter (in the
                    // case that this latter is different than zero)
                    if (m_fstop != 0. && std::abs(fit[i][0]) <= m_fstop) {
                        std::cout << "Fitness value:" << std::endl;
                        std::cout << fit[i][0] << std::endl;
                        std::cout << "if a value of zero is desired as fstop, please insert a very small value instead "
                                     "(e.g. 0.0000001)"
                                  << std::endl;
                        return pop;
                    }

                    // Penalty computation is here executed:
                    penalties.push_back(penalty_computation(fit[i], n_obj, n_ec, n_ic));
>>>>>>> origin/master
                }

                // 2 - update and sort solutions in the sol_archive, based on the computed penalties

<<<<<<< HEAD
                // I declare a vector where the penalties are sorted:
                vector_double sorted_penalties(penalties);
                // This sorts the penalties from the smallest ([0]) to the biggest ([end])
                std::sort(sorted_penalties.begin(), sorted_penalties.end(), detail::less_than_f<double>);
                // I declare a vector where I will store the positions of the various individuals:
                std::vector<decltype(penalties.size())> sort_list(penalties.size());
                // We fill it with 0,1,2,3,...,K
                std::iota(std::begin(sort_list), std::end(sort_list), 0);
                std::sort(sort_list.begin(), sort_list.end(),
                          [&penalties](decltype(penalties.size()) idx1, decltype(penalties.size()) idx2) {
                              return detail::less_than_f(penalties[idx1], penalties[idx2]);
                          });

                if (m_memory == true && m_counter > 1) {
                    sol_archive = m_sol_archive;
                }
                if (gen == 1 && m_counter < 2) {
=======
                // I declare a vector where I will store the positions of the various individuals:
                std::vector<unsigned long> sort_list;
                // I declare a vector where the penalties are sorted:
                vector_double sorted_penalties(penalties);
                // This sorts the penalties from the smallest ([0]) to the biggest ([end])
                std::sort(sorted_penalties.begin(), sorted_penalties.end());

                // I now create a vector where I store the position of the stored values. This will help
                // us to find the corresponding individuals and their objective values, later on
                for (decltype(penalties.size()) j = 0u; j < penalties.size(); ++j) {
                    int count = 0;
                    for (decltype(penalties.size()) i = 0u; i < penalties.size() && count == 0; ++i) {
                        if (sorted_penalties[j] == penalties[i]) {
                            if (j == 0) {
                                sort_list.push_back(i);
                                count = 1;

                            } else {
                                // with the following piece of code I avoid to store the same position in case that two
                                // another element exist with the same value
                                int count_2 = 0;
                                for (decltype(sort_list.size()) jj = 0u; jj < sort_list.size() && count_2 == 0; ++jj) {
                                    if (sort_list[jj] == i) {
                                        count_2 = 1;
                                    }
                                }
                                if (count_2 == 0) {
                                    sort_list.push_back(i);
                                    count = 1;
                                }
                            }
                        }
                    }
                }

                if (gen == 1) {
>>>>>>> origin/master

                    // We initialize the solution archive (sol_archive). This is done by storing the individuals from
                    // the best one (in terms of penalty), placed in the first row, to the worst one, placed in the last
                    // row:
                    for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
                        sol_archive[i][0] = penalties[sort_list[i]];
<<<<<<< HEAD
                        for (decltype(dim) k = 0; k < dim; ++k) {
                            sol_archive[i][k + 1] = dvs[sort_list[i]][k];
                        }
                        for (decltype(n_f) j = 0; j < n_f; ++j) {
                            sol_archive[i][j + 1 + dim] = fit[sort_list[i]][j];
                        }
                    }
                    if (m_memory == true) {
                        m_sol_archive = sol_archive;
                    }

                } else {
                    update_sol_archive(pop, sorted_penalties, sort_list, sol_archive, n_ec);
                    if (m_memory == true) {
                        m_sol_archive = sol_archive;
                    }
                }

                // 3 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)

                if ((m_verbosity > 0u && gen != m_gen) || (m_verbosity > 0u && m_memory == true)) {
                    // Every m_verbosity generations print a log line
                    if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                        auto best_fit = sol_archive[0][1 + dim];
=======
                        for (decltype(dim) J = 0; J < dim; ++J) {
                            sol_archive[i][J + 1] = X[sort_list[i]][J];
                        }
                        for (decltype(n_all) J = 0; J < n_all; ++J) {
                            sol_archive[i][J + 1 + dim] = fit[sort_list[i]][J];
                        }
                    }
                } else {
                    update_sol_archive(gen_mark, pop, sorted_penalties, sort_list, n_impstop, n_evalstop, sol_archive);
                }

                // 3 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
                if (m_verbosity > 0u) {
                    // Every m_verbosity generations print a log line
                    if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                        auto best_fit = sol_archive[0][1 + dim];
                        auto worst_fit = sol_archive[m_ker - 1][1 + dim];
>>>>>>> origin/master
                        double dx = 0., dp = 0.;
                        // The population flattness in variables
                        for (decltype(dim) i = 0u; i < dim; ++i) {
                            dx += std::abs(sol_archive[m_ker - 1][1 + i] - sol_archive[0][1 + i]);
                        }
                        // The population flattness in penalty
                        dp = std::abs(sol_archive[m_ker - 1][0] - sol_archive[0][0]);
                        // Every line print the column names
<<<<<<< HEAD
                        if (m_memory == false) {
                            if (count_screen % 50u == 1u) {
                                print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                                      "Best:", std::setw(15), "Kernel:", std::setw(15), "Oracle:", std::setw(15),
                                      "dx:", std::setw(15), std::setw(15), "dp:", '\n');
                            }

                        } else if ((m_memory == true && m_counter == 1)
                                   || (m_memory == true && m_counter % 50u == 1u)) {
                            print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                                  "Best:", std::setw(15), "Kernel:", std::setw(15), "Oracle:", std::setw(15),
                                  "dx:", std::setw(15), std::setw(15), "dp:", '\n');
                        }
                        print(std::setw(7), gen, std::setw(15), m_fevals, std::setw(15), best_fit, std::setw(15), m_ker,
                              std::setw(15), m_oracle, std::setw(15), dx, std::setw(15), dp, '\n');

                        ++count_screen;
                        // Logs
                        m_log.emplace_back(gen, m_fevals, best_fit, m_ker, m_oracle, dx, dp);
=======
                        if (count_screen % 50u == 1u) {
                            print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals", std::setw(15),
                                  "Best:", std::setw(15), "Kernel", std::setw(15), "Worst:", std::setw(15), "Oracle",
                                  std::setw(15), "dx:", std::setw(15), std::setw(15), "dp:", '\n');
                        }

                        print(std::setw(7), gen, std::setw(15), fevals, std::setw(15), best_fit, std::setw(15), m_ker,
                              std::setw(15), worst_fit, std::setw(15), m_oracle, std::setw(15), dx, std::setw(15), dp,
                              '\n');

                        ++count_screen;
                        // Logs
                        m_log.emplace_back(gen, best_fit, m_ker, worst_fit, m_oracle, dx, dp);
>>>>>>> origin/master
                    }
                }

                // 4 - compute pheromone values
<<<<<<< HEAD

                vector_double sigma(dim);
                pheromone_computation(gen, prob_cumulative, omega, sigma, pop, sol_archive);

                // 5 - use pheromone values to generate new ants (i.e., individuals)

                // I create the vector of vectors where I will store all the new ants which will be generated:
                std::vector<vector_double> new_ants(pop_size, vector_double(dim, 1));
                generate_new_ants(popnew, dist, gauss, prob_cumulative, sigma, new_ants, sol_archive);

                for (population::size_type i = 0; i < pop_size; ++i) {
                    vector_double ant(dim);
                    // I compute the fitness for each new individual which was generated in the generated_new_ants(..)
                    // function
                    for (decltype(new_ants[i].size()) ii = 0u; ii < new_ants[i].size(); ++ii) {
                        ant[ii] = new_ants[i][ii];
                    }

                    auto fitness = prob.fitness(ant);
                    ++m_fevals;
=======
                vector_double sigma;
                pheromone_computation(gen_mark, gen, prob_cumulative, omega, sigma, pop, sol_archive);

                // 5 - use pheromone values to generate new ants (i.e., individuals)
                // I create the vector of vectors where I will store all the new ants which will be generated:
                std::vector<vector_double> new_ants;
                generate_new_ants(popnew, prob_cumulative, omega, sigma, new_ants, gen, sol_archive);

                for (population::size_type i = 0; i < pop_size; ++i) {
                    vector_double ant;
                    // I compute the fitness for each new individual which was generated in the generated_new_ants(..)
                    // function
                    for (decltype(new_ants[i].size()) ii = 0u; ii < new_ants[i].size(); ++ii) {
                        ant.push_back(new_ants[i][ii]);
                    }

                    auto fitness = prob.fitness(ant);
                    ++fevals;
>>>>>>> origin/master
                    // I set the individuals for the next generation
                    pop.set_xf(i, ant, fitness);
                }

<<<<<<< HEAD
                // The oracle parameter is updated after each optimization run, if needed:
                if (sol_archive[0][1 + dim] < m_oracle) {
                    double residual = 0.0;
                    for (decltype(m_ker) rows = 0u; rows < m_ker; ++rows) {
                        if (rows == 0u && n_ic == 0 && n_ec == 0) {
                            m_oracle = sol_archive[0][1 + dim];

                        } else {
                            vector_double f(n_obj + n_ec + n_ic);
                            for (decltype(n_obj + n_ec + n_ic) jj = 0u; jj < n_obj + n_ec + n_ic; ++jj) {
                                f[jj] = sol_archive[rows][1 + dim + jj];
                            }

                            // Here we compute the L_2 norm of the penalty violations
                            auto violation_ic = detail::test_eq_constraints(f.data() + n_obj, f.data() + n_obj + n_ec,
                                                                            prob.get_c_tol().data());
                            auto violation_ec = detail::test_ineq_constraints(
                                f.data() + n_obj + n_ec, f.data() + f.size(), prob.get_c_tol().data() + n_ec);
                            residual = std::sqrt(std::pow(violation_ic.second, 2) + std::pow(violation_ec.second, 2));

                            if (rows == 0u && residual == 0.0) {
                                m_oracle = sol_archive[0][1 + dim];
                            }
                        }

                        double fitness_value = sol_archive[rows][1 + dim];
                        double alpha = 0.0;
                        double diff = std::abs(fitness_value - m_oracle); // I define this value which I will use often

                        if (fitness_value > m_oracle && residual < diff / 3.0) {
                            alpha = (diff * (6.0 * std::sqrt(3.0) - 2.0) / (6.0 * std::sqrt(3)) - residual)
                                    / (diff - residual);

                        } else if (fitness_value > m_oracle && residual >= diff / 3.0 && residual <= diff) {
                            alpha = 1.0 - 1.0 / (2.0 * std::sqrt(diff / residual));

                        } else if (fitness_value > m_oracle && residual > diff) {
                            alpha = 1.0 / 2.0 * std::sqrt(diff / residual);
                        }

                        // I can now compute the penalty function value
                        if (fitness_value > m_oracle || residual > 0) {
                            sol_archive[rows][0] = alpha * diff + (1 - alpha) * residual;

                        } else if (fitness_value <= m_oracle && residual == 0) {
                            sol_archive[rows][0] = -diff;
                        }
                    }
                    if (m_memory == true) {
                        m_sol_archive = sol_archive;
                    }
                }

            } // end of single objective part

        } // end of main ACO loop

        // Before returning the final population I make sure that the solution archive is included and the oracle
        // parameter is finally updated:
        if (m_memory == false) {
            for (decltype(m_ker) i_ker = 0; i_ker < m_ker; ++i_ker) {

                vector_double ant_final(dim);
                vector_double fitness_final(n_f);
                for (decltype(dim) ii_dim = 0u; ii_dim < dim; ++ii_dim) {
                    ant_final[ii_dim] = sol_archive[i_ker][1 + ii_dim];
                }

                for (decltype(n_f) ii_f = 0u; ii_f < n_f; ++ii_f) {
                    fitness_final[ii_f] = sol_archive[i_ker][1 + dim + ii_f];
                }
                pop.set_xf(i_ker, ant_final, fitness_final);
            }
        }

        if (m_verbosity > 0u && m_memory == false) {
            if (m_gen % m_verbosity == 1u || m_verbosity == 1u) {
                double dx = 0., dp = 0.;
                // The population flattness in variables
                for (decltype(dim) i = 0u; i < dim; ++i) {
                    dx += std::abs(sol_archive[m_ker - 1][1 + i] - sol_archive[0][1 + i]);
                }
                // The population flattness in penalty
                dp = std::abs(sol_archive[m_ker - 1][0] - sol_archive[0][0]);

                if (m_gen == 1) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Kernel:", std::setw(15), "Oracle:", std::setw(15), "dx:", std::setw(15), std::setw(15),
                          "dp:", '\n');
                }
                print(std::setw(7), m_gen, std::setw(15), m_fevals, std::setw(15), pop.champion_f()[0], std::setw(15),
                      m_ker, std::setw(15), m_oracle, std::setw(15), dx, std::setw(15), dp, '\n');

                // Logs
                m_log.emplace_back(m_gen, m_fevals, pop.champion_f()[0], m_ker, m_oracle, dx, dp);
            }
        }
=======
                // The oracle parameter is updated after each optimization run:
                if (sol_archive[0][1 + dim] < m_oracle && m_res == 0) {
                    m_oracle = sol_archive[0][1 + dim];

                    for (decltype(m_ker) rows = 0u; rows < m_ker; ++rows) {
                        double fitness_value = sol_archive[rows][1 + dim];
                        double alpha = 0;
                        double diff = std::abs(fitness_value - m_oracle); // I define this value which I will use often

                        if (fitness_value > m_oracle && m_res < diff / 3.0) {
                            alpha
                                = (diff * (6.0 * std::sqrt(3.0) - 2.0) / (6.0 * std::sqrt(3)) - m_res) / (diff - m_res);

                        } else if (fitness_value > m_oracle && m_res >= diff / 3.0 && m_res <= diff) {

                            alpha = 1.0 - 1.0 / (2.0 * std::sqrt(diff / m_res));

                        } else if (fitness_value > m_oracle && m_res > diff) {

                            alpha = 1.0 / 2.0 * std::sqrt(diff / m_res);
                        }

                        // I can now compute the penalty function value
                        if (fitness_value > m_oracle || m_res > 0) {
                            sol_archive[rows][0] = alpha * diff + (1 - alpha) * m_res;

                        } else if (fitness_value <= m_oracle && m_res == 0) {
                            sol_archive[rows][0] = -diff;
                        }
                    }
                }
            }
        } // end of main ACO loop

>>>>>>> origin/master
        return pop;
    }
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    }
    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned get_seed() const
    {
        return m_seed;
    }
    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 1):
     * @code{.unparsed}
<<<<<<< HEAD
     *Gen:        Fevals:          Best:        Kernel:        Oracle:            dx:            dp:
     *  1              0        179.464             13            100        4.33793          47876
     *  2             15         14.205             13            100        5.20084        5928.12
     *  3             30         14.205             13         14.205        1.24173        1037.44
     *  4             45         14.205             13         14.205        3.05807         395.89
     *  5             60        7.91087             13         14.205       0.711446        286.599
     *  6             75        2.81437             13        7.91087        5.80451        71.8174
     *  7             90        2.81437             13        2.81437        1.90561        48.3829
     *  8            105        2.81437             13        2.81437         1.3072        26.9496
     *  9            120         1.4161             13        2.81437        1.61732        10.6527
     * 10            150         1.4161             13         1.4161        2.54262        3.67034
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness function
     *value found until that generation, Kernel is the kernel size, Oracle is the oracle parameter value, dx is the
     *flatness in the individuals, dp is the flatness in the penalty function values.
=======
     * Gen:        Fevals:        ideal1:        ideal2:        ideal3:
     *   1              0      0.0257554       0.267768       0.974592
     *   2             52      0.0257554       0.267768       0.908174
     *   3            104      0.0257554       0.124483       0.822804
     *   4            156      0.0130094       0.121889       0.650099
     *   5            208     0.00182705      0.0987425       0.650099
     *   6            260      0.0018169      0.0873995       0.509662
     *   7            312     0.00154273      0.0873995       0.492973
     *   8            364     0.00154273      0.0873995       0.471251
     *   9            416    0.000379582      0.0873995       0.471251
     *  10            468    0.000336743      0.0855247       0.432144
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used. The ideal point of the current
     * population follows cropped to its 5th component.
>>>>>>> origin/master
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    }
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }
    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned get_gen() const
    {
        return m_gen;
    }
    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
<<<<<<< HEAD
        return "GACO: Ant Colony Optimization";
=======
        return "g_aco:";
>>>>>>> origin/master
    }
    /// Extra informations
    /**
     * Returns extra information on the algorithm.
     *
     * @return an <tt> std::string </tt> containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tGenerations: ", m_gen);
        stream(ss, "\n\tAccuracy parameter: ", m_acc);
<<<<<<< HEAD
=======
        stream(ss, "\n\tObjective stopping criterion: ", m_fstop);
>>>>>>> origin/master
        stream(ss, "\n\tImprovement stopping criterion: ", m_impstop);
        stream(ss, "\n\tEvaluation stopping criterion: ", m_evalstop);
        stream(ss, "\n\tFocus parameter: ", m_focus);
        stream(ss, "\n\tKernel: ", m_ker);
        stream(ss, "\n\tOracle parameter: ", m_oracle);
        stream(ss, "\n\tMax number of non-dominated solutions: ", m_paretomax);
<<<<<<< HEAD
        stream(ss, "\n\tPareto precision: ", m_epsilon);
        stream(ss, "\n\tPseudo-random number generator (Marsenne Twister 19937): ", m_e);
=======
        stream(ss, "\n\tThreshold: ", m_threshold);
        stream(ss, "\n\tq parameter: ", m_q);
        stream(ss, "\n\tOmega strategy parameter: ", m_omega_strategy);
        stream(ss, "\n\tLimit number of gen_mark: ", m_n_gen_mark);
        stream(ss, "\n\tPareto precision: ", m_epsilon);
        stream(ss, "\n\tDistribution index for mutation: ", m_e);
>>>>>>> origin/master
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);

        return ss.str();
    }
    /// Get log
<<<<<<< HEAD
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a gaco::log_line_type containing: gen, m_fevals, best_fit, m_ker,
     * m_oracle, dx, dp
     * as described in gaco::set_verbosity
     * @return an <tt>std::vector</tt> of gaco::log_line_type containing the logged values gen, m_fevals,
     * best_fit, m_ker, m_oracle, dx, dp
     */
=======

>>>>>>> origin/master
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
<<<<<<< HEAD
        ar(m_gen, m_acc, m_impstop, m_evalstop, m_focus, m_ker, m_oracle, m_paretomax, m_epsilon, m_e, m_seed,
           m_verbosity, m_log, m_res, m_threshold, m_q, m_n_gen_mark, m_memory, m_counter, m_n_evalstop, m_n_impstop,
           m_gen_mark, m_fevals);
    }

private:
    /**
     * Function which computes the penalty function values for each individual of the population
     *
     * @param[in] f Fitness values: vector in which the objective functions values, equality constraints and inequality
     * constraints are stored for each passed individual
     * @param[in] pop Population: the population of individuals is passed
     * @param[in] nobj Number of objectives: the number of objectives is passed
     * @param[in] nec Number of equality constraints: the number of equality constraints is passed
     * @param[in] nic Number of inequality constraints: the number of inequality constraints is passed
     */

    double penalty_computation(const vector_double &f, const population &pop, const unsigned long long nobj,
                               const unsigned long long nec, const unsigned long long nic) const
    {
        const auto &prob = pop.get_problem();

        // The residual function variable is assigned:
        m_res = 0.0;

        if (nic != 0 || nec != 0) {
            // I first retrieve the tolerance vector:
            auto tol_vec = prob.get_c_tol();
            // Here we compute the L_2 norm of the constraint violations
            auto violation_ic = detail::test_eq_constraints(f.data() + nobj, f.data() + nobj + nec, tol_vec.data());
            auto violation_ec
                = detail::test_ineq_constraints(f.data() + nobj + nec, f.data() + f.size(), tol_vec.data() + nec);
            m_res = std::sqrt(std::pow(violation_ic.second, 2) + std::pow(violation_ec.second, 2));
=======
        ar(m_gen, m_acc, m_fstop, m_impstop, m_evalstop, m_focus, m_ker, m_oracle, m_paretomax, m_epsilon, m_e, m_seed,
           m_verbosity, m_log, m_res, m_q, m_threshold);
    }

private:
    double penalty_computation(const vector_double &f, const unsigned nfunc, const unsigned nec,
                               const unsigned nic) const
    {
        /**
         * Function which computes the penalty function values for each individual of the population
         *
         * @param[in] f Fitness values: vector in which the objective functions values, equality constraints and
         * inequality constraints are stored for each passed individual
         * @param[in] nfunc Number of objectives: the number of objectives is passed
         * @param[in] nec Number of equality constraints: the number of equality constraints is passed
         * @param[in] nic Number of inequality constraints: the number of inequality constraints is passed
         */

        // The residual function variable is assigned:
        m_res = 0;

        if (nic != 0 || nec != 0 || nfunc > 1) {

            // In the fitness vector the objective vector, equality constraints vector, and inequality
            // constraints vector, respectively, are stored in this order
            double max_ec = f[nfunc];
            double min_ic = f[nfunc + nec];
            double ec_sum_1 = 0;
            double ic_sum_1 = 0;
            double ec_sum_2 = 0;
            double ic_sum_2 = 0;

            // I compute the sum over the equality and inequality constraints (to be used for the residual computation):
            for (decltype(nfunc + nec) i = nfunc; i < nfunc + nec; ++i) {
                ec_sum_1 = ec_sum_1 + std::abs(f[i]);
                ec_sum_2 = ec_sum_2 + std::pow(f[i], 2);

                if (i > nfunc && max_ec < f[i]) {
                    max_ec = f[i];
                }
            }

            for (decltype(nfunc + nec) j = nfunc + nec; j < nfunc + nec + nic; ++j) {
                ic_sum_1 = ic_sum_1 + std::min(std::abs(f[j]), 0.0);
                ic_sum_2 = ic_sum_2 + std::pow(std::min(std::abs(f[j]), 0.0), 2);

                if (j > nfunc + nec && min_ic > f[j]) {
                    min_ic = f[j];
                }
            }

            unsigned L = 2; // if L=1 --> it computes the L_1 norm,
                            // if L=2 --> it computes the L_2 norm,
                            // if L=3 --> it computes the L_inf norm

            if (L == 1) {
                m_res = ec_sum_1 - ic_sum_1;

            } else if (L == 2) {
                m_res = std::sqrt(ec_sum_2 + ic_sum_2);

            } else {
                m_res = std::max(max_ec, min_ic);
            }
>>>>>>> origin/master
        }

        // I compute the alpha parameter:

<<<<<<< HEAD
        // for single objective, for now, is enough to do:
        auto fitness = f[0];

        double alpha = 0.0;
        double diff = std::abs(fitness - m_oracle); // I define this value which I will use often
        double penalty = 0.0;                       // I declare the penalty function value variable
=======
        //(for single objective, for now, is enough to do:)
        auto fitness = f[0];

        double alpha = 0;
        double diff = std::abs(fitness - m_oracle); // I define this value which I will use often
        double penalty;                             // I declare the penalty function value variable
>>>>>>> origin/master

        if (fitness > m_oracle && m_res < diff / 3.0) {
            alpha = (diff * (6.0 * std::sqrt(3.0) - 2.0) / (6.0 * std::sqrt(3)) - m_res) / (diff - m_res);

        } else if (fitness > m_oracle && m_res >= diff / 3.0 && m_res <= diff) {
            alpha = 1.0 - 1.0 / (2.0 * std::sqrt(diff / m_res));

        } else if (fitness > m_oracle && m_res > diff) {
            alpha = 1.0 / 2.0 * std::sqrt(diff / m_res);
        }

        // I can now compute the penalty function value
        if (fitness > m_oracle || m_res > 0.) {
            penalty = alpha * diff + (1 - alpha) * m_res;

        } else if (fitness <= m_oracle && m_res == 0.) {
            penalty = -diff;
        }

        return penalty;
    }

<<<<<<< HEAD
    /**
     * Function which updates the solution archive, if better solutions are found
     *
     * @param[in] pop Population: the current population is passed
     * @param[in] sorted_vector Stored penalty vector: the vector in which the penalties of the current population are
     * stored from the best to the worst is passed
     * @param[in] sorted_list Positions of stored penalties: this represents the positions of the individuals wrt their
     * penalties as they are stored in the stored_vector
     * @param[in] sol_archive Solution archive: the solution archive is useful for retrieving the current
     * individuals (which will be the means of the new pdf)
     * @param[in] nec Number of equality constraints: the number of equality constraints is passed
     */

    void update_sol_archive(const population &pop, vector_double &sorted_vector,
                            std::vector<decltype(sorted_vector.size())> &sorted_list,
                            std::vector<vector_double> &sol_archive, const unsigned long long nec) const
    {

        auto variables = pop.get_x();
        auto fitness = pop.get_f();

        // This part can only be used for single obj:
        vector_double fitness_sol_arch(fitness[0].size());

        for (decltype(fitness[0].size()) j = 0u; j < fitness[0].size(); ++j) {
            fitness_sol_arch[j] = sol_archive[0][1 + variables[0].size() + j];
        }

        bool check = compare_fc(fitness_sol_arch, pop.champion_f(), nec, pop.get_problem().get_c_tol());

        // We increment the evalstop counter in case the best of the population is not better than the solution archive
        // best:
        if (check == true) {
            ++m_n_evalstop;
        }

        std::vector<vector_double> old_archive(sol_archive);
        std::vector<vector_double> temporary_archive(sol_archive);
        vector_double temporary_penalty(m_ker);
=======
    void update_sol_archive(unsigned &gen_mark, const population &pop, vector_double &sorted_vector,
                            std::vector<unsigned long> &sorted_list, unsigned &n_impstop, unsigned &n_evalstop,
                            std::vector<vector_double> &sol_archive) const
    {
        /**
         * Function which updates the solution archive, if better solutions are found
         *
         * @param[in] gen_mark Generation mark: the generation mark parameter is hereby defined and it will be used for
         * the standard deviations computation
         * @param[in] pop Population: the current population is passed
         * @param[in] stored_vector Stored penalty vector: the vector in which the penalties of the current population
         * are stored from the best to the worst is passed
         * @param[in] stored_list Positions of stored penalties: this represents the positions of the individuals wrt
         * their penalties as they are stored in the stored_vector
         * @param[in] n_impstop Impstop counter: it counts number of runs in which the sol_archive is not updated
         * @param[in] n_evalstop Evalstop counter: it counts the number of runs in which the best solution of the
         * sol_archive is not updated
         * @param[in] Solution_Archive Solution archive: the solution archive is useful for retrieving the current
         * individuals (which will be the means of the new pdf)
         */

        auto variables = pop.get_x();
        auto objectives = pop.get_f();
        std::vector<vector_double> old_archive(sol_archive);
        std::vector<vector_double> temporary_archive(sol_archive);
        vector_double temporary_penalty;
>>>>>>> origin/master

        // I now re-order the variables and objective vectors (remember that the objective vector also contains the eq
        // and ineq constraints). This is done only if at least the best solution of the current population is better
        // than the worst individual stored in the solution archive
        if (sorted_vector[0] < old_archive[m_ker - 1][0]) {
<<<<<<< HEAD

            // I reset the impstop counter since the solution archive is updated:
            m_n_impstop = 1;
=======
            // I reset the impstop parameter since the solution archive is updated:
            n_impstop = 1;
>>>>>>> origin/master
            // I save the variables, objectives, eq. constraints, ineq. constraints and penalties in three different
            // vectors:
            for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
                variables[i] = pop.get_x()[sorted_list[i]];
<<<<<<< HEAD
                fitness[i] = pop.get_f()[sorted_list[i]];
                temporary_penalty[i] = sol_archive[i][0];
=======
                objectives[i] = pop.get_f()[sorted_list[i]];
                temporary_penalty.push_back(sol_archive[i][0]);
>>>>>>> origin/master
            }
            std::vector<unsigned> saved_value_position;
            bool count = true;
            // I merge the new and old penalties:
            temporary_penalty.insert(temporary_penalty.end(), sorted_vector.begin(), sorted_vector.end());
            // I now reorder them depending on the penalty values (smallest first):
<<<<<<< HEAD
            std::sort(temporary_penalty.begin(), temporary_penalty.end(), detail::less_than_f<double>);

            saved_value_position.push_back(0);
            temporary_archive[0][0] = temporary_penalty[0];
            unsigned j;
            unsigned k = 1u;
=======
            std::sort(temporary_penalty.begin(), temporary_penalty.end());
            saved_value_position.push_back(0);
            temporary_archive[0][0] = temporary_penalty[0];
            unsigned j;
            unsigned k = 1;
>>>>>>> origin/master
            for (decltype(m_ker) i = 1u; i < 2 * m_ker && count == true; ++i) {
                j = i;
                if (i > saved_value_position.back()) {
                    // I check if the new penalties are better than the old ones of at least m_acc difference (user
                    // defined parameter).
                    while (temporary_penalty[j] - temporary_archive[k - 1][0] < m_acc && j < 2 * m_ker) {
                        ++j;
                    }
                    saved_value_position.push_back(j);
                    temporary_archive[k][0] = temporary_penalty[j];
                    if (saved_value_position.size() == m_ker) {
                        count = false;
                    }
                    ++k;
                }
            }

            // I now update the temporary archive according to the new individuals stored:
            bool count_2;
            for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
                count_2 = false;
<<<<<<< HEAD
                for (decltype(m_ker) jj = 0u; jj < m_ker && count_2 == false; ++jj) {
                    if (temporary_archive[i][0] == sol_archive[jj][0]) {
                        temporary_archive[i] = sol_archive[jj];
                        count_2 = true;
                    } else if (temporary_archive[i][0] == sorted_vector[jj]) {
                        for (decltype(variables[0].size()) i_var = 0u; i_var < variables[0].size(); ++i_var) {
                            temporary_archive[i][1 + i_var] = variables[jj][i_var];
                        }
                        for (decltype(fitness[0].size()) i_obj = 0u; i_obj < fitness[0].size(); ++i_obj) {
                            temporary_archive[i][1 + variables[0].size() + i_obj] = fitness[jj][i_obj];
=======
                for (decltype(m_ker) j = 0u; j < m_ker && count_2 == false; ++j) {
                    if (temporary_archive[i][0] == sol_archive[j][0]) {
                        temporary_archive[i] = sol_archive[j];
                        count_2 = true;
                    } else if (temporary_archive[i][0] == sorted_vector[j]) {
                        for (decltype(variables[0].size()) i_var = 0u; i_var < variables[0].size(); ++i_var) {
                            temporary_archive[i][1 + i_var] = variables[j][i_var];
                        }
                        for (decltype(objectives[0].size()) i_obj = 0u; i_obj < objectives[0].size(); ++i_obj) {
                            temporary_archive[i][1 + variables[0].size() + i_obj] = objectives[j][i_obj];
>>>>>>> origin/master
                        }
                        count_2 = true;
                    }
                }
            }

<<<<<<< HEAD
=======
            // I increase the evalstop parameter only if the best solution of the archive is updated:
            if (sol_archive[0][0] == temporary_archive[0][0]) {
                ++n_evalstop;

            } else {
                n_evalstop = 1;
            }

>>>>>>> origin/master
            // I hereby update the solution archive:
            for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
                sol_archive[i] = temporary_archive[i];
            }
        } else {
<<<<<<< HEAD
            ++m_n_impstop;
        }
        if (m_n_evalstop == 1 || m_n_evalstop > 2) {
            ++m_gen_mark;
        }
        if (m_gen_mark > m_n_gen_mark) {
            m_gen_mark = 1;
        }
    }

    /**
     * Function which computes the pheromone values (useful for generating offspring)
     *
     * @param[in] gen Generations: current generation number is passed
     * @param[in] prob_cumulative Cumulative probability vector: this vector will be crucial for the new individuals'
     * creation
     * @param[in] omega_vec Omega: the weights are passed to be modified (i.e., they are one of the pheromone values)
     * @param[in] sigma_vec Sigma: the standard deviations are passed to be modified (i.e., they are one of the
     * pheromone values)
     * @param[in] popul Population: the current population is passed
     * @param[in] sol_archive Solution archive: the solution archive is useful for retrieving the current individuals
     * (which will be the means of the new pdf)
     */
    void pheromone_computation(const unsigned gen, vector_double &prob_cumulative, vector_double &omega_vec,
                               vector_double &sigma_vec, const population &popul,
                               std::vector<vector_double> &sol_archive) const
    {

=======
            ++n_impstop;
            ++n_evalstop;
        }
        if (n_evalstop == 1 || n_evalstop > 2) {
            ++gen_mark;
        }
        if (gen_mark > m_n_gen_mark) {
            gen_mark = 1;
        }
    }

    void pheromone_computation(unsigned gen_mark, const unsigned gen, vector_double &prob_cumulative,
                               vector_double &omega_vec, vector_double &sigma_vec, const population &popul,
                               std::vector<vector_double> &sol_archive) const
    {

        /**
         * Function which computes the pheromone values (useful for generating offspring)
         *
         * @param[in] gen_mark Generation mark: this parameter is used to reduce the standard deviation
         * @param[in] gen Generations: current generation number is passed
         * @param[in] prob_cumulative Cumulative probability vector: this vector will be crucial for the new
         * individuals' creation
         * @param[in] omega_vec Omega: the weights are passed to be modified (i.e., they are one of the pheromone
         * values)
         * @param[in] sigma_vec Sigma: the standard deviations are passed to be modified (i.e., they are one of the
         * pheromone values)
         * @param[in] popul Population: the current population is passed
         * @param[in] sol_archive Solution archive: the solution archive is useful for retrieving the current
         * individuals (which will be the means of the new pdf)
         */

>>>>>>> origin/master
        const auto &prob = popul.get_problem();
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto n_con = prob.get_nx();

<<<<<<< HEAD
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
                                * exp(-std::pow(l - 1.0, 2) / (2.0 * std::pow(m_q, 2) * std::pow(m_ker, 2)));
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
        } else {
            if (m_counter == m_threshold) {
=======
        // Here I define the weights. Their definition depends on the user's selection of the m_omega_strategy parameter
        // (either 1 or 2)
        if (gen == 1 && m_omega_strategy == 1) {
            // We declare a vector with 'ker' doubles
            vector_double ker_vector(m_ker);
            // We fill it with 1,2,3,...,K
            std::iota(std::begin(ker_vector), std::end(ker_vector), 1);
            // We compute the sum of the elements
            double sum = std::accumulate(ker_vector.begin(), ker_vector.end(), 0);
            // We compute omega (first pheromone value):
            double omega;

            for (decltype(m_ker) k = 1; k <= m_ker; ++k) {
                omega = (m_ker - k + 1.0) / (sum);
                omega_vec.push_back(omega);
            }
        }
        if (gen == 1 && m_omega_strategy == 2) {
            if (gen >= m_threshold) {
>>>>>>> origin/master
                m_q = 0.01;
            }

            double omega_new;
            double sum_omega = 0;
<<<<<<< HEAD

            for (decltype(m_ker) l = 1; l <= m_ker; ++l) {
                omega_new = 1.0 / (m_q * m_ker * std::sqrt(2 * boost::math::constants::pi<double>()))
                            * exp(-std::pow(l - 1.0, 2) / (2.0 * std::pow(m_q, 2) * std::pow(m_ker, 2)));
                omega_vec[l - 1] = omega_new;
=======
            for (decltype(m_ker) l = 1; l <= m_ker; ++l) {
                omega_new = 1.0 / (m_q * m_ker * std::sqrt(2 * M_PI))
                            * exp(-std::pow(l - 1.0, 2) / (2.0 * std::pow(m_q, 2) * std::pow(m_ker, 2)));
                omega_vec.push_back(omega_new);
>>>>>>> origin/master
                sum_omega += omega_new;
            }

            for (decltype(m_ker) k = 0u; k < m_ker; ++k) {
                double cumulative = 0;
<<<<<<< HEAD
                for (decltype(m_ker) j = 0u; j <= k; ++j) {
                    cumulative += omega_vec[j] / sum_omega;
                }
                prob_cumulative[k] = cumulative;
            }
        }

        // We now compute the standard deviations (sigma):
=======
                for (decltype(m_ker) l = 0u; l <= k; ++l) {
                    cumulative += omega_vec[l] / sum_omega;
                }
                prob_cumulative.push_back(cumulative);
            }
        }

        // I now compute the standard deviations (sigma):
>>>>>>> origin/master
        for (decltype(n_con) h = 1; h <= n_con; ++h) {

            // I declare and define D_min and D_max:
            // at first I define D_min, D_max using the first two individuals stored in the sol_archive
            double d_min = std::abs(sol_archive[0][h] - sol_archive[1][h]);
<<<<<<< HEAD

            double d_max = std::abs(sol_archive[0][h] - sol_archive[1][h]);
=======
            vector_double d_min_vec;

            double d_max = std::abs(sol_archive[0][h] - sol_archive[1][h]);
            vector_double d_max_vec;
>>>>>>> origin/master

            // I loop over the various individuals of the variable:
            for (decltype(m_ker) count = 0; count < m_ker - 1.0; ++count) {

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

<<<<<<< HEAD
            // In case a value for the focus parameter (different than zero) is passed, this limits the maximum
            // value of the standard deviation
            if (m_focus != 0. && ((d_max - d_min) / gen > (ub[h - 1] - lb[h - 1]) / m_focus) && m_memory == false) {
                sigma_vec[h - 1] = (ub[h - 1] - lb[h - 1]) / m_focus;

            } else if (m_focus != 0. && ((d_max - d_min) / m_counter > (ub[h - 1] - lb[h - 1]) / m_focus)
                       && m_memory == true) {
                sigma_vec[h - 1] = (ub[h - 1] - lb[h - 1]) / m_focus;

            } else {
                sigma_vec[h - 1] = (d_max - d_min) / m_gen_mark;
=======
            d_min_vec.push_back(d_min);
            d_max_vec.push_back(d_max);

            if (m_focus != 0. && ((d_max - d_min) / gen > (ub[h - 1] - lb[h - 1]) / m_focus)) {
                // In case a value for the focus parameter (different than zero) is passed, this limits the maximum
                // value of the standard deviation
                sigma_vec.push_back((ub[h - 1] - lb[h - 1]) / m_focus);
            } else {
                sigma_vec.push_back((d_max - d_min) / gen_mark);
>>>>>>> origin/master
            }
        }
    }

<<<<<<< HEAD
    /**
     * Function which generates new individuals (i.e., ants)
     *
     * @param[in] pop Population: the current population of individuals is passed
     * @param[in] dist Uniform real distribution: the uniform real pdf is passed
     * @param[in] gauss_pdf Gaussian real distribution: the gaussian pdf is passed
     * @param[in] prob_cumulative Cumulative probability vector: this vector determines the probability for choosing the
     * pdf to generate new individuals
     * @param[in] sigma Sigma: one of the three pheromone values. These are the standard deviations which are used in
     * the multi-kernel gaussian probability distribution
     * @param[in] dvs_new New ants: in this vector the new ants which will be generated are stored
     * @param[in] sol_archive Solution archive: the solution archive is useful for retrieving the current individuals
     * (which will be the means of the new pdf)
     */
    void generate_new_ants(const population &pop, std::uniform_real_distribution<> dist,
                           std::normal_distribution<double> gauss_pdf, vector_double prob_cumulative,
                           vector_double sigma, std::vector<vector_double> &dvs_new,
                           std::vector<vector_double> &sol_archive) const
    {
=======
    void generate_new_ants(const population &pop, vector_double prob_cumulative, vector_double omega,
                           vector_double sigma, std::vector<vector_double> &X_new, double gen,
                           std::vector<vector_double> &sol_archive) const
    {
        /**
         * Function which generates new individuals (i.e., ants)
         *
         * @param[in] pop Population: the current population of individuals is passed
         * @param[in] prob_cumulative Cumulative probability vector: this vector determines the probability for choosing
         * the pdf to generate new individuals
         * @param[in] omega Omega: one of the three pheromone values. These are the weights which are used in the
         * multi-kernel gaussian probability distribution
         * @param[in] sigma Sigma: one of the three pheromone values. These are the standard deviations which are used
         * in the multi-kernel gaussian probability distribution
         * @param[in] X_new New ants: in this vector the new ants which will be generated are stored
         * @param[in] gen Current number of generation: this represents the current generation number
         * @param[in] sol_archive Solution archive: the solution archive is useful for retrieving the current
         * individuals (which will be the means of the new pdf)
         */
>>>>>>> origin/master

        const auto &prob = pop.get_problem();
        auto pop_size = pop.size();
        auto n_con = prob.get_nx();
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        vector_double fitness_old;
        vector_double fitness_new;

        // I hereby generate the new ants based on a multi-kernel gaussian probability density function. In particular,
        // I select one of the pdfs that make up the multi-kernel, by using the probability stored in the
        // prob_cumulative vector. A multi-kernel pdf is a weighted sum of several gaussian pdf.

<<<<<<< HEAD
        for (decltype(pop_size) j = 0u; j < pop_size; ++j) {
            vector_double dvs_new_j(n_con);

            double number = dist(m_e);
            double g_h = 0.0;
            unsigned long k_omega = 0u;

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
            for (decltype(n_con) h = 0u; h < n_con; ++h) {
                g_h = sol_archive[k_omega][1 + h] + sigma[h] * gauss_pdf(m_e);

                if (g_h < lb[h] || g_h > ub[h]) {

                    // We define the max number of attempts to reset the ant within the bounds,
                    // before placing it to the bounds themselves (in case every attempt fails)
                    unsigned attempts = 0u;
                    while ((g_h < lb[h] || g_h > ub[h]) && attempts < 20u) {
                        g_h = sol_archive[k_omega][1 + h] + sigma[h] * gauss_pdf(m_e);
                        ++attempts;
                    }
                    if (attempts == 20 && g_h > ub[h]) {
                        g_h = ub[h];
                    }
                    if (attempts == 20 && g_h < lb[h]) {
                        g_h = lb[h];
                    }
                }
                dvs_new_j[h] = g_h;
            }
            dvs_new[j] = dvs_new_j;
=======
        if (m_omega_strategy == 1) {
            for (decltype(pop_size) j = 0u; j < pop_size; ++j) {
                vector_double
                    X_new_j; // here I store all the variables associated with the j_th element of the sol_archive

                for (decltype(n_con) h = 0u; h < n_con; ++h) {
                    double g_h = 0;

                    for (decltype(sol_archive.size()) k = 0u; k < sol_archive.size(); ++k) {
                        // Mersenne twister PRNG
                        std::mt19937 generator(m_seed + gen + h + j);
                        std::normal_distribution<double> gauss_pdf{sol_archive[k][1 + h], sigma[h]};
                        g_h += omega[k] * gauss_pdf(generator);

                        // the pdf has the following form:
                        // G_h (t) = sum_{k=1}^{K} omega_{k,h} 1/(sigma_h * sqrt(2*pi)) * exp(- (t-mu_{k,h})^2 /
                        // (2*(sigma_h)^2) )
                        // I thus have all the elements to compute it (which I retrieved from the pheromone_computation
                        // function)
                    }

                    if (g_h < lb[h] || g_h > ub[h]) {

                        while (g_h < lb[h] || g_h > ub[h]) {
                            g_h = 0;
                            double index = pagmo::random_device::next();

                            for (decltype(sol_archive.size()) k = 0u; k < sol_archive.size(); ++k) {
                                if (gen > 9000) {

                                    if (k == 0) {
                                        // Mersenne twister PRNG
                                        std::mt19937 generator(m_seed + gen + h + j + index);
                                        std::normal_distribution<double> gauss_pdf{sol_archive[0][1 + h], sigma[h]};
                                        g_h += gauss_pdf(generator);
                                    }
                                } else {
                                    // Mersenne twister PRNG
                                    std::mt19937 generator(m_seed + gen + h + j + index);
                                    std::normal_distribution<double> gauss_pdf{sol_archive[k][1 + h], sigma[h]};
                                    g_h += omega[k] * gauss_pdf(generator);
                                }
                            }
                        }
                    }

                    X_new_j.push_back(g_h);
                }

                X_new.push_back(X_new_j);
            }
        }

        else if (m_omega_strategy == 2) {

            for (decltype(pop_size) j = 0u; j < pop_size; ++j) {
                vector_double X_new_j;

                double rd = pagmo::random_device::next();
                std::mt19937 generator_0(rd);
                std::uniform_real_distribution<> dist(0, 1);

                double number = dist(generator_0);
                double g_h = 0;
                double k_omega = -1;

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
                for (decltype(n_con) h = 0u; h < n_con; ++h) {
                    std::mt19937 generator(m_seed + gen + h + j);
                    std::normal_distribution<double> gauss_pdf{sol_archive[k_omega][1 + h], sigma[h]};
                    g_h = gauss_pdf(generator);

                    if (g_h < lb[h] || g_h > ub[h]) {

                        while (g_h < lb[h] || g_h > ub[h]) {
                            double index = pagmo::random_device::next();
                            std::mt19937 generator(m_seed + gen + h + j + index);
                            std::normal_distribution<double> gauss_pdf{sol_archive[k_omega][1 + h], sigma[h]};
                            g_h = gauss_pdf(generator);
                        }
                    }
                    X_new_j.push_back(g_h);
                }
                X_new.push_back(X_new_j);
            }
>>>>>>> origin/master
        }
    }

    unsigned m_gen;
    double m_acc;
<<<<<<< HEAD
=======
    double m_fstop;
>>>>>>> origin/master
    unsigned m_impstop;
    unsigned m_evalstop;
    double m_focus;
    unsigned m_ker;
    mutable double m_oracle;
    unsigned m_paretomax;
    double m_epsilon;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    mutable double m_res;
    unsigned m_threshold;
    mutable double m_q;
<<<<<<< HEAD
    unsigned m_n_gen_mark;
    bool m_memory;
    mutable unsigned m_counter;
    mutable std::vector<vector_double> m_sol_archive;
    mutable unsigned m_n_evalstop;
    mutable unsigned m_n_impstop;
    mutable unsigned m_gen_mark;
    mutable unsigned m_fevals;
=======
    unsigned m_omega_strategy;
    unsigned m_n_gen_mark;
>>>>>>> origin/master
};

} // namespace pagmo

<<<<<<< HEAD
PAGMO_REGISTER_ALGORITHM(pagmo::gaco)
=======
PAGMO_REGISTER_ALGORITHM(pagmo::g_aco)
>>>>>>> origin/master

#endif