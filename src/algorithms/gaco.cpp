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
#include <boost/serialization/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>

namespace pagmo
{

gaco::gaco(unsigned gen, unsigned ker, double q, double oracle, double acc, unsigned threshold, unsigned n_gen_mark,
           unsigned impstop, unsigned evalstop, double focus, bool memory, unsigned seed)
    : m_gen(gen), m_acc(acc), m_impstop(impstop), m_evalstop(evalstop), m_focus(focus), m_ker(ker), m_oracle(oracle),
      m_e(seed), m_seed(seed), m_verbosity(0u), m_log(), m_res(), m_threshold(threshold), m_q(q),
      m_n_gen_mark(n_gen_mark), m_memory(memory), m_counter(0u), m_sol_archive(), m_n_evalstop(1u), m_n_impstop(1u),
      m_gen_mark(1u), m_fevals(0u)
{
    if (acc < 0.) {
        pagmo_throw(std::invalid_argument,
                    "The accuracy parameter must be >=0, while a value of " + std::to_string(acc) + " was detected");
    }
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
        pagmo_throw(std::invalid_argument,
                    "If memory is active, the threshold parameter must be >=1 while a value of "
                        + std::to_string(threshold) + " was detected");
    }
    if (q < 0.) {
        pagmo_throw(std::invalid_argument,
                    "The convergence speed parameter must be >=0  while a value of " + std::to_string(q)
                        + " was detected");
    }
}

/// Algorithm evolve method
/**
 * Evolves the population for the requested number of generations.
 *
 * @param pop population to be evolved
 * @return evolved population
 * @throw std::invalid_argument if pop.get_problem() is stochastic.
 */
population gaco::evolve(population pop) const
{
    // If the memory is active, we increase the counter:
    if (m_memory == true) {
        ++m_counter;
    }

    // We store some useful variables:
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
    // allowed
    auto n_x = prob.get_nx(); // This getter does not return a const reference but a copy of the number of
    // all the variables
    auto pop_size = pop.size(); // Population size
    unsigned count_screen = 1u; // regulates the screen output

    // Note that the number of equality and inequality constraints has to be set up manually in the problem
    // definition, otherwise PaGMO assumes that there aren't any.
    auto n_obj = prob.get_nobj();
    auto n_ec = prob.get_nec();
    auto n_ic = prob.get_nic();
    auto n_f = prob.get_nf(); // n_f=prob.get_nobj()+prob.get_nec()+prob.get_nic()

    // Other useful variables are stored:
    std::vector<vector_double> sol_archive(m_ker, vector_double(1 + n_x + n_f, 1));
    vector_double omega(m_ker);
    vector_double prob_cumulative(m_ker);
    std::uniform_real_distribution<> dist(0, 1);
    std::normal_distribution<> gauss{0., 1.};
    vector_double penalties(pop_size);
    // I declare a vector where I will store the positions of the various individuals:
    std::vector<decltype(penalties.size())> sort_list(penalties.size());
    // I create the vector of vectors where I will store all the new ants which will be generated:
    std::vector<vector_double> new_ants(pop_size, vector_double(n_x, 1));
    vector_double ant(n_x);
    vector_double sigma(n_x);
    vector_double fitness(n_f);
    vector_double ant_final(n_x);
    vector_double fitness_final(n_f);

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this
    // particular algorithm.

    if (!pop_size) {
        return pop;
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
    if (n_obj != 1u) {
        pagmo_throw(std::invalid_argument,
                    "Multiple objectives detected in " + prob.get_name() + " instance. " + get_name()
                        + " cannot deal with them");
    }

    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    // Main ACO loop over generations:
    for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
        // At each generation we make a copy of the population into popold
        population popold(pop);

        const auto &dvs = pop.get_x(); // note that pop.get_x()[n][k] goes through the different individuals of the
        // population (index n) and the number of variables (index k) the number of
        // variables can be easily be deduced from counting the bounds.
        const auto &fit = pop.get_f(); // The following returns a vector of vectors in which objectives, equality and
        // inequality constraints are concatenated,for each individual

        // I check whether the maximum number of function evaluations or improvements has been exceeded:
        if (m_impstop != 0 && m_n_impstop >= m_impstop) {
            if (m_verbosity >= 1) {
                std::cout << "max number of impstop exceeded" << std::endl;
            }
            return pop;
        }

        if (m_evalstop != 0 && m_n_evalstop >= m_evalstop) {
            if (m_verbosity >= 1) {
                std::cout << "max number of evalstop exceeded" << std::endl;
            }
            return pop;
        }

        // 1 - compute penalty functions

        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            // Penalty computation is here executed:
            penalties[i] = penalty_computation(fit[i], pop, n_obj, n_ec, n_ic);
        }

        // 2 - update and sort solutions in the sol_archive, based on the computed penalties
        // We fill it with 0,1,2,3,...,K
        std::iota(std::begin(sort_list), std::end(sort_list), decltype(penalties.size())(0));
        std::sort(sort_list.begin(), sort_list.end(),
                  [&penalties](decltype(penalties.size()) idx1, decltype(penalties.size()) idx2) {
                      return detail::less_than_f(penalties[idx1], penalties[idx2]);
                  });

        if (m_memory == true && m_counter > 1) {
            sol_archive = m_sol_archive;
        }
        if (gen == 1 && m_counter < 2) {

            // We initialize the solution archive (sol_archive). This is done by storing the individuals from
            // the best one (in terms of penalty), placed in the first row, to the worst one, placed in the last
            // row:
            for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
                sol_archive[i][0] = penalties[sort_list[i]];
                for (decltype(n_x) k = 0; k < n_x; ++k) {
                    sol_archive[i][k + 1] = dvs[sort_list[i]][k];
                }
                for (decltype(n_f) j = 0; j < n_f; ++j) {
                    sol_archive[i][j + 1 + n_x] = fit[sort_list[i]][j];
                }
            }
            if (m_memory == true) {
                m_sol_archive = sol_archive;
            }

        } else {
            // This sorts the penalties from the smallest ([0]) to the biggest ([end])
            std::sort(penalties.begin(), penalties.end(), detail::less_than_f<double>);
            update_sol_archive(pop, penalties, sort_list, sol_archive);
            if (m_memory == true) {
                m_sol_archive = sol_archive;
            }
        }

        // 3 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)

        if ((m_verbosity > 0u && gen != m_gen) || (m_verbosity > 0u && m_memory == true)) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                auto best_fit = sol_archive[0][1 + n_x];
                double dx = 0., dp = 0.;
                // The population flattness in variables
                for (decltype(n_x) i = 0u; i < n_x; ++i) {
                    dx += std::abs(sol_archive[m_ker - 1][1 + i] - sol_archive[0][1 + i]);
                }
                // The population flattness in penalty
                dp = std::abs(sol_archive[m_ker - 1][0] - sol_archive[0][0]);
                // Every line print the column names
                if (m_memory == false) {
                    if (count_screen % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                              "Best:", std::setw(15), "Kernel:", std::setw(15), "Oracle:", std::setw(15),
                              "dx:", std::setw(15), std::setw(15), "dp:", '\n');
                    }

                } else if ((m_memory == true && m_counter == 1) || (m_memory == true && m_counter % 50u == 1u)) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Kernel:", std::setw(15), "Oracle:", std::setw(15), "dx:", std::setw(15), std::setw(15),
                          "dp:", '\n');
                }
                print(std::setw(7), gen, std::setw(15), m_fevals, std::setw(15), best_fit, std::setw(15), m_ker,
                      std::setw(15), m_oracle, std::setw(15), dx, std::setw(15), dp, '\n');

                ++count_screen;
                // Logs
                m_log.emplace_back(gen, m_fevals, best_fit, m_ker, m_oracle, dx, dp);
            }
        }

        // 4 - compute pheromone values
        pheromone_computation(gen, prob_cumulative, omega, sigma, pop, sol_archive);

        // 5 - use pheromone values to generate new ants (i.e., individuals)
        generate_new_ants(popold, dist, gauss, prob_cumulative, sigma, new_ants, sol_archive);

        if (m_bfe) {
            // bfe is available:
            vector_double ants(pop_size * new_ants[0].size());
            decltype(ants.size()) pos = 0u;

            for (population::size_type i = 0; i < pop_size; ++i) {
                // I compute the fitness for each new individual which was generated in the generated_new_ants(..)
                // function
                for (decltype(new_ants[i].size()) ii = 0u; ii < new_ants[i].size(); ++ii) {
                    ants[pos] = new_ants[i][ii];
                    ++pos;
                }
            }

            auto fitnesses = (*m_bfe)(prob, ants);
            m_fevals += pop_size;
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
                for (decltype(new_ants[i].size()) ii = 0u; ii < new_ants[i].size(); ++ii) {
                    ant[ii] = new_ants[i][ii];
                }

                auto ftns = prob.fitness(ant);
                ++m_fevals;
                // I set the individuals for the next generation
                pop.set_xf(i, ant, ftns);
            }
        }

        // If the current best of the population has not improved wrt to the previous one, then check=false
        bool check = compare_fc(pop.champion_f(), popold.champion_f(), n_ec, pop.get_problem().get_c_tol());

        // We increment the evalstop counter in case the best of the population is not better than the solution
        // archive best:
        if (check == false) {
            ++m_n_evalstop;
        } else {
            m_n_evalstop = 1u;
        }

        // The oracle parameter is updated after each optimization run, if needed:
        if (sol_archive[0][1 + n_x] < m_oracle) {
            double residual = 0.0;
            for (decltype(m_ker) rows = 0u; rows < m_ker; ++rows) {
                if (rows == 0u && n_ic == 0 && n_ec == 0) {
                    m_oracle = sol_archive[0][1 + n_x];

                } else {
                    vector_double f(n_obj + n_ec + n_ic);
                    for (decltype(n_obj + n_ec + n_ic) jj = 0u; jj < n_obj + n_ec + n_ic; ++jj) {
                        f[jj] = sol_archive[rows][1 + n_x + jj];
                    }

                    // Here we compute the L_2 norm of the penalty violations
                    auto violation_ic = detail::test_eq_constraints(f.data() + n_obj, f.data() + n_obj + n_ec,
                                                                    prob.get_c_tol().data());
                    auto violation_ec = detail::test_ineq_constraints(f.data() + n_obj + n_ec, f.data() + f.size(),
                                                                      prob.get_c_tol().data() + n_ec);
                    residual = std::sqrt(std::pow(violation_ic.second, 2) + std::pow(violation_ec.second, 2));

                    if (rows == 0u && residual == 0.0) {
                        m_oracle = sol_archive[0][1 + n_x];
                    }
                }

                double fitness_value = sol_archive[rows][1 + n_x];
                double alpha = 0.0;
                double diff = std::abs(fitness_value - m_oracle); // I define this value which I will use often

                if (fitness_value > m_oracle && residual < diff / 3.0) {
                    alpha = (diff * (6.0 * std::sqrt(3.0) - 2.0) / (6.0 * std::sqrt(3)) - residual) / (diff - residual);

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

    } // end of main ACO loop

    // Before returning the final population I make sure that the solution archive is included and the oracle
    // parameter is finally updated:
    if (m_memory == false) {
        for (decltype(m_ker) i_ker = 0; i_ker < m_ker; ++i_ker) {

            for (decltype(n_x) ii_dim = 0u; ii_dim < n_x; ++ii_dim) {
                ant_final[ii_dim] = sol_archive[i_ker][1 + ii_dim];
            }

            for (decltype(n_f) ii_f = 0u; ii_f < n_f; ++ii_f) {
                fitness_final[ii_f] = sol_archive[i_ker][1 + n_x + ii_f];
            }
            pop.set_xf(i_ker, ant_final, fitness_final);
        }
    }

    if (m_verbosity > 0u && m_memory == false) {
        if (m_gen % m_verbosity == 1u || m_verbosity == 1u) {
            double dx = 0., dp = 0.;
            // The population flattness in variables
            for (decltype(n_x) i = 0u; i < n_x; ++i) {
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
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void gaco::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Sets the batch function evaluation scheme
/**
 * @param b batch function evaluation object
 */
void gaco::set_bfe(const bfe &b)
{
    m_bfe = b;
}

/// Extra info
/**
 * Returns extra information on the algorithm.
 *
 * @return an <tt> std::string </tt> containing extra info on the algorithm
 */
std::string gaco::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tAccuracy parameter: ", m_acc);
    stream(ss, "\n\tImprovement stopping criterion: ", m_impstop);
    stream(ss, "\n\tEvaluation stopping criterion: ", m_evalstop);
    stream(ss, "\n\tFocus parameter: ", m_focus);
    stream(ss, "\n\tKernel: ", m_ker);
    stream(ss, "\n\tOracle parameter: ", m_oracle);
    stream(ss, "\n\tPseudo-random number generator (Marsenne Twister 19937): ", m_e);
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
void gaco::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_acc, m_impstop, m_evalstop, m_focus, m_ker, m_oracle, m_e, m_seed, m_verbosity, m_log,
                    m_res, m_threshold, m_q, m_n_gen_mark, m_memory, m_counter, m_sol_archive, m_n_evalstop,
                    m_n_impstop, m_gen_mark, m_fevals, m_bfe);
}

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

double gaco::penalty_computation(const vector_double &f, const population &pop, const unsigned long long nobj,
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
    }

    // I compute the alpha parameter:

    // for single objective, for now, is enough to do:
    auto fitness = f[0];

    double alpha = 0.0;
    double diff = std::abs(fitness - m_oracle); // I define this value which I will use often
    double penalty = 0.0;                       // I declare the penalty function value variable

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
 */

void gaco::update_sol_archive(const population &pop, vector_double &sorted_vector,
                              std::vector<decltype(sorted_vector.size())> &sorted_list,
                              std::vector<vector_double> &sol_archive) const
{

    auto variables = pop.get_x();
    auto fitness = pop.get_f();

    // This part can only be used for single obj:
    vector_double fitness_sol_arch(fitness[0].size());

    for (decltype(fitness[0].size()) j = 0u; j < fitness[0].size(); ++j) {
        fitness_sol_arch[j] = sol_archive[0][1 + variables[0].size() + j];
    }

    std::vector<vector_double> old_archive(sol_archive);
    std::vector<vector_double> temporary_archive(sol_archive);
    vector_double temporary_penalty(m_ker);
    vector_double archive_penalties(m_ker);

    // I now re-order the variables and objective vectors (remember that the objective vector also contains the eq
    // and ineq constraints). This is done only if at least the best solution of the current population is better
    // than the worst individual stored in the solution archive
    if (sorted_vector[0] < old_archive[m_ker - 1][0]) {

        // I reset the impstop counter since the solution archive is updated:
        m_n_impstop = 1;
        // I save the variables, objectives, eq. constraints, ineq. constraints and penalties in three different
        // vectors:
        for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
            variables[i] = pop.get_x()[sorted_list[i]];
            fitness[i] = pop.get_f()[sorted_list[i]];
            temporary_penalty[i] = sorted_vector[i];
            // Now the penalties of the archive:
            archive_penalties[i] = sol_archive[i][0];
        }
        // I merge the new and old penalties:
        temporary_penalty.insert(temporary_penalty.end(), archive_penalties.begin(), archive_penalties.end());
        std::vector<decltype(temporary_penalty.size())> sort_list_penalty(temporary_penalty.size());
        // I now reorder them depending on the penalty values (smallest first):
        std::iota(std::begin(sort_list_penalty), std::end(sort_list_penalty), decltype(temporary_penalty.size())(0));
        std::sort(
            sort_list_penalty.begin(), sort_list_penalty.end(),
            [&temporary_penalty](decltype(temporary_penalty.size()) idx1, decltype(temporary_penalty.size()) idx2) {
                return detail::less_than_f(temporary_penalty[idx1], temporary_penalty[idx2]);
            });
        // Only if one of the current population's penalty is better than the one of the archive, I replace the
        // first element of the temporary archive (which is a copy of the solution archive until now) with that
        vector_double::size_type count = 0;
        if (sort_list_penalty[0] < m_ker) {
            temporary_archive[0][0] = temporary_penalty[sort_list_penalty[0]];
            for (decltype(variables[0].size()) i_var = 0u; i_var < variables[0].size(); ++i_var) {
                temporary_archive[0][1 + i_var] = variables[sort_list_penalty[0]][i_var];
            }
            for (decltype(fitness[0].size()) i_obj = 0u; i_obj < fitness[0].size(); ++i_obj) {
                temporary_archive[0][1 + variables[0].size() + i_obj] = fitness[sort_list_penalty[0]][i_obj];
            }
        } else {
            ++count;
        }
        std::vector<vector_double::size_type> new_sort_list;
        new_sort_list.push_back(0);
        for (decltype(temporary_penalty.size()) j = 1u; j < temporary_penalty.size(); ++j) {
            if (std::abs(temporary_penalty[sort_list_penalty[j]] - temporary_penalty[sort_list_penalty[count]])
                < m_acc) {
            } else {
                ++count;
                new_sort_list.push_back(j);
            }
        }
        for (decltype(new_sort_list.size()) ii = 0u; ii < m_ker && ii < new_sort_list.size(); ++ii) {
            if (sort_list_penalty[new_sort_list[ii]] < m_ker) {
                temporary_archive[ii][0] = temporary_penalty[sort_list_penalty[new_sort_list[ii]]];
                for (decltype(variables[0].size()) i_var = 0u; i_var < variables[0].size(); ++i_var) {
                    temporary_archive[ii][1 + i_var] = variables[sort_list_penalty[new_sort_list[ii]]][i_var];
                }
                for (decltype(fitness[0].size()) i_obj = 0u; i_obj < fitness[0].size(); ++i_obj) {
                    temporary_archive[ii][1 + variables[0].size() + i_obj]
                        = fitness[sort_list_penalty[new_sort_list[ii]]][i_obj];
                }
            } else {
                temporary_archive[ii][0] = sol_archive[sort_list_penalty[new_sort_list[ii]] - m_ker][0];
                for (decltype(variables[0].size()) i_var = 0u; i_var < variables[0].size(); ++i_var) {
                    temporary_archive[ii][1 + i_var]
                        = sol_archive[sort_list_penalty[new_sort_list[ii]] - m_ker][1 + i_var];
                }
                for (decltype(fitness[0].size()) i_obj = 0u; i_obj < fitness[0].size(); ++i_obj) {
                    temporary_archive[ii][1 + variables[0].size() + i_obj]
                        = sol_archive[sort_list_penalty[new_sort_list[ii]] - m_ker][1 + variables[0].size() + i_obj];
                }
            }
        }

        // I hereby update the solution archive:
        for (decltype(m_ker) i = 0u; i < m_ker; ++i) {
            sol_archive[i] = temporary_archive[i];
        }
    } else {
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
void gaco::pheromone_computation(const unsigned gen, vector_double &prob_cumulative, vector_double &omega_vec,
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
                double cumulative = 0;
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
    for (decltype(n_tot) h = 1; h <= n_tot; ++h) {

        // I declare and define D_min and D_max:
        // at first I define D_min, D_max using the first two individuals stored in the sol_archive
        double d_min = std::abs(sol_archive[0][h] - sol_archive[1][h]);

        double d_max = std::abs(sol_archive[0][h] - sol_archive[1][h]);

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

        // In case a value for the focus parameter (different than zero) is passed, this limits the maximum
        // value of the standard deviation
        if (m_focus != 0. && ((d_max - d_min) / gen > (ub[h - 1] - lb[h - 1]) / m_focus) && m_memory == false) {
            sigma_vec[h - 1] = (ub[h - 1] - lb[h - 1]) / m_focus;

        } else if (m_focus != 0. && ((d_max - d_min) / m_counter > (ub[h - 1] - lb[h - 1]) / m_focus)
                   && m_memory == true) {
            sigma_vec[h - 1] = (ub[h - 1] - lb[h - 1]) / m_focus;

        } else {
            if (h <= n_con) {
                // continuous variables case:
                sigma_vec[h - 1] = (d_max - d_min) / m_gen_mark;

            } else {
                // integer variables case:
                sigma_vec[h - 1] = std::max(std::max((d_max - d_min) / m_gen_mark, 1.0 / m_gen_mark),
                                            (1.0 - 1.0 / (std::sqrt(n_tot - n_con))));
            }
        }
    }
}

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
void gaco::generate_new_ants(const population &pop, std::uniform_real_distribution<> dist,
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
            g_h = sol_archive[k_omega][1 + h] + sigma[h] * gauss_pdf(m_e);

            if (g_h < lb[h] || g_h > ub[h]) {

                while ((g_h < lb[h] || g_h > ub[h])) {
                    g_h = sol_archive[k_omega][1 + h] + sigma[h] * gauss_pdf(m_e);
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

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::gaco)
