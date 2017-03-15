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

#ifndef PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP
#define PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP

#include <boost/functional/hash.hpp>
#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../rng.hpp"
#include "../utils/generic.hpp"

namespace pagmo
{
namespace detail
{
/// Constrainted self adaptive udp
/**
 * Implements a udp that wraps a population and results in self adaptive constraints handling.
 *
 * The key idea of this constraint handling technique is to represent the constraint violation by one single
 * infeasibility measure, and to adapt dynamically the penalization of infeasible solutions. As the penalization process
 * depends on a given population, a method to update the penalties to a new population is provided.
 *
 * @see Farmani R., & Wright, J. A. (2003). Self-adaptive fitness formulation for constrained optimization.
 * Evolutionary Computation, IEEE Transactions on, 7(5), 445-455 for the paper introducing the method.
 *
 */
class unconstrain_with_adaptive_penalty
{
public:
    /// Constructs the udp. At construction all member get initialized using the incoming pop
    unconstrain_with_adaptive_penalty(population &pop) : m_fitness_map(), m_decision_vector_hash()
    {
        // Only constrained problems can use this
        if (pop.get_problem().get_nc() == 0u) {
            pagmo_throw(std::invalid_argument, "Cannot define an adaptive penalty for unconstrained problems.");
        }
        // Only single objective problems can use this
        if (pop.get_problem().get_nobj() != 1u) {
            pagmo_throw(std::invalid_argument, "Cannot define an adaptive penalty for multi objective problems.");
        }
        // Population cannot be empty
        if (pop.size() == 0u) {
            pagmo_throw(std::invalid_argument, "Cannot define an adaptive penalty for an empty population");
        }
        m_pop_ptr = &pop;
        update_ref_pop();
    }

    /// The fitness computation
    vector_double fitness(const vector_double &x) const
    {
        std::map<std::size_t, vector_double>::const_iterator it_f;
        double solution_infeasibility;
        vector_double f(1, 0.);

        // 1 - We check if the decision vector is already in the reference population and return that or recompute.
        it_f = m_fitness_map.find(m_decision_vector_hash(x));
        if (it_f != m_fitness_map.end()) {
            f = it_f->second;
            solution_infeasibility = compute_infeasibility(it_f->second);
        } else { // we have to compute the fitness (this will increase the feval counter in the ref pop problem )
            f = m_pop_ptr->get_problem().fitness(x);
            solution_infeasibility = compute_infeasibility(f);
        }

        // 2 - Then we apply the penalty
        if (solution_infeasibility > 0.) {
            // apply penalty 1 only if necessary
            if (m_apply_penalty_1) {
                double inf_tilde = 0.;
                inf_tilde = (solution_infeasibility - m_i_hat_down) / (m_i_hat_up - m_i_hat_down);
                f[0] += inf_tilde * (m_f_hat_down[0] - m_f_hat_up[0]);
            }
            // apply penalty 2
            f[0] += m_scaling_factor * std::abs(f[0])
                    * ((std::exp(2. * solution_infeasibility) - 1.) / (std::exp(2.) - 1.));
        }
        return f;
    }

    population *get_ptr()
    {
        return m_pop_ptr;
    }

    // Call to this method updates all the members that are used to penalize the objective function
    // As the penalization algorithm depends heavily on the ref population this method takes care of
    // updating the necessary information. It also builds the hash map used to avoid unecessary fitness
    // evaluations
    void update_ref_pop()
    {
        auto pop_size = m_pop_ptr->size();
        // 1 - We build the hash map to be able (later) to return already computed fitnesses corresponding to
        // some decision vector
        m_fitness_map.clear();
        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            m_fitness_map[m_decision_vector_hash(m_pop_ptr->get_x()[i])] = m_pop_ptr->get_f()[i];
        }

        std::vector<population::size_type> feasible_idx(0);
        std::vector<population::size_type> infeasible_idx(0);

        // 2 - Store indexes of feasible and non feasible individuals
        for (decltype(pop_size) i = 0u; i < pop_size; i++) {
            if (m_pop_ptr->get_problem().feasibility_f(m_pop_ptr->get_f()[i])) {
                feasible_idx.push_back(i);
            } else {
                infeasible_idx.push_back(i);
            }
        }
        // Init some data member values
        m_apply_penalty_1 = false;
        compute_c_max();

        // 3 - If the reference population contains only feasible fitnesses return
        if (infeasible_idx.size() == 0u) {
            return;
        }

        // 4 - Evaluate solutions infeasibility
        std::vector<double> infeasibility(pop_size, 0.);

        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            // compute the infeasibility of the fitness
            infeasibility[i] = compute_infeasibility(m_pop_ptr->get_f()[i]);
        }

        // search position of x_hat_down, x_hat_up and x_hat_round
        population::size_type hat_down_idx = -1;
        population::size_type hat_up_idx = -1;
        population::size_type hat_round_idx = -1;

        // 5 - First case: the population contains at least one feasible solution
        if (feasible_idx.size() > 0u) {
            // 5a - hat_down, a.k.a feasible individual with lowest objective value in p
            hat_down_idx = feasible_idx[0];
            for (decltype(feasible_idx.size()) i = 1u; i < feasible_idx.size(); ++i) {
                auto current_idx = feasible_idx[i];
                if (m_pop_ptr->get_f()[current_idx][0] < m_pop_ptr->get_f()[hat_down_idx][0]) {
                    hat_down_idx = current_idx;
                }
            }
            auto f_hat_down = m_pop_ptr->get_f()[hat_down_idx];

            // 5b - hat_up, its value depends if the population contains infeasible individual with objective
            // function better than f_hat_down
            bool pop_contains_infeasible_f_better_x_hat_down = false;
            for (decltype(infeasible_idx.size()) i = 0u; i < infeasible_idx.size(); ++i) {
                auto current_idx = infeasible_idx[i];
                if (m_pop_ptr->get_f()[current_idx][0] < f_hat_down[0]) {
                    pop_contains_infeasible_f_better_x_hat_down = true;
                    hat_up_idx = current_idx;
                    break;
                }
            }
            if (pop_contains_infeasible_f_better_x_hat_down) {
                // gets the individual with maximum infeasibility and objfun lower than f_hat_down
                for (decltype(infeasible_idx.size()) i = 0u; i < infeasible_idx.size(); ++i) {
                    auto current_idx = infeasible_idx[i];
                    if (m_pop_ptr->get_f()[current_idx][0] < f_hat_down[0]
                        && infeasibility[current_idx] >= infeasibility[hat_up_idx]) {
                        if (infeasibility[current_idx] == infeasibility[hat_up_idx]) {
                            if (m_pop_ptr->get_f()[current_idx][0] < m_pop_ptr->get_f()[hat_up_idx][0]) {
                                hat_up_idx = current_idx;
                            }
                        } else {
                            hat_up_idx = current_idx;
                        }
                    }
                }
                // Do apply penalty 1
                m_apply_penalty_1 = true;
            } else {
                // all the infeasible soutions have an objective function value greater than f_hat_down
                // the worst is the one that has the maximum infeasibility
                // initialize hat_up_idx
                hat_up_idx = infeasible_idx[0];

                for (decltype(infeasible_idx.size()) i = 0u; i < infeasible_idx.size(); ++i) {
                    auto current_idx = infeasible_idx[i];
                    if (infeasibility[current_idx] >= infeasibility[hat_up_idx]) {
                        if (infeasibility[current_idx] == infeasibility[hat_up_idx]) {
                            if (m_pop_ptr->get_f()[hat_up_idx][0] < m_pop_ptr->get_f()[current_idx][0]) {
                                hat_up_idx = current_idx;
                            }
                        } else {
                            hat_up_idx = current_idx;
                        }
                    }
                }
                // Do not apply penalty 1
                m_apply_penalty_1 = false;
            }

        } else {
            // 6 - Second case where there is no feasible solution in the population
            // best is the individual with the lowest infeasibility
            hat_down_idx = 0u;
            hat_up_idx = 0u;
            // 6a - hat_down
            for (decltype(pop_size) i = 0u; i < pop_size; i++) {
                if (infeasibility[i] <= infeasibility[hat_down_idx]) {
                    if (infeasibility[i] == infeasibility[hat_down_idx]) {
                        if (m_pop_ptr->get_f()[i][0] < m_pop_ptr->get_f()[hat_down_idx][0]) {
                            hat_down_idx = i;
                        }
                    } else {
                        hat_down_idx = i;
                    }
                }
            }
            // 6a - hat_up
            for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
                if (infeasibility[i] >= infeasibility[hat_up_idx]) {
                    if (infeasibility[i] == infeasibility[hat_up_idx]) {
                        if (m_pop_ptr->get_f()[hat_up_idx][0] < m_pop_ptr->get_f()[i][0]) {
                            hat_up_idx = i;
                        }
                    } else {
                        hat_up_idx = i;
                    }
                }
            }
            // Do apply penalty 1
            m_apply_penalty_1 = true;
        }

        // 7 - hat round idx,a.k.a the solution with highest objective
        // function value in the population
        hat_round_idx = 0u;
        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            if (m_pop_ptr->get_f()[hat_round_idx][0] < m_pop_ptr->get_f()[i][0]) {
                hat_round_idx = i;
            }
        }

        // Stores the objective function values of the three individuals
        m_f_hat_round = m_pop_ptr->get_f()[hat_round_idx];
        m_f_hat_down = m_pop_ptr->get_f()[hat_down_idx];
        m_f_hat_up = m_pop_ptr->get_f()[hat_up_idx];

        // Stores the solution infeasibility values of the three individuals
        m_i_hat_round = infeasibility[hat_round_idx];
        m_i_hat_down = infeasibility[hat_down_idx];
        m_i_hat_up = infeasibility[hat_up_idx];

        // Computes the scaling factor
        m_scaling_factor = 0.;
        if (m_f_hat_down[0] < m_f_hat_up[0]) {
            m_scaling_factor = (m_f_hat_round[0] - m_f_hat_up[0]) / m_f_hat_up[0];
        } else {
            m_scaling_factor = (m_f_hat_round[0] - m_f_hat_down[0]) / m_f_hat_down[0];
        }
        if (m_f_hat_up[0] == m_f_hat_round[0]) {
            m_scaling_factor = 0.;
        }
    }

private:
    // Computes c_max holding the maximum value of the violation of each constraint in the whole ref population
    void compute_c_max()
    {
        // Let's store some useful variables.
        auto pop_size = m_pop_ptr->size();
        auto nc = m_pop_ptr->get_problem().get_nc();
        auto nec = m_pop_ptr->get_problem().get_nec();
        auto c_tol = m_pop_ptr->get_problem().get_c_tol();

        // We init c_max
        m_c_max = vector_double(nc, 0.);

        // We evaluate the scaling factor
        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            // updates the current constraint vector
            auto fit = m_pop_ptr->get_f()[i];

            // computes scaling with the right definition of the constraints
            for (decltype(nec) j = 0u; j < nec; j++) {
                m_c_max[j] = std::max(m_c_max[j], std::max(0., (std::abs(fit[j + 1]) - c_tol[j])));
            }
            for (decltype(nc) j = nec; j < nc; j++) {
                m_c_max[j] = std::max(m_c_max[j], std::max(0., fit[j + 1] - c_tol[j]));
            }
        }
    }

    // Assuming the various data member contain useful information, this computes the
    // infeasibility measure of a certain fitness
    double compute_infeasibility(const vector_double &fit) const
    {
        // 1 - Let's store some useful variables.
        auto nc = m_pop_ptr->get_problem().get_nc();
        auto nec = m_pop_ptr->get_problem().get_nec();
        auto c_tol = m_pop_ptr->get_problem().get_c_tol();
        double retval = 0.;

        // 2 -  We compute the infeasibility measure
        for (decltype(nec) j = 0u; j < nec; ++j) {
            // test needed otherwise the m_c_max can be 0, and division by 0 occurs
            // f is accessed only in the constrain part hence the +1
            if (m_c_max[j] > 0.) {
                retval += std::max(0., (std::abs(fit[j + 1]) - c_tol[j])) / m_c_max[j];
            }
        }
        for (decltype(nc) j = nec; j < nc; ++j) {
            if (m_c_max[j] > 0.) {
                retval += std::max(0., fit[j + 1] - c_tol[j]) / m_c_max[j];
            }
        }
        retval /= nc;
        return retval;
    }
    // According to the population, the first penalty may or may not be applied
    bool m_apply_penalty_1;
    // The parameter gamma that scales the second penalty
    double m_scaling_factor;
    // The normalization of each constraint
    vector_double m_c_max;
    // The fitness of the three reference individuals
    vector_double m_f_hat_down;
    vector_double m_f_hat_up;
    vector_double m_f_hat_round;
    // The infeasibilities of the three reference individuals
    double m_i_hat_down;
    double m_i_hat_up;
    double m_i_hat_round;
    // A pointer to the reference population, allowing to call the fitness function and later recover
    // the counters outside of the class, and avoiding unecessary copies.
    population *m_pop_ptr;
    // The hash map connecting the decision vector hashes to their fitnesses
    std::map<std::size_t, vector_double> m_fitness_map;
    // The hasher (and impossible)
    boost::hash<std::vector<double>> m_decision_vector_hash;
};
}

} // namespace pagmo

// PAGMO_REGISTER_ALGORITHM(pagmo::sea)

#endif
