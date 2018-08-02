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

#ifndef PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP
#define PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP

#include <cassert>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

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
struct penalized_udp {
public:
    /// Unused default constructor to please the is_udp type trait
    penalized_udp()
    {
        assert(false);
    };
    /// Constructs the udp. At construction all member get initialized calling update().
    penalized_udp(population &pop) : m_fitness_map()
    {
        assert(pop.get_problem().get_nc() != 0u);   // Only constrained problems can use this
        assert(pop.get_problem().get_nobj() == 1u); // Only single objective problems can use this
        assert(pop.size() >= 4u);                   // Population cannot contain less than 3 individuals

        // We assign the naked pointer The pointer will be immutable (as in its never changed afterwards)
        m_pop_ptr = &pop;
        // Update all data members and init the cache
        update();
    }

    // The bounds are unchanged
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return m_pop_ptr->get_problem().get_bounds();
    }

    /// The fitness computation
    vector_double fitness(const vector_double &x) const
    {
        double solution_infeasibility;
        vector_double f(1, 0.);

        // 1 - We check if the decision vector is already in the reference population and return that or recompute.
        auto it_f = m_fitness_map.find(x);
        if (it_f != m_fitness_map.end()) {
            f[0] = it_f->second[0];
            solution_infeasibility = compute_infeasibility(it_f->second);
        } else { // we have to compute the fitness (this will increase the feval counter in the ref pop problem )
            auto fit = m_pop_ptr->get_problem().fitness(x);
            f[0] = fit[0];
            solution_infeasibility = compute_infeasibility(fit);
            m_fitness_map[x] = fit;
        }
        // 2 - Then we apply the penalty
        if (solution_infeasibility > 0.) {
            double inf_tilde = solution_infeasibility;
            if (m_i_hat_up != m_i_hat_down) {
                inf_tilde = (solution_infeasibility - m_i_hat_down) / (m_i_hat_up - m_i_hat_down);
            } else {
                inf_tilde = solution_infeasibility; // This will trigger, for example, when the whole population is
                                                    // feasible.
            }
            // apply penalty 1 only if necessary. This penalizes infeasible solutions so that their objective
            // values cannot be much better than that of the best solution.
            if (m_apply_penalty_1) {
                f[0] += inf_tilde * (m_f_hat_down[0] - m_f_hat_up[0]);
            }
            // apply penalty 2 NOTE: uses (2. * inf_tilde) correcting what seems a clear bug in pagmo legacy
            // This penalizes all infeasible solutions exponentially with their infeasibility
            f[0] += m_scaling_factor * std::abs(f[0]) * ((std::exp(2. * inf_tilde) - 1.) / (std::exp(2.) - 1.));
        }
        return f;
    }

    // Call to this method updates all the members that are used to penalize the objective function
    // As the penalization algorithm depends heavily on the ref population this method takes care of
    // updating the necessary information. It also builds the hash map used to avoid unecessary fitness
    // evaluations. We exclude this method from the test as all of its corner cases are difficult to trigger
    // and test for correctness
    void update()
    {
        auto pop_size = m_pop_ptr->size();
        // 1 - We build the hash map to be able (later) to return already computed fitnesses corresponding to
        // some decision vector
        m_fitness_map.clear();
        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            m_fitness_map[m_pop_ptr->get_x()[i]] = m_pop_ptr->get_f()[i];
        }

        // Init some data member values
        m_apply_penalty_1 = false;
        compute_c_max();

        // 2 - Compute feasibility of all individuals and store the indexes of feasible / unfeasible ones
        std::vector<population::size_type> feasible_idx(0);
        std::vector<population::size_type> infeasible_idx(0);
        std::vector<double> infeasibility(pop_size, 0.);
        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            // compute the infeasibility of the fitness
            infeasibility[i] = compute_infeasibility(m_pop_ptr->get_f()[i]);
            if (infeasibility[i] > 0.) {
                infeasible_idx.push_back(i);
            } else {
                feasible_idx.push_back(i);
            }
        }
        m_n_feasible = feasible_idx.size();

        // 3 - If the reference population contains only feasible fitnesses
        if (m_n_feasible == pop_size) {
            // Since no infeasible individuals exist in the reference population, we
            // still need to decide what to do when evaluating the fitness of a decision vector
            // not in the ref_pop. We here set the members so that all penalties are zero.
            m_scaling_factor = 0.;
            m_i_hat_up = 0.;
            m_i_hat_down = 0.;
            // We init these as well even though they will not be used
            m_i_hat_round = 0.;
            m_f_hat_down = m_pop_ptr->get_f()[0];
            m_f_hat_up = m_pop_ptr->get_f()[0];
            m_f_hat_round = m_pop_ptr->get_f()[0];
            return;
        }

        // 4 - First case: the population contains at least one feasible solution
        population::size_type hat_down_idx = 0u, hat_up_idx = 0u, hat_round_idx = 0u;
        if (feasible_idx.size() > 0u) {
            // 4a - hat_down, a.k.a feasible individual with lowest objective value in the ref_pop
            hat_down_idx = feasible_idx[0];
            for (decltype(feasible_idx.size()) i = 1u; i < feasible_idx.size(); ++i) {
                auto current_idx = feasible_idx[i];
                if (m_pop_ptr->get_f()[current_idx][0] < m_pop_ptr->get_f()[hat_down_idx][0]) {
                    hat_down_idx = current_idx;
                }
            }
            auto f_hat_down = m_pop_ptr->get_f()[hat_down_idx];

            // 4b - hat_up, its value depends if the population contains infeasible individual with objective
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

                for (decltype(infeasible_idx.size()) i = 1u; i < infeasible_idx.size(); ++i) {
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
            // 5 - Second case: there is no feasible solution in the reference population
            // 5a - hat_down, a.k.a the individual with the lowest infeasibility (and minimum objective function)
            hat_down_idx = 0u;
            for (decltype(pop_size) i = 1u; i < pop_size; ++i) {
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
            // 5b - hat_up, ak.a. the individual with the maximum infeasibility (and maximum objective function)
            hat_up_idx = 0u;
            for (decltype(pop_size) i = 1u; i < pop_size; ++i) {
                if (infeasibility[i] >= infeasibility[hat_up_idx]) {
                    if (infeasibility[i] == infeasibility[hat_up_idx]) {
                        if (m_pop_ptr->get_f()[i][0] > m_pop_ptr->get_f()[hat_up_idx][0]) {
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

        // 6 - hat round idx, a.k.a the solution with highest objective
        // function value in the reference population
        hat_round_idx = 0u;
        for (decltype(pop_size) i = 1u; i < pop_size; ++i) {
            if (m_pop_ptr->get_f()[i][0] > m_pop_ptr->get_f()[hat_round_idx][0]) {
                hat_round_idx = i;
            }
        }

        // Stores the fitness values of the three special individuals
        m_f_hat_round = m_pop_ptr->get_f()[hat_round_idx];
        m_f_hat_down = m_pop_ptr->get_f()[hat_down_idx];
        m_f_hat_up = m_pop_ptr->get_f()[hat_up_idx];

        // Stores the solution infeasibility values of the three individuals
        m_i_hat_round = infeasibility[hat_round_idx];
        m_i_hat_down = infeasibility[hat_down_idx];
        m_i_hat_up = infeasibility[hat_up_idx];

        // Computes the scaling factor
        m_scaling_factor = 0.;
        if (m_f_hat_up[0] <= m_f_hat_down[0]) {
            m_scaling_factor = (m_f_hat_round[0] - m_f_hat_down[0]) / m_f_hat_down[0];
        } else {
            m_scaling_factor = (m_f_hat_round[0] - m_f_hat_up[0]) / m_f_hat_up[0];
        }
        if (m_f_hat_up[0] == m_f_hat_round[0]) {
            m_scaling_factor = 0.;
        }
    }

    // Only for debug purposes
    friend std::ostream &operator<<(std::ostream &os, const penalized_udp &p)
    {
        auto pop_size = p.m_pop_ptr->size();
        // Evaluate all solutions infeasibility
        std::vector<double> infeasibility(pop_size, 0.);

        for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
            // compute the infeasibility of the fitness
            infeasibility[i] = p.compute_infeasibility(p.m_pop_ptr->get_f()[i]);
        }
        os << "\nInfeasibilities: ";
        os << "\n\tBest (hat down): " << p.m_i_hat_down;
        os << "\n\tWorst (hat up): " << p.m_i_hat_up;
        os << "\n\tWorst objective (hat round): " << p.m_i_hat_round;
        stream(os, "\n\tAll: ", infeasibility);
        os << "\nFitness: ";
        stream(os, "\n\tBest (hat down): ", p.m_f_hat_down);
        stream(os, "\n\tWorst (hat up): ", p.m_f_hat_up);
        stream(os, "\n\tWorst objective (hat round): ", p.m_f_hat_round);
        os << "\nMisc: ";
        stream(os, "\n\tConstraints normalization: ", p.m_c_max);
        stream(os, "\n\tApply penalty 1: ", p.m_apply_penalty_1);
        stream(os, "\n\tGamma (scaling factor): ", p.m_scaling_factor);
        return os;
    }

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
            // fitness of the i-th decision vector
            auto fit = m_pop_ptr->get_f()[i];

            // computes scaling with the right definition of the constraints
            for (decltype(nec) j = 0u; j < nec; ++j) {
                m_c_max[j] = std::max(m_c_max[j], std::max(0., (std::abs(fit[j + 1]) - c_tol[j])));
            }
            for (decltype(nc) j = nec; j < nc; ++j) {
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
        // NOTE: if, for some i, m_c_max[i] is zero, that constraint is satisfied by the whole population
        // we thus neglect it favouring its violation in the assumption it is not a problem to fix it
        for (decltype(nec) j = 0u; j < nec; ++j) {
            if (m_c_max[j] > 0.) {
                retval += std::max(0., (std::abs(fit[j + 1]) - c_tol[j])) / m_c_max[j];
            }
        }
        for (decltype(nc) j = nec; j < nc; ++j) {
            if (m_c_max[j] > 0.) {
                retval += std::max(0., fit[j + 1] - c_tol[j]) / m_c_max[j];
            }
        }
        retval /= static_cast<double>(nc);
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

    vector_double::size_type m_n_feasible;
    // A NAKED pointer to the reference population, allowing to call the fitness function and later recover
    // the counters outside of the class, and avoiding unecessary copies. Use with care.
    population *m_pop_ptr;
    // The hash map connecting the decision vector to their fitnesses. The use of
    // custom comparison is needed to take care of nans, while the custom hasher is needed as std::hash does not
    // work on std::vectors
    mutable std::unordered_map<vector_double, vector_double, detail::hash_vf<double>, detail::equal_to_vf<double>>
        m_fitness_map;
};
} // namespace detail

/// Self-adaptive constraints handling
/**
 *
 * This meta-algorithm implements a constraint handling technique that allows the use of any user-defined algorithm
 * (UDA) able to deal with single-objective unconstrained problems, on single-objective constrained problems. The
 * technique self-adapts its parameters during
 * each successive call to the inner UDA basing its decisions on the entire underlying population. The resulting
 * approach is an alternative to using the meta-problem pagmo::unconstrain to transform the
 * constrained fitness into an unconstrained fitness.
 *
 * The self-adaptive constraints handling meta-algorithm is largely based on the ideas of Faramani and Wright but it
 * extends their use to any-algorithm, in particular to non generational population based evolutionary approaches where
 * a steady-state reinsertion is used (i.e., as soon as an individual is found fit it is immediately reinserted into the
 * pop and will influence the next offspring genetic material).
 *
 * Each decision vector is assigned an infeasibility measure \f$\iota\f$ which accounts for the normalized violation of
 * all the constraints (discounted by the constraints tolerance as returned by pagmo::problem::get_c_tol()). The
 * normalization factor used \f$c_{j_{max}}\f$ is the maximum violation of the \f$j-th\f$ constraint.
 *
 * As in the original paper, three individuals in the evolving population are then used to penalize the single
 * objective.
 *
 * \f[
 * \begin{array}{rl}
 *   \check X & \mbox{: the best decision vector} \\
 *   \hat X & \mbox{: the worst decision vector} \\
 *   \breve X & \mbox{: the decision vector with the highest objective}
 * \end{array}
 * \f]
 *
 * The best and worst decision vectors are defined accounting for their infeasibilities and for the value of the
 * objective function. Using the above definitions the overall pseudo code can be summarized as follows:
 *
 * @code{.unparsed}
 * > Select a pagmo::population (related to a single-objective constrained problem)
 * > Select a UDA (able to solve single-objective unconstrained problems)
 * > while i < iter
 * > > Compute the normalization factors (will depend on the current population)
 * > > Compute the best, worst, highest (will depend on the current population)
 * > > Evolve the population using the UDA and a penalized objective
 * > > Reinsert the best decision vector from the previous evolution
 * @endcode
 *
 * pagmo::cstrs_self_adaptive is a user-defined algorithm (UDA) that can be used to construct pagmo::algorithm objects.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    Self-adaptive constraints handling implements an internal cache to avoid the re-evaluation of the fitness
 *    for decision vectors already evaluated. This makes the final counter of function evaluations somewhat
 *    unpredictable. The number of function evaluation will be bounded to ``iters`` times the fevals made by one call to
 *    the inner UDA. The internal cache is reset at each iteration, but its size will grow unlimited during each call to
 *    the inner UDA evolve method.
 *
 * .. note::
 *
 *    Several modification were made to the original Faramani and Wright ideas to allow their approach to work on
 *    corner cases and with any UDAs. Most notably, a violation to the :math:`j`-th  constraint is ignored if all
 *    the decision vectors in the population satisfy that particular constraint (i.e. if :math:`c_{j_{max}} = 0`).
 *
 * .. note::
 *
 *    The performances of :cpp:class:`pagmo::cstrs_self_adaptive` are highly dependent on the particular inner UDA
 *    employed and in particular to its parameters (generations / iterations).
 *
 * .. seealso::
 *
 *    Farmani, Raziyeh, and Jonathan A. Wright. "Self-adaptive fitness formulation for constrained optimization." IEEE
 *    Transactions on Evolutionary Computation 7.5 (2003): 445-455.
 *
 * \endverbatim
 */
class cstrs_self_adaptive
{
    // Enabler for the ctor from UDA.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<algorithm, T &&>::value, int>;

public:
    /// Single entry of the log (iter, fevals, best_f, infeas, n. constraints violated, violation norm).
    typedef std::tuple<unsigned, unsigned long long, double, double, vector_double::size_type, double,
                       vector_double::size_type>
        log_line_type;
    /// The log.
    typedef std::vector<log_line_type> log_type;
    /// Default constructor.
    /**
     *
     * @param iters Number of iterations (calls to the inner UDA). After each iteration the penalty is adapted
     * The default constructor will initialize the algorithm with the following parameters:
     * - inner algorithm: pagmo::de{1u};
     * - seed: random.
     *
     */
    cstrs_self_adaptive(unsigned iters = 1u) : m_algorithm(de{1}), m_iters(iters), m_verbosity(0u), m_log()
    {
        const auto rnd = pagmo::random_device::next();
        m_seed = rnd;
        m_e.seed(rnd);
    }
    /// Constructor.
    /**
     *
     * @param iters Number of iterations (calls to the inner algorithm). After each iteration the penalty is adapted
     * @param a a pagmo::algorithm (or UDA) that will be used to construct the inner algorithm.
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws unspecified any exception thrown by the constructor of pagmo::algorithm.
     */
    template <typename T, ctor_enabler<T> = 0>
    explicit cstrs_self_adaptive(unsigned iters, T &&a, unsigned seed = pagmo::random_device::next())
        : m_algorithm(std::forward<T>(a)), m_iters(iters), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
    }

    /// Evolve method.
    /**
     * This method will call evolve on the inner algorithm \p iters times updating the penalty to be applied to the
     * objective after each call
     *
     * @param pop population to be evolved.
     *
     * @return evolved population.
     *
     * @throws std::invalid_argument if the problem is multi-objective or stochastic, or unconstrained and if the
     * population does not contain at least 3 individuals.
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto nec = prob.get_nec();            // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        auto NP = pop.size();

        auto fevals0 = prob.get_fevals(); // discount for the already made fevals
        unsigned count = 1u;              // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        if (prob.get_nobj() != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument, "The input problem " + prob.get_name() + " appears to be stochastic, "
                                                   + get_name() + " cannot deal with it");
        }
        if (prob.get_nc() == 0u) {
            pagmo_throw(std::invalid_argument, "No constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " needs a constrained problem");
        }
        if (NP < 4u) {
            pagmo_throw(std::invalid_argument,
                        "Cannot use " + prob.get_name() + " on a population with less than 4 individuals");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();
        // cstrs_self_adaptive main loop

        // 1 - We create a penalized meta-problem that mantains a pointer to pop and uses it to define and adapt the
        // penalty. Upon consruction a cache is also initialized mapping decision vectors to constrained fitnesses.
        detail::penalized_udp udp_p{pop};
        // 2 - We construct a new population with the penalized udp so that we can evolve it with single objective,
        // unconstrained solvers. Upon construction the problem is copied and so is the cache.
        population new_pop{udp_p};
        // The following lines do not cause fevals increments as the cache is hit.
        for (decltype(NP) i = 0u; i < NP; ++i) {
            new_pop.push_back(pop.get_x()[i]);
        }
        // Main iterations
        auto penalized_udp_ptr = new_pop.get_problem().extract<detail::penalized_udp>();
        for (decltype(m_iters) iter = 1u; iter <= m_iters; ++iter) {
            // We record the current best decision vector and fitness as we will
            // reinsert it at each iteration
            auto best_idx = pop.best_idx();
            auto best_x = pop.get_x()[best_idx];
            auto best_f = pop.get_f()[best_idx];
            auto worst_idx = pop.worst_idx();
            // As the population changes (evolves) we update all penalties and reset the cache
            // (the first iter this is not needed as upon construction this was already done and the pop
            // has not changed since)
            penalized_udp_ptr->update();
            for (decltype(new_pop.size()) i = 0u; i < new_pop.size(); ++i) {
                new_pop.set_x(i, pop.get_x()[i]);
            }
            // We log to screen
            if (m_verbosity > 0u) {
                if (iter % m_verbosity == 1u || m_verbosity == 1u) {
                    // Prints a log line after each call to the inner algorithm
                    // 1 - Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Iter:", std::setw(15), "Fevals:", std::setw(15),
                              "Best:", std::setw(15), "Infeasibility:", std::setw(15), "Violated:", std::setw(15),
                              "Viol. Norm:", std::setw(15), "N. Feasible:", '\n');
                    }
                    // 2 - Print
                    auto cur_best_f = pop.get_f()[pop.best_idx()];
                    auto c1eq = detail::test_eq_constraints(cur_best_f.data() + 1, cur_best_f.data() + 1 + nec,
                                                            prob.get_c_tol().data());
                    auto c1ineq = detail::test_ineq_constraints(cur_best_f.data() + 1 + nec,
                                                                cur_best_f.data() + cur_best_f.size(),
                                                                prob.get_c_tol().data() + nec);
                    auto n = prob.get_nc() - c1eq.first - c1ineq.first;
                    auto l = c1eq.second + c1ineq.second;
                    auto infeas = penalized_udp_ptr->compute_infeasibility(penalized_udp_ptr->m_f_hat_down);
                    auto n_feasible = penalized_udp_ptr->m_n_feasible;
                    print(std::setw(7), iter, std::setw(15), prob.get_fevals() - fevals0, std::setw(15), cur_best_f[0],
                          std::setw(15), infeas, std::setw(15), n, std::setw(15), l, std::setw(15), n_feasible);
                    if (!prob.feasibility_f(cur_best_f)) {
                        std::cout << " i";
                    }
                    ++count;
                    std::cout << std::endl; // we flush here as we want the user to read in real time ...
                    // Logs
                    m_log.emplace_back(iter, prob.get_fevals() - fevals0, cur_best_f[0], infeas, n, l, n_feasible);
                }
            }
            // We call the evolution on the unconstrained population (here is where fevals will increase)
            new_pop = m_algorithm.evolve(new_pop);
            penalized_udp_ptr = new_pop.get_problem().extract<detail::penalized_udp>();
            // We update the original pop avoiding fevals thanks to the cache
            for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
                auto x = new_pop.get_x()[i];
                auto it_f = penalized_udp_ptr->m_fitness_map.find(x);
                assert(it_f
                       != penalized_udp_ptr->m_fitness_map.end()); // We are assasserting here the cache will be hit
                pop.set_xf(i, x, it_f->second);
            }
            pop.set_xf(worst_idx, best_x, best_f);
        }
        return pop;
    }
    /// Set the seed.
    /**
     * @param seed the seed controlling the algorithm's stochastic behaviour.
     */
    void set_seed(unsigned seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    }
    /// Get the seed.
    /**
     * @return the seed controlling the algorithm's stochastic behaviour.
     */
    unsigned get_seed() const
    {
        return m_seed;
    }
    /// Set the algorithm verbosity.
    /**
     * This method will sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity,
     * - >0: will print and log one line each  \p level call to the inner algorithm.
     *
     * Example (verbosity 10):
     * @code{.unparsed}
     *   Iter:        Fevals:          Best: Infeasibility:      Violated:    Viol. Norm:   N. Feasible:
     *       1              0       -69.2141       0.235562              6        117.743              0 i
     *      11            200       -69.2141       0.248216              6        117.743              0 i
     *      21            400       -29.4754      0.0711599              5          44.39              0 i
     *      31            600       -30.0791      0.0878253              4        44.3803              0 i
     *     ...            ...       ........      .........              .        .......              . .
     *     211           4190       -7.68336    0.000341894              1       0.273829              0 i
     *     221           4390       -7.89941     0.00031154              1       0.273829              0 i
     *     231           4590       -8.38299    0.000168309              1       0.147935              0 i
     *     241           4790       -8.38299    0.000181461              1       0.147935              0 i
     *     251           4989       -8.71021    0.000191197              1       0.100357              0 i
     *     261           5189       -8.71021    0.000165734              1       0.100357              0 i
     *     271           5389       -10.7421              0              0              0              3
     *     281           5585       -10.7421              0              0              0              3
     *     291           5784       -11.4868              0              0              0              4
     *
     * @endcode
     * \p Iter is the iteration number, \p Fevals is the number of fitness evaluations, \p Best is the objective
     * function of the best fitness currently in the population, \p Infeasibility is the normailized infeasibility
     * measure, \p Violated is the number of constraints currently violated by the best solution, <tt>Viol. Norm</tt> is
     * the norm of the violation (discounted already by the constraints tolerance) and N. Feasible is the number of
     * feasible individuals in the current iteration. The small \p i appearing at the end of the line stands for
     * "infeasible" and will disappear only once \p Violated is 0.
     *
     * @param level verbosity level.
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    };
    /// Get the verbosity level.
    /**
     * @return the verbosity level.
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }

    /// Get log.
    /**
     * A log containing relevant quantities monitoring the last call to cstrs_self_adaptive::evolve(). Each element of
     * the returned <tt>std::vector</tt> is a cstrs_self_adaptive::log_line_type containing: \p Iter, \p Fevals, \p
     * Best, \p Infeasibility, \p Violated, <tt>Viol. Norm</tt>, <tt>N. Feasible</tt> as described in
     * cstrs_self_adaptive::set_verbosity().
     *
     * @return an <tt>std::vector</tt> of cstrs_self_adaptive::log_line_type containing the logged values Iters,
     * Fevals, Best, Infeasibility, Violated and Viol. Norm and N. Feasible.
     */
    const log_type &get_log() const
    {
        return m_log;
    }
    /// Algorithm name
    /**
     * @return a string containing the algorithm name.
     */
    std::string get_name() const
    {
        return "sa-CNSTR: Self-adaptive constraints handling";
    }
    /// Extra informations
    /**
     * @return a string containing extra informations on the algorithm.
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\n\tIterations: ", m_iters);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        stream(ss, "\n\n\tInner algorithm: ", m_algorithm.get_name());
        stream(ss, "\n\tInner algorithm extra info: ");
        stream(ss, "\n", m_algorithm.get_extra_info());
        return ss.str();
    }
    /// Algorithm's thread safety level.
    /**
     * The thread safety of a meta-algorithm is defined by the thread safety of the interal pagmo::algorithm.
     *
     * @return the thread safety level of the interal pagmo::algorithm.
     */
    thread_safety get_thread_safety() const
    {
        return m_algorithm.get_thread_safety();
    }
    /// Getter for the inner algorithm.
    /**
     * Returns a const reference to the inner pagmo::algorithm.
     *
     * @return a const reference to the inner pagmo::algorithm.
     */
    const algorithm &get_inner_algorithm() const
    {
        return m_algorithm;
    }
    /// Getter for the inner problem.
    /**
     * Returns a reference to the inner pagmo::algorithm.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The ability to extract a non const reference is provided only in order to allow to call
     *    non-const methods on the internal :cpp:class:`pagmo::algorithm` instance. Assigning a new
     *    :cpp:class:`pagmo::algorithm` via this reference is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a reference to the inner pagmo::algorithm.
     */
    algorithm &get_inner_algorithm()
    {
        return m_algorithm;
    }
    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDA and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_algorithm, m_iters, m_e, m_seed, m_verbosity, m_log);
    }

private:
    // Inner algorithm
    algorithm m_algorithm;
    unsigned m_iters;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::cstrs_self_adaptive)

#endif
