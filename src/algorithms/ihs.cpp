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

#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo
{

ihs::ihs(unsigned gen, double phmcr, double ppar_min, double ppar_max, double bw_min, double bw_max, unsigned seed)
    : m_gen(gen), m_phmcr(phmcr), m_ppar_min(ppar_min), m_ppar_max(ppar_max), m_bw_min(bw_min), m_bw_max(bw_max),
      m_e(seed), m_seed(seed), m_verbosity(0u)
{
    if (phmcr > 1 || phmcr < 0 || ppar_min > 1 || ppar_min < 0 || ppar_max > 1 || ppar_max < 0) {
        pagmo_throw(std::invalid_argument, "The probability of choosing from memory (phmcr) and the pitch "
                                           "adjustment rates (ppar_min, ppar_max) must all be in the [0,1] range");
    }
    if (ppar_min > ppar_max) {
        pagmo_throw(std::invalid_argument,
                    "The minimum pitch adjustment rate must not be greater than maximum pitch adjustment rate");
    }
    if (bw_min <= 0 || bw_max < bw_min) {
        pagmo_throw(std::invalid_argument, "The bandwidth values must be positive, and minimum bandwidth must not "
                                           "be greater than maximum bandwidth");
    }
}

/// Algorithm evolve method
/**
 * @param pop population to be evolved
 * @return evolved population
 * @throws std::invalid_argument if the problem is multi-objective or constrained
 */
population ihs::evolve(population pop) const
{
    // We store some useful properties
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    auto dim = prob.get_nx();
    auto int_dim = prob.get_nix();
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto fevals0 = prob.get_fevals(); // discount for the already made fevals
    unsigned count = 1u;              // regulates the screen output

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // Get out if there is nothing to do.
    if (m_gen == 0u) {
        return pop;
    }
    if (!pop.size()) {
        pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
    }
    if (prob.get_nc() != 0u && prob.get_nobj() > 1u) {
        pagmo_throw(std::invalid_argument, "Multiple objectives and non linear constraints detected in the "
                                               + prob.get_name() + " instance. " + get_name()
                                               + " cannot deal with this type of problem.");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    vector_double lu_diff(dim);
    for (decltype(dim) i = 0u; i < dim; ++i) {
        lu_diff[i] = ub[i] - lb[i];
    }
    // Distributions used
    std::uniform_int_distribution<decltype(pop.size())> uni_int(0, pop.size() - 1u); // to pick an individual
    std::uniform_real_distribution<double> drng(0., 1.);                             // to generate a number in [0, 1)

    // Used for parameter control
    const double c = std::log(m_bw_min / m_bw_max) / m_gen;

    // Declarations
    vector_double new_x(dim, 0.);
    std::vector<vector_double::size_type> best_idxs(pop.size());

    // Main loop
    for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
        // 1 - We adjust the algorithm parameters (parameter control)
        const double ppar_cur = m_ppar_min + ((m_ppar_max - m_ppar_min) * gen) / m_gen;
        const double bw_cur = m_bw_max * std::exp(c * gen);

        // 2 - We create a new decision vector
        // Continuous part.
        for (decltype(dim) i = 0u; i < dim - int_dim; ++i) {
            if (drng(m_e) < m_phmcr) {
                // new_x's i-th chromosome element is the one from a randomly chosen individual.
                new_x[i] = pop.get_x()[uni_int(m_e)][i];
                // Do pitch adjustment with ppar_cur probability.
                if (drng(m_e) < ppar_cur) {
                    // Randomly, add or subtract pitch from the current chromosome element.
                    new_x[i] += 2. * (0.5 - drng(m_e)) * bw_cur * lu_diff[i];
                }
            } else {
                // Pick randomly within the bounds.
                new_x[i] = uniform_real_from_range(lb[i], ub[i], m_e);
            }
        }

        // Integer Part
        for (decltype(dim) i = dim - int_dim; i < dim; ++i) {
            if (drng(m_e) < m_phmcr) {
                // new_x's i-th chromosome element is the one from a randomly chosen individual.
                new_x[i] = pop.get_x()[uni_int(m_e)][i];
                // Do pitch adjustment with ppar_cur probability.
                if (drng(m_e) < ppar_cur) {
                    // This generates minimum 1 and, only if bw_cur == 1 the pitch will be bigger
                    // than the width. Which is anyway dealt with later by the bound reflection
                    const double pitch = std::trunc(1 + bw_cur * lu_diff[i]);
                    // Randomly, add or subtract pitch from the current chromosome element.
                    new_x[i] += uniform_integral_from_range(new_x[i] - pitch, new_x[i] + pitch, m_e);
                }
            } else {
                // Draw a random integer in [lb, ub].
                new_x[i] = uniform_integral_from_range(lb[i], ub[i], m_e);
            }
        }

        // 4 - We fix the new decision vector within the bounds and evaluate
        detail::force_bounds_reflection(new_x, lb, ub);
        auto new_f = prob.fitness(new_x);

        // 5 - We insert the new decision vector in the population
        if (prob.get_nobj() == 1u) {      // Single objective case
            auto w_idx = pop.worst_idx(); // this is always defined by pagmo for single-objective cases
            if (prob.get_nc() == 0u) {    // unconstrained, we simply check fnew < fworst
                if (pop.get_f()[w_idx][0] >= new_f[0]) {
                    pop.set_xf(w_idx, new_x, new_f);
                }
            } else { // constrained, we use compare_fc
                if (compare_fc(new_f, pop.get_f()[w_idx], prob.get_nec(), prob.get_c_tol())) {
                    pop.set_xf(w_idx, new_x, new_f);
                }
            }
        } else { // multiobjective case
            auto fitnesses = pop.get_f();
            // we augment the list with the new fitness
            fitnesses.push_back(new_f);
            // select the best pop.size() individuals
            best_idxs = select_best_N_mo(fitnesses, pop.size());
            // define the new population
            for (population::size_type i = 0u; i < pop.size(); ++i) {
                if (best_idxs[i] == pop.size()) { // this is the new guy
                    pop.set_xf(i, new_x, new_f);
                } else { // these were already in the pop somewhere
                    pop.set_xf(i, pop.get_x()[best_idxs[i]], pop.get_f()[best_idxs[i]]);
                }
            }
        }

        // 6 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                log_a_line(pop, count, fevals0, ppar_cur, bw_cur);
            }
        }
    }
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void ihs::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Extra info
/**
 * One of the optional methods of any user-defined algorithm (UDA).
 *
 * @return a string containing extra info on the algorithm
 */
std::string ihs::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tProbability of choosing from memory: ", m_phmcr);
    stream(ss, "\n\tMinimum pitch adjustment rate: ", m_ppar_min);
    stream(ss, "\n\tMaximum pitch adjustment rate: ", m_ppar_max);
    stream(ss, "\n\tMinimum distance bandwidth: ", m_bw_min);
    stream(ss, "\n\tMaximum distance bandwidth: ", m_bw_max);
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
void ihs::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_phmcr, m_ppar_min, m_ppar_max, m_bw_min, m_bw_max, m_e, m_seed, m_verbosity, m_log);
}

// logging is complex fir ihs as the algorithm is an "any-problem" wannabe
void ihs::log_a_line(const population &pop, unsigned &count, unsigned long long fevals0, double ppar_cur,
                     double bw_cur) const
{
    const auto &prob = pop.get_problem();
    auto dim = prob.get_nx();
    auto nec = prob.get_nec();
    decltype(dim) best_idx;
    decltype(dim) worst_idx;

    auto dx = 0.;    // pop flatness
    auto df = 0.;    // fitness flatness
    decltype(dim) n; // number of violated constraints
    double l;        // violation norm
    vector_double ideal_point;

    if (prob.get_nobj() == 1u) {
        best_idx = pop.best_idx();
        worst_idx = pop.worst_idx();
        // The population flattness in chromosome
        for (decltype(dim) i = 0u; i < dim; ++i) {
            dx += std::abs(pop.get_x()[worst_idx][i] - pop.get_x()[best_idx][i]);
        }
        // The population flattness in fitness
        df = std::abs(pop.get_f()[worst_idx][0] - pop.get_f()[best_idx][0]);
        // Constraints stuff
        auto cur_best_f = pop.get_f()[pop.best_idx()];
        auto c1eq
            = detail::test_eq_constraints(cur_best_f.data() + 1, cur_best_f.data() + 1 + nec, prob.get_c_tol().data());
        auto c1ineq = detail::test_ineq_constraints(cur_best_f.data() + 1 + nec, cur_best_f.data() + cur_best_f.size(),
                                                    prob.get_c_tol().data() + nec);
        n = prob.get_nc() - c1eq.first - c1ineq.first;
        l = c1eq.second + c1ineq.second;
        // The best
        ideal_point.push_back(pop.champion_f()[0]);
    } else { // In a multiple objective problem df and dx are not defined and constraints are not present
        dx = -1.;
        df = -1.;
        n = 0;
        l = 0;
        ideal_point = ideal(pop.get_f());
    }

    // Every 50 lines print the column names (fevals, ppar, bw, dx, df, n. constraints violated, violation norm,
    // ideal [or best])
    if (count % 50u == 1u) {
        print("\n", std::setw(7), "Fevals:", std::setw(15), "ppar:", std::setw(15), "bw:", std::setw(15),
              "dx:", std::setw(15), "df:", std::setw(15), "Violated:", std::setw(15), "Viol. Norm:");
        for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
            if (i >= 5u) {
                print(std::setw(15), "... :");
                break;
            }
            print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
        }
        print('\n');
    }

    print(std::setw(7), prob.get_fevals() - fevals0, std::setw(15), ppar_cur, std::setw(15), bw_cur, std::setw(15), dx,
          std::setw(15), df, std::setw(15), n, std::setw(15), l);
    for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
        if (i >= 5u) {
            break;
        }
        print(std::setw(15), ideal_point[i]);
    }
    print('\n');

    ++count;
    // Logs
    m_log.emplace_back(prob.get_fevals() - fevals0, ppar_cur, bw_cur, dx, df, n, l, ideal_point);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::ihs)
