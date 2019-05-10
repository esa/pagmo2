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
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/utils/constrained.hpp>

namespace pagmo
{

compass_search::compass_search(unsigned max_fevals, double start_range, double stop_range, double reduction_coeff)
    : m_max_fevals(max_fevals), m_start_range(start_range), m_stop_range(stop_range),
      m_reduction_coeff(reduction_coeff), m_verbosity(0u), m_log()
{
    if (start_range > 1. || start_range <= 0. || std::isnan(start_range)) {
        pagmo_throw(std::invalid_argument, "The start range must be in (0, 1], while a value of "
                                               + std::to_string(start_range) + " was detected.");
    }
    if (stop_range > 1. || stop_range >= start_range || std::isnan(stop_range)) {
        pagmo_throw(std::invalid_argument, "the stop range must be in (start_range, 1], while a value of "
                                               + std::to_string(stop_range) + " was detected.");
    }
    if (reduction_coeff >= 1. || reduction_coeff <= 0. || std::isnan(reduction_coeff)) {
        pagmo_throw(std::invalid_argument, "The reduction coefficient must be in (0,1), while a value of "
                                               + std::to_string(reduction_coeff) + " was detected.");
    }
}

/// Algorithm evolve method (juice implementation of the algorithm)
/**
 * Evolves the population up to when the search range becomes smaller than
 * the defined stop_range
 *
 * @param pop population to be evolved
 * @return evolved population
 * @throws std::invalid_argument if the problem is multi-objective or stochastic
 * @throws std::invalid_argument if the population is empty
 */
population compass_search::evolve(population pop) const
{
    // We store some useful variables
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto neq = prob.get_nec();

    auto fevals0 = prob.get_fevals(); // discount for the already made fevals
    unsigned count = 1u;              // regulates the screen output

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this
    // particular algorithm.
    if (prob.get_nobj() != 1u) {
        pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    if (pop.size() == 0u) {
        pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
    }
    // Get out if there is nothing to do.
    if (m_max_fevals == 0u) {
        return pop;
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    // We init the starting point
    auto sel_xf = select_individual(pop);
    vector_double cur_best_x(std::move(sel_xf.first)), cur_best_f(std::move(sel_xf.second));

    // We need some auxiliary variables
    bool flag = false;
    unsigned fevals = 0u;

    double newrange = m_start_range;

    while (newrange > m_stop_range && fevals <= m_max_fevals) {
        flag = false;
        for (decltype(dim) i = 0u; i < dim; i++) {
            auto x_trial = cur_best_x;
            // move up
            x_trial[i] = cur_best_x[i] + newrange * (ub[i] - lb[i]);
            // feasibility correction
            if (x_trial[i] > ub[i]) x_trial[i] = ub[i];
            // objective function evaluation
            auto f_trial = prob.fitness(x_trial);
            fevals++;
            if (compare_fc(f_trial, cur_best_f, prob.get_nec(), prob.get_c_tol())) {
                cur_best_f = f_trial;
                cur_best_x = x_trial;
                flag = true;
                break; // accept
            }

            // move down
            x_trial[i] = cur_best_x[i] - newrange * (ub[i] - lb[i]);
            // feasibility correction
            if (x_trial[i] < lb[i]) x_trial[i] = lb[i];
            // objective function evaluation
            f_trial = prob.fitness(x_trial);
            fevals++;
            if (compare_fc(f_trial, cur_best_f, prob.get_nec(), prob.get_c_tol())) {
                cur_best_f = f_trial;
                cur_best_x = x_trial;
                flag = true;
                break; // accept
            }
        }
        if (!flag) {
            newrange *= m_reduction_coeff;
        }

        // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Prints a log line if a new best is found or the range has been decreased

            // 1 - Every 50 lines print the column names
            if (count % 50u == 1u) {
                print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15), "Violated:", std::setw(15),
                      "Viol. Norm:", std::setw(15), "Range:", '\n');
            }
            // 2 - Print
            auto c1eq = detail::test_eq_constraints(cur_best_f.data() + 1, cur_best_f.data() + 1 + neq,
                                                    prob.get_c_tol().data());
            auto c1ineq = detail::test_ineq_constraints(
                cur_best_f.data() + 1 + neq, cur_best_f.data() + cur_best_f.size(), prob.get_c_tol().data() + neq);
            auto n = prob.get_nc() - c1eq.first - c1ineq.first;
            auto l = c1eq.second + c1ineq.second;
            print(std::setw(7), prob.get_fevals() - fevals0, std::setw(15), cur_best_f[0], std::setw(15), n,
                  std::setw(15), l, std::setw(15), newrange, '\n');
            ++count;
            // Logs
            m_log.emplace_back(prob.get_fevals() - fevals0, cur_best_f[0], n, l, newrange);
        }
    } // end while

    if (m_verbosity) {
        if (newrange <= m_stop_range) {
            std::cout << "Exit condition -- range: " << newrange << " <= " << m_stop_range << "\n";
        } else {
            std::cout << "Exit condition -- fevals: " << fevals << " > " << m_max_fevals << "\n";
        }
    }

    // Force the current best into the original population
    replace_individual(pop, cur_best_x, cur_best_f);
    return pop;
}

/// Extra info
/**
 * One of the optional methods of any user-defined algorithm (UDA).
 *
 * @return a string containing extra info on the algorithm
 */
std::string compass_search::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tMaximum number of objective function evaluations: ", m_max_fevals);
    stream(ss, "\n\tStart range: ", m_start_range);
    stream(ss, "\n\tStop range: ", m_stop_range);
    stream(ss, "\n\tReduction coefficient: ", m_reduction_coeff);
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
void compass_search::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, boost::serialization::base_object<not_population_based>(*this), m_max_fevals, m_start_range,
                    m_stop_range, m_reduction_coeff, m_verbosity, m_log);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::compass_search)
