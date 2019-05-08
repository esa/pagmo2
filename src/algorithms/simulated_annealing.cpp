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
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

simulated_annealing::simulated_annealing(double Ts, double Tf, unsigned n_T_adj, unsigned n_range_adj,
                                         unsigned bin_size, double start_range, unsigned seed)
    : m_Ts(Ts), m_Tf(Tf), m_n_T_adj(n_T_adj), m_n_range_adj(n_range_adj), m_bin_size(bin_size),
      m_start_range(start_range), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
{
    if (Ts <= 0. || !std::isfinite(Ts)) {
        pagmo_throw(std::invalid_argument, "The starting temperature must be finite and positive, while a value of "
                                               + std::to_string(Ts) + " was detected.");
    }
    if (Tf <= 0. || !std::isfinite(Tf)) {
        pagmo_throw(std::invalid_argument, "The final temperature must be finite and positive, while a value of "
                                               + std::to_string(Tf) + " was detected.");
    }
    if (Tf > Ts) {
        pagmo_throw(std::invalid_argument,
                    "The final temperature must be smaller than the initial temperature, while a value of "
                        + std::to_string(Tf) + " >= " + std::to_string(Ts) + " was detected.");
    }
    if (start_range <= 0. || start_range > 1. || !std::isfinite(start_range)) {
        pagmo_throw(std::invalid_argument, "The start range must be in the (0, 1] range, while a value of "
                                               + std::to_string(start_range) + " was detected.");
    }
    if (n_T_adj == 0u) {
        pagmo_throw(std::invalid_argument,
                    "The number of temperature adjustments must be strictly positive, while a value of "
                        + std::to_string(n_T_adj) + " was detected.");
    }
    if (n_range_adj == 0u) {
        pagmo_throw(std::invalid_argument,
                    "The number of range adjustments must be strictly positive, while a value of "
                        + std::to_string(n_range_adj) + " was detected.");
    }
}

/// Algorithm evolve method
/**
 * @param pop population to be evolved
 * @return evolved population
 * @throws std::invalid_argument if the problem is multi-objective, constrained or stochastic
 * @throws std::invalid_argument if the population size is < 1u
 */
population simulated_annealing::evolve(population pop) const
{
    // We store some useful properties
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    auto dim = prob.get_nx();             // not const as used type for counters
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto fevals0 = prob.get_fevals(); // disount for the already made fevals
    unsigned count = 1u;              // regulates the screen output

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this particular algorithm.
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (prob.get_nf() != 1u) {
        pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    if (!pop.size()) {
        pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)

    // We init the starting point
    auto sel_xf = select_individual(pop);
    vector_double x0(std::move(sel_xf.first)), fit0(std::move(sel_xf.second));
    // Determines the coefficient to decrease the temperature
    const double Tcoeff = std::pow(m_Tf / m_Ts, 1.0 / static_cast<double>(m_n_T_adj));
    // Stores the current and new points
    auto xNEW = x0;
    auto xOLD = x0;
    auto best_x = x0;
    auto fNEW = fit0;
    auto fOLD = fit0;
    auto best_f = fit0;

    // Stores the adaptive ranges for each component
    vector_double step(dim, m_start_range);

    // Stores the number of accepted points for each component
    std::vector<int> acp(dim, 0u);
    double ratio = 0., currentT = m_Ts, probab = 0.;

    // Main SA loops
    for (decltype(m_n_T_adj) jter = 0u; jter < m_n_T_adj; ++jter) {
        for (decltype(m_n_range_adj) mter = 0u; mter < m_n_range_adj; ++mter) {
            // 1 - Annealing
            for (decltype(m_bin_size) kter = 0u; kter < m_bin_size; ++kter) {
                auto nter = std::uniform_int_distribution<vector_double::size_type>(0u, dim - 1u)(m_e);
                for (decltype(dim) numb = 0u; numb < dim; ++numb) {
                    nter = (nter + 1u) % dim;
                    // We modify the current point by mutating its nter component within the adaptive step
                    auto width = step[nter] * (ub[nter] - lb[nter]);
                    xNEW[nter] = uniform_real_from_range(std::max(xOLD[nter] - width, lb[nter]),
                                                         std::min(xOLD[nter] + width, ub[nter]), m_e);
                    // And we valuate the objective function for the new point
                    fNEW = prob.fitness(xNEW);
                    // We decide wether to accept or discard the point
                    if (fNEW[0] <= fOLD[0]) {
                        // accept
                        xOLD[nter] = xNEW[nter];
                        fOLD = fNEW;
                        acp[nter]++; // Increase the number of accepted values
                        // We update the best
                        if (fNEW[0] <= best_f[0]) {
                            best_f = fNEW;
                            best_x = xNEW;
                        }
                    } else {
                        // test it with Boltzmann to decide the acceptance
                        probab = std::exp(-std::abs(fOLD[0] - fNEW[0]) / currentT);
                        // we compare prob with a random probability.
                        if (probab > drng(m_e)) {
                            xOLD[nter] = xNEW[nter];
                            fOLD = fNEW;
                            acp[nter]++; // Increase the number of accepted values
                        } else {
                            xNEW[nter] = xOLD[nter];
                        }
                    }
                    // 2 - We log to screen
                    if (m_verbosity > 0u) {
                        // Prints a log line every m_verbosity fitness evaluations
                        auto fevals_count = prob.get_fevals() - fevals0;
                        if (fevals_count >= (count - 1u) * m_verbosity) {
                            // 1 - Every 50 lines print the column names
                            if (count % 50u == 1u) {
                                print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15),
                                      "Current:", std::setw(15), "Mean range:", std::setw(15), "Temperature:", '\n');
                            }
                            auto avg_range
                                = std::accumulate(step.begin(), step.end(), 0.) / static_cast<double>(step.size());
                            // 2 - Print
                            print(std::setw(7), fevals_count, std::setw(15), best_f[0], std::setw(15), fOLD[0],
                                  std::setw(15), avg_range, std::setw(15), currentT);
                            ++count;
                            std::cout << std::endl; // we flush here as we want the user to read in real time ...
                            // Logs
                            m_log.emplace_back(fevals_count, best_f[0], fOLD[0], avg_range, currentT);
                        }
                    }
                } // end for(nter = 0; ...
            }     // end for(kter = 0; ...
            // adjust the step (adaptively)
            for (decltype(dim) iter = 0u; iter < dim; ++iter) {
                ratio = static_cast<double>(acp[iter]) / static_cast<double>(m_bin_size);
                acp[iter] = 0u; // reset the counter
                if (ratio > .6) {
                    // too many acceptances, increase the step by a factor 3 maximum
                    step[iter] = step[iter] * (1. + 2. * (ratio - .6) / .4);
                } else {
                    if (ratio < .4) {
                        // too few acceptance, decrease the step by a factor 3 maximum
                        step[iter] = step[iter] / (1. + 2. * ((.4 - ratio) / .4));
                    };
                };
                // And if it becomes too large, reset it to its initial value
                if (step[iter] > m_start_range) step[iter] = m_start_range;
            }
        }
        // Cooling schedule
        currentT *= Tcoeff;
    }
    // We update the decision vector in pop, but only if things have improved
    if (best_f[0] <= fit0[0]) {
        replace_individual(pop, best_x, best_f);
    }
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void simulated_annealing::set_seed(unsigned seed)
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
std::string simulated_annealing::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tStarting temperature: ", m_Ts);
    stream(ss, "\n\tFinal temperature: ", m_Tf);
    stream(ss, "\n\tNumber of temperature adjustments: ", m_n_T_adj);
    stream(ss, "\n\tNumber of range adjustments: ", m_n_range_adj);
    stream(ss, "\n\tBin size: ", m_bin_size);
    stream(ss, "\n\tStarting range: ", m_start_range);
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
 * @throws unspecified any exception thrown by the serialization of the UDA and of primitive types.
 */
template <typename Archive>
void simulated_annealing::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, boost::serialization::base_object<not_population_based>(*this), m_Ts, m_Tf, m_n_T_adj,
                    m_n_range_adj, m_bin_size, m_start_range, m_e, m_seed, m_verbosity, m_log);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::simulated_annealing)
