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
#include <sstream>
#include <stdexcept>
#include <string>

#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/generic.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

/// Default constructor.
/**
 * The default constructor will initialize the algorithm with the following parameters:
 * - inner algorithm: pagmo::compass_search;
 * - consecutive runs of the inner algorithm that need to result in no improvement for pagmo::mbh to stop: 5;
 * - scalar perturbation: 1E-2;
 * - seed: random.
 *
 * @throws unspecified any exception thrown by the constructor of pagmo::algorithm.
 */
mbh::mbh() : m_algorithm(compass_search{}), m_stop(5u), m_perturb(1, 1e-2), m_verbosity(0u)
{
    const auto rnd = pagmo::random_device::next();
    m_seed = rnd;
    m_e.seed(rnd);
}

void mbh::scalar_ctor_impl(double perturb)
{
    if (std::isnan(perturb) || perturb > 1. || perturb <= 0.) {
        pagmo_throw(std::invalid_argument, "The scalar perturbation must be in (0, 1], while a value of "
                                               + std::to_string(perturb) + " was detected.");
    }
}

void mbh::vector_ctor_impl(const vector_double &perturb)
{
    if (!std::all_of(perturb.begin(), perturb.end(),
                     [](double item) { return (!std::isnan(item) && item > 0. && item <= 1.); })) {
        pagmo_throw(std::invalid_argument,
                    "The perturbation must have all components in (0, 1], while that is not the case.");
    }
}

/// Evolve method.
/**
 * This method will evolve the input population up to when \p stop consecutve runs of the internal
 * algorithm do not improve the solution.
 *
 * @param pop population to be evolved.
 *
 * @return evolved population.
 *
 * @throws std::invalid_argument if the problem is multi-objective or stochastic, or if the perturbation vector size
 * does not equal the problem size.
 */
population mbh::evolve(population pop) const
{
    // We store some useful variables
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
    auto nec = prob.get_nec();            // This getter does not return a const reference but a copy
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
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
    // Get out if there is nothing to do.
    if (m_stop == 0u) {
        return pop;
    }
    // Check if the perturbation vector has size 1, in which case the whole perturbation vector is filled with
    // the same value equal to its first entry
    if (m_perturb.size() == 1u) {
        for (decltype(dim) i = 1u; i < dim; ++i) {
            m_perturb.push_back(m_perturb[0]);
        }
    }
    // Check that the perturbation vector size equals the size of the problem
    if (m_perturb.size() != dim) {
        pagmo_throw(std::invalid_argument, "The perturbation vector size is: " + std::to_string(m_perturb.size())
                                               + ", while the problem dimension is: " + std::to_string(dim)
                                               + ". They need to be equal for MBH to work.");
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();
    // mbh main loop
    unsigned i = 0u;
    while (i < m_stop) {
        // 1 - We make a copy of the current population
        population pop_old(pop);
        // 2 - We perturb the current population (NP funevals are made here)
        for (decltype(NP) j = 0u; j < NP; ++j) {
            vector_double tmp_x(dim);
            for (decltype(dim) k = 0u; k < dim; ++k) {
                tmp_x[k]
                    = uniform_real_from_range(std::max(pop.get_x()[j][k] - m_perturb[k] * (ub[k] - lb[k]), lb[k]),
                                              std::min(pop.get_x()[j][k] + m_perturb[k] * (ub[k] - lb[k]), ub[k]), m_e);
            }
            pop.set_x(j, tmp_x); // fitness is evaluated here
        }
        // 3 - We evolve the current population with the selected algorithm
        pop = m_algorithm.evolve(pop);
        i++;
        // 4 - We reset the counter if we have improved, otherwise we reset the population
        if (compare_fc(pop.get_f()[pop.best_idx()], pop_old.get_f()[pop_old.best_idx()], nec, prob.get_c_tol())) {
            i = 0u;
        } else {
            for (decltype(NP) j = 0u; j < NP; ++j) {
                pop.set_xf(j, pop_old.get_x()[j], pop_old.get_f()[j]);
            }
        }
        // 5 - We log to screen
        if (m_verbosity > 0u) {
            // Prints a log line after each call to the inner algorithm
            // 1 - Every 50 lines print the column names
            if (count % 50u == 1u) {
                print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15), "Violated:", std::setw(15),
                      "Viol. Norm:", std::setw(15), "Trial:", '\n');
            }
            // 2 - Print
            auto cur_best_f = pop.get_f()[pop.best_idx()];
            auto c1eq = detail::test_eq_constraints(cur_best_f.data() + 1, cur_best_f.data() + 1 + nec,
                                                    prob.get_c_tol().data());
            auto c1ineq = detail::test_ineq_constraints(
                cur_best_f.data() + 1 + nec, cur_best_f.data() + cur_best_f.size(), prob.get_c_tol().data() + nec);
            auto n = prob.get_nc() - c1eq.first - c1ineq.first;
            auto l = c1eq.second + c1ineq.second;
            print(std::setw(7), prob.get_fevals() - fevals0, std::setw(15), cur_best_f[0], std::setw(15), n,
                  std::setw(15), l, std::setw(15), i);
            if (!prob.feasibility_f(pop.get_f()[pop.best_idx()])) {
                std::cout << " i";
            }
            ++count;
            std::cout << std::endl; // we flush here as we want the user to read in real time ...
            // Logs
            m_log.emplace_back(prob.get_fevals() - fevals0, cur_best_f[0], n, l, i);
        }
    }
    // We extract chromosomes and fitnesses
    return pop;
}

/// Set the seed.
/**
 * @param seed the seed controlling the algorithm's stochastic behaviour.
 */
void mbh::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Set the perturbation vector.
/**
 * @param perturb the perturbation vector.
 *
 * @throws std::invalid_argument if not all the components of the perturbation vector are in the (0,1] range.
 */
void mbh::set_perturb(const vector_double &perturb)
{
    if (!std::all_of(perturb.begin(), perturb.end(),
                     [](double item) { return (!std::isnan(item) && item > 0. && item <= 1.); })) {
        pagmo_throw(std::invalid_argument,
                    "The perturbation must have all components in (0, 1], while that is not the case.");
    }
    m_perturb = perturb;
}

/// Algorithm's thread safety level.
/**
 * The thread safety of this meta-algorithm is the minimum between the thread safety
 * of the internal pagmo::algorithm and the basic thread safety level. I.e., this algorithm
 * never provides more than the basic thread safety level.
 *
 * @return the thread safety level of this algorithm.
 */
thread_safety mbh::get_thread_safety() const
{
    return std::min(m_algorithm.get_thread_safety(), thread_safety::basic);
}

/// Extra info.
/**
 * @return a string containing extra info on the algorithm.
 */
std::string mbh::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tStop: ", m_stop);
    stream(ss, "\n\tPerturbation vector: ", m_perturb);
    stream(ss, "\n\tSeed: ", m_seed);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
    stream(ss, "\n\n\tInner algorithm: ", m_algorithm.get_name());
    stream(ss, "\n\tInner algorithm extra info: ");
    stream(ss, "\n", m_algorithm.get_extra_info());
    return ss.str();
}

/// Object serialization.
/**
 * This method will save/load \p this into the archive \p ar.
 *
 * @param ar target archive.
 *
 * @throws unspecified any exception thrown by the serialization of the inner algorithm and of primitive types.
 */
template <typename Archive>
void mbh::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_algorithm, m_stop, m_perturb, m_e, m_seed, m_verbosity, m_log);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::mbh)
