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
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/moead.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/multi_objective.hpp>

namespace pagmo
{

moead::moead(unsigned gen, std::string weight_generation, std::string decomposition, population::size_type neighbours,
             double CR, double F, double eta_m, double realb, unsigned limit, bool preserve_diversity, unsigned seed)
    : m_gen(gen), m_weight_generation(weight_generation), m_decomposition(decomposition), m_neighbours(neighbours),
      m_CR(CR), m_F(F), m_eta_m(eta_m), m_realb(realb), m_limit(limit), m_preserve_diversity(preserve_diversity),
      m_e(seed), m_seed(seed), m_verbosity(0u)
{
    // Sanity checks
    if (m_weight_generation != "random" && m_weight_generation != "grid" && m_weight_generation != "low discrepancy") {
        pagmo_throw(std::invalid_argument, "Weight generation method requested is '" + m_weight_generation
                                               + "', but only one of 'random', 'low discrepancy', 'grid' is allowed");
    }
    if (m_decomposition != "tchebycheff" && m_decomposition != "weighted" && m_decomposition != "bi") {
        pagmo_throw(std::invalid_argument, "Weight generation method requested is '" + m_decomposition
                                               + "', but only one of 'tchebycheff', 'weighted', 'bi' is allowed");
    }
    if (CR > 1.0 || CR < 0.) {
        pagmo_throw(
            std::invalid_argument,
            "The parameter CR (used by the differential evolution operator) needs to be in [0,1], while a value of "
                + std::to_string(CR) + " was detected");
    }
    if (F > 1.0 || F < 0.) {
        pagmo_throw(
            std::invalid_argument,
            "The parameter F (used by the differential evolution operator) needs to be in [0,1], while a value of "
                + std::to_string(F) + " was detected");
    }
    if (eta_m < 0.) {
        pagmo_throw(std::invalid_argument,
                    "The distribution index for the polynomial mutation (eta_m) needs to be positive, while a value of "
                        + std::to_string(eta_m) + " was detected");
    }
    if (realb > 1.0 || realb < 0.) {
        pagmo_throw(std::invalid_argument,
                    "The chance of considering a neighbourhood (realb) needs to be in [0,1], while a value of "
                        + std::to_string(realb) + " was detected");
    }
    if (neighbours < 2) {
        pagmo_throw(std::invalid_argument, "The size of the weight's neighborhood needs to be >= 2, while a size of "
                                               + std::to_string(neighbours) + " was detected");
    }
}

/// Algorithm evolve method
/**
 * Evolves the population for the requested number of generations.
 *
 * @param pop population to be evolved
 * @return evolved population
 */
population moead::evolve(population pop) const
{
    // We store some useful variables
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed
    auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto NP = pop.size();

    auto fevals0 = prob.get_fevals(); // discount for the fevals already made
    unsigned count = 1u;              // regulates the screen output

    // PREAMBLE-------------------------------------------------------------------------------------------------
    // We start by checking that the problem is suitable for this
    // particular algorithm.
    if (detail::some_bound_is_equal(prob)) {
        pagmo_throw(std::invalid_argument,
                    get_name()
                        + " cannot work on problems having a lower bound equal to an upper bound. Check your bounds.");
    }
    if (!NP) {
        pagmo_throw(std::invalid_argument, get_name() + " cannot work on an empty population");
    }
    if (prob.get_nf() < 2u) {
        pagmo_throw(std::invalid_argument, "This is a multiobjective algortihm, while number of objectives detected in "
                                               + prob.get_name() + " is " + std::to_string(prob.get_nf()));
    }
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (prob.is_stochastic()) {
        pagmo_throw(std::invalid_argument,
                    "The problem appears to be stochastic " + get_name() + " cannot deal with it");
    }
    if (m_neighbours > NP - 1u) {
        pagmo_throw(std::invalid_argument, "The neighbourhood size specified (T) is " + std::to_string(m_neighbours)
                                               + ": too large for the input population having size "
                                               + std::to_string(NP));
    }
    // Get out if there is nothing to do.
    if (m_gen == 0u) {
        return pop;
    }
    // Generate NP weight vectors for the decomposed problems. Will throw if the population size is not compatible
    // with the weight generation scheme chosen
    auto weights = decomposition_weights(prob.get_nf(), NP, m_weight_generation, m_e);
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    // Setting up necessary quantities------------------------------------------------------------------------------
    // Random distributions
    std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)
    std::uniform_int_distribution<vector_double::size_type> p_idx(
        0u, NP - 1u); // to generate a random index for the population
                      // Declaring the candidate chromosome
    vector_double candidate(dim);
    // We compute, for each vector of weights, the k = m_neighbours neighbours
    auto neigh_idxs = kNN(weights, m_neighbours);
    // We compute the initial ideal point (will be adapted along the course of the algorithm)
    vector_double ideal_point = ideal(pop.get_f());
    // We create the container that will represent a pseudo-random permutation of the population indexes 1..NP
    std::vector<population::size_type> shuffle(NP);
    std::iota(shuffle.begin(), shuffle.end(), std::vector<population::size_type>::size_type(0u));

    // Main MOEA/D loop --------------------------------------------------------------------------------------------
    for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
        // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                // We compute the average decomposed fitness (ADF)
                auto adf = 0.;
                for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
                    adf += decompose_objectives(pop.get_f()[i], weights[i], ideal_point, m_decomposition)[0];
                }
                // Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "ADF:");
                    for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                        if (i >= 5u) {
                            print(std::setw(15), "... :");
                            break;
                        }
                        print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                    }
                    print('\n');
                }
                print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15), adf);
                for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                    if (i >= 5u) {
                        break;
                    }
                    print(std::setw(15), ideal_point[i]);
                }
                print('\n');
                ++count;
                // Logs
                m_log.emplace_back(gen, prob.get_fevals() - fevals0, adf, ideal_point);
            }
        }
        // 1 - Shuffle the population indexes
        std::shuffle(shuffle.begin(), shuffle.end(), m_e);
        // 2 - Loop over the shuffled NP decomposed problems
        for (auto n : shuffle) {
            // 3 - if the diversity preservation mechanism is active we select at random whether to consider the
            // whole
            // population or just a neighbourhood to select two parents
            bool whole_population;
            if (drng(m_e) < m_realb || !m_preserve_diversity) {
                whole_population = false; // neighborhood
            } else {
                whole_population = true; // whole population
            }
            // 4 - We select two parents in the neighbourhood
            std::vector<population::size_type> parents_idx(2);
            parents_idx = select_parents(n, neigh_idxs, whole_population);
            // 5 - Crossover using the Differential Evolution operator (binomial crossover)
            for (decltype(dim) kk = 0u; kk < dim; ++kk) {
                if (drng(m_e) < m_CR) {
                    /*Selected Two Parents*/
                    candidate[kk] = pop.get_x()[n][kk]
                                    + m_F * (pop.get_x()[parents_idx[0]][kk] - pop.get_x()[parents_idx[1]][kk]);
                    // Fix the bounds
                    if (candidate[kk] < lb[kk]) {
                        candidate[kk] = lb[kk] + drng(m_e) * (pop.get_x()[n][kk] - lb[kk]);
                    }
                    if (candidate[kk] > ub[kk]) {
                        candidate[kk] = ub[kk] - drng(m_e) * (ub[kk] - pop.get_x()[n][kk]);
                    }
                } else {
                    candidate[kk] = pop.get_x()[n][kk];
                }
            }
            // 6 - We apply a further mutation using polynomial mutation
            polynomial_mutation(candidate, pop, 1.0 / static_cast<double>(dim));
            // 7- We evaluate the fitness function.
            auto new_f = prob.fitness(candidate);
            // 8 - We update the ideal point
            for (decltype(prob.get_nf()) j = 0u; j < prob.get_nf(); ++j) {
                ideal_point[j] = std::min(new_f[j], ideal_point[j]);
            }
            std::transform(ideal_point.begin(), ideal_point.end(), new_f.begin(), ideal_point.begin(),
                           [](double a, double b) { return std::min(a, b); });
            // 9 - We insert the newly found solution into the population
            decltype(NP) size, time = 0;
            // First try on problem n
            auto f1 = decompose_objectives(pop.get_f()[n], weights[n], ideal_point, m_decomposition);
            auto f2 = decompose_objectives(new_f, weights[n], ideal_point, m_decomposition);
            if (f2[0] < f1[0]) {
                pop.set_xf(n, candidate, new_f);
                time++;
            }
            // Then, on neighbouring problems up to m_limit (to preserve diversity)
            if (whole_population) {
                size = NP;
            } else {
                size = neigh_idxs[n].size();
            }
            std::vector<population::size_type> shuffle2(size);
            std::iota(shuffle2.begin(), shuffle2.end(), std::vector<population::size_type>::size_type(0u));
            std::shuffle(shuffle2.begin(), shuffle2.end(), m_e);
            for (decltype(size) k = 0u; k < size; ++k) {
                population::size_type pick;
                if (whole_population) {
                    pick = shuffle2[k];
                } else {
                    pick = neigh_idxs[n][shuffle2[k]];
                }
                f1 = decompose_objectives(pop.get_f()[pick], weights[pick], ideal_point, m_decomposition);
                f2 = decompose_objectives(new_f, weights[pick], ideal_point, m_decomposition);
                if (f2[0] < f1[0]) {
                    pop.set_xf(pick, candidate, new_f);
                    time++;
                }
                // the maximal number of solutions updated is not allowed to exceed 'limit' if diversity is to be
                // preserved
                if (time >= m_limit && m_preserve_diversity) {
                    break;
                }
            }
        }
    }
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void moead::set_seed(unsigned seed)
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
std::string moead::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\tWeight generation: ", m_weight_generation);
    stream(ss, "\n\tDecomposition method: ", m_decomposition);
    stream(ss, "\n\tNeighbourhood size: ", m_neighbours);
    stream(ss, "\n\tParameter CR: ", m_CR);
    stream(ss, "\n\tParameter F: ", m_F);
    stream(ss, "\n\tDistribution index: ", m_eta_m);
    stream(ss, "\n\tChance for diversity preservation: ", m_realb);
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
void moead::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_weight_generation, m_decomposition, m_neighbours, m_CR, m_F, m_eta_m, m_realb, m_limit,
                    m_preserve_diversity, m_e, m_seed, m_verbosity, m_log);
}

// Performs polynomial mutation (same as nsgaII)
void moead::polynomial_mutation(vector_double &child, const population &pop, double rate) const
{
    const auto &prob = pop.get_problem();
    auto D = prob.get_nx(); // This getter does not return a const reference but a copy
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    double rnd, delta1, delta2, mut_pow, deltaq;
    double y, yl, yu, val, xy;
    std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)

    // This implements the real polinomial mutation of an individual
    for (decltype(D) j = 0u; j < D; ++j) {
        if (drng(m_e) <= rate) {
            y = child[j];
            yl = lb[j];
            yu = ub[j];
            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rnd = drng(m_e);
            mut_pow = 1. / (m_eta_m + 1.);
            if (rnd <= 0.5) {
                xy = 1. - delta1;
                val = 2. * rnd + (1. - 2. * rnd) * (std::pow(xy, (m_eta_m + 1.)));
                deltaq = std::pow(val, mut_pow) - 1.;
            } else {
                xy = 1. - delta2;
                val = 2. * (1. - rnd) + 2. * (rnd - 0.5) * (std::pow(xy, (m_eta_m + 1.)));
                deltaq = 1. - (std::pow(val, mut_pow));
            }
            y = y + deltaq * (yu - yl);
            if (y < yl) y = yl;
            if (y > yu) y = yu;
            child[j] = y;
        }
    }
}

std::vector<population::size_type>
moead::select_parents(population::size_type n, const std::vector<std::vector<population::size_type>> &neigh_idx,
                      bool whole_population) const
{
    std::vector<population::size_type> retval;
    auto ss = neigh_idx[n].size();
    decltype(ss) p;
    assert(neigh_idx[n].size() > 1);

    std::uniform_int_distribution<vector_double::size_type> p_idx(
        0, neigh_idx.size() - 1u); // to generate a random index for the neighbourhood

    while (retval.size() < 2u) {
        if (!whole_population) {
            p = neigh_idx[n][p_idx(m_e) % ss];
        } else {
            p = p_idx(m_e);
        }
        bool flag = true;
        for (decltype(retval.size()) i = 0u; i < retval.size(); i++) {
            if (retval[i] == p) // p is in the list
            {
                flag = false;
                break;
            }
        }
        if (flag) retval.push_back(p);
    }
    return retval;
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::moead)
