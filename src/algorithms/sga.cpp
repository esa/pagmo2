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
#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/bimap.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

namespace detail
{

namespace
{

// All this scaffolding is to establish a one to one correspondance between enums and genetic operator types
// represented as strings.
using sga_selection_map_t = boost::bimap<std::string, sga_selection>;
using sga_crossover_map_t = boost::bimap<std::string, sga_crossover>;
using sga_mutation_map_t = boost::bimap<std::string, sga_mutation>;

// Helper init functions
sga_selection_map_t sga_init_selection_map()
{
    sga_selection_map_t retval;
    using value_type = sga_selection_map_t::value_type;
    retval.insert(value_type("tournament", sga_selection::TOURNAMENT));
    retval.insert(value_type("truncated", sga_selection::TRUNCATED));
    return retval;
}

sga_crossover_map_t sga_init_crossover_map()
{
    sga_crossover_map_t retval;
    using value_type = sga_crossover_map_t::value_type;
    retval.insert(value_type("exponential", sga_crossover::EXPONENTIAL));
    retval.insert(value_type("binomial", sga_crossover::BINOMIAL));
    retval.insert(value_type("sbx", sga_crossover::SBX));
    retval.insert(value_type("single", sga_crossover::SINGLE));
    return retval;
}

sga_mutation_map_t sga_init_mutation_map()
{
    sga_mutation_map_t retval;
    using value_type = sga_mutation_map_t::value_type;
    retval.insert(value_type("gaussian", sga_mutation::GAUSSIAN));
    retval.insert(value_type("uniform", sga_mutation::UNIFORM));
    retval.insert(value_type("polynomial", sga_mutation::POLYNOMIAL));
    return retval;
}

const sga_selection_map_t sga_selection_map = sga_init_selection_map();
const sga_crossover_map_t sga_crossover_map = sga_init_crossover_map();
const sga_mutation_map_t sga_mutation_map = sga_init_mutation_map();

} // namespace

} // namespace detail

sga::sga(unsigned gen, double cr, double eta_c, double m, double param_m, unsigned param_s, std::string crossover,
         std::string mutation, std::string selection, unsigned seed)
    : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_m(m), m_param_m(param_m), m_param_s(param_s), m_e(seed), m_seed(seed),
      m_verbosity(0u), m_log()
{
    if (cr > 1. || cr < 0.) {
        pagmo_throw(std::invalid_argument, "The crossover probability must be in the [0,1] range, while a value of "
                                               + std::to_string(cr) + " was detected");
    }
    if (eta_c < 1. || eta_c > 100.) {
        pagmo_throw(std::invalid_argument,
                    "The distribution index for SBX crossover must be in [1, 100], while a value of "
                        + std::to_string(eta_c) + " was detected");
    }
    if (m < 0. || m > 1.) {
        pagmo_throw(std::invalid_argument, "The mutation probability must be in the [0,1] range, while a value of "
                                               + std::to_string(cr) + " was detected");
    }
    if (param_s == 0u) {
        pagmo_throw(std::invalid_argument, "The selection parameter must be at least 1, while a value of "
                                               + std::to_string(param_s) + " was detected");
    }
    if (mutation != "gaussian" && mutation != "uniform" && mutation != "polynomial") {
        pagmo_throw(
            std::invalid_argument,
            R"(The mutation type must either be "gaussian" or "uniform" or "polynomial": unknown type requested: )"
                + mutation);
    }
    if (selection != "truncated" && selection != "tournament") {
        pagmo_throw(
            std::invalid_argument,
            R"(The selection type must either be "roulette" or "truncated" or "tournament": unknown type requested: )"
                + selection);
    }
    if (crossover != "exponential" && crossover != "binomial" && crossover != "sbx" && crossover != "single") {
        pagmo_throw(
            std::invalid_argument,
            R"(The crossover type must either be "exponential" or "binomial" or "sbx" or "single": unknown type requested: )"
                + crossover);
    }
    // param_m represents the distribution index if polynomial mutation is selected
    if (mutation == "polynomial" && (param_m < 1. || param_m > 100.)) {
        pagmo_throw(std::invalid_argument, "Polynomial mutation was selected, the mutation parameter (distribution "
                                           "index) must be in [1, 100], while a value of "
                                               + std::to_string(param_m) + " was detected");
    }

    // otherwise param_m represents the width of the mutation relative to the box bounds
    if (mutation != "polynomial" && (param_m < 0 || param_m > 1.)) {
        pagmo_throw(std::invalid_argument, "The mutation parameter must be in [0,1], while a value of "
                                               + std::to_string(param_m) + " was detected");
    }
    // We can now init the data members representing the various choices made using std::string
    m_crossover = detail::sga_crossover_map.left.at(crossover);
    m_mutation = detail::sga_mutation_map.left.at(mutation);
    m_selection = detail::sga_selection_map.left.at(selection);
}

/// Algorithm evolve method
/**
 * Evolves the population for a maximum number of generations
 *
 * @param pop population to be evolved
 * @return evolved population
 * @throws std::invalid_argument if the problem is multi-objective or constrained, if the population size is smaller
 * than 2, if \p param_s is larger than the population size, if the size of \p pop is odd and a "sbx" crossover has
 * been selected upon construction.
 */
population sga::evolve(population pop) const
{
    const auto &prob = pop.get_problem();
    auto dim_i = prob.get_nix();
    const auto bounds = prob.get_bounds();
    auto NP = pop.size();
    auto fevals0 = prob.get_fevals(); // fevals already made
    auto count = 1u;                  // regulates the screen output
    // PREAMBLE-------------------------------------------------------------------------------------------------
    // Check whether the problem/population are suitable for bee_colony
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Constraints detected in " + prob.get_name() + " instance. " + get_name()
                                               + " cannot deal with them");
    }
    if (prob.get_nf() != 1u) {
        pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (NP < 2u) {
        pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 2 individuals in the population, "
                                               + std::to_string(NP) + " detected");
    }
    if (m_param_s > pop.size()) {
        pagmo_throw(std::invalid_argument,
                    "The parameter for selection must be smaller than the population size, while a value of: "
                        + std::to_string(m_param_s)
                        + " was detected in a population of size: " + std::to_string(pop.size()));
    }
    if (m_crossover == detail::sga_crossover::SBX && (pop.size() % 2 != 0u)) {
        pagmo_throw(std::invalid_argument,
                    "Population size must be even if sbx crossover is selected. Detected pop size is: "
                        + std::to_string(pop.size()));
    }
    // Get out if there is nothing to do.
    if (m_gen == 0u) {
        return pop;
    }
    // ---------------------------------------------------------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    double improvement; // stores the difference in fitness between parents and offsprings
    std::uniform_int_distribution<unsigned> urng;
    for (decltype(m_gen) i = 1u; i <= m_gen; ++i) {
        // 1 - if the problem is stochastic we change seed and re-evaluate the entire population
        if (prob.is_stochastic()) {
            pop.get_problem().set_seed(urng(m_e));
            // re-evaluate the whole population w.r.t. the new seed
            for (decltype(pop.size()) j = 0u; j < pop.size(); ++j) {
                pop.set_xf(j, pop.get_x()[j], prob.fitness(pop.get_x()[j]));
            }
        }
        auto XNEW = pop.get_x();
        auto FNEW = pop.get_f();
        // 2 - Selection.
        auto selected_idx = perform_selection(FNEW);
        for (decltype(NP) j = 0u; j < NP; ++j) {
            XNEW[j] = pop.get_x()[selected_idx[j]];
        }
        // 3 - Crossover
        perform_crossover(XNEW, prob.get_bounds(), dim_i);
        // 4 - Mutation
        perform_mutation(XNEW, prob.get_bounds(), dim_i);
        // 5 - Evaluate the new population
        for (decltype(NP) j = 0u; j < NP; ++j) {
            FNEW[j] = prob.fitness(XNEW[j]);
        }
        // 6 - Logs and prints
        if (m_verbosity > 0u) {
            double bestf = std::numeric_limits<double>::max();
            for (decltype(NP) j = 0u; j < NP; ++j) {
                if (FNEW[j][0] < bestf) bestf = FNEW[j][0];
            }
            improvement = pop.get_f()[pop.best_idx()][0] - bestf;
            // (verbosity modes = 1: a line is added at each improvement
            // (verbosity modes > 1: a line is added every m_verbosity generations)
            if (((i % m_verbosity == 1u) && (m_verbosity > 1u)) || ((improvement > 0) && (m_verbosity == 1u))) {

                // Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Improvement:", '\n');
                }
                print(std::setw(7), i, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                      pop.get_f()[pop.best_idx()][0], std::setw(15), improvement, '\n');
                ++count;
                // Logs
                m_log.emplace_back(i, prob.get_fevals() - fevals0, pop.get_f()[pop.best_idx()][0], improvement);
            }
        }
        // 7 - And insert the best into pop
        // We add all the parents to the new population
        for (decltype(NP) j = 0u; j < NP; ++j) {
            XNEW.push_back(pop.get_x()[j]);
            FNEW.push_back(pop.get_f()[j]);
        }
        // sort the entire pool
        std::vector<vector_double::size_type> best_idxs(FNEW.size());
        std::iota(best_idxs.begin(), best_idxs.end(), vector_double::size_type(0u));
        std::sort(best_idxs.begin(), best_idxs.end(), [&FNEW](vector_double::size_type a, vector_double::size_type b) {
            return detail::less_than_f(FNEW[a][0], FNEW[b][0]);
        });
        for (decltype(NP) j = 0u; j < NP; ++j) {
            pop.set_xf(j, XNEW[best_idxs[j]], FNEW[best_idxs[j]]);
        }
    }
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void sga::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Extra info
/**
 * @return a string containing extra info on the algorithm
 */
std::string sga::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tNumber of generations: ", m_gen);
    stream(ss, "\n\tCrossover:");
    stream(ss, "\n\t\tType: " + detail::sga_crossover_map.right.at(m_crossover));
    stream(ss, "\n\t\tProbability: ", m_cr);
    if (m_crossover == detail::sga_crossover::SBX) stream(ss, "\n\t\tDistribution index: ", m_eta_c);
    stream(ss, "\n\tMutation:");
    stream(ss, "\n\t\tType: ", detail::sga_mutation_map.right.at(m_mutation));
    stream(ss, "\n\t\tProbability: ", m_m);
    if (m_mutation != detail::sga_mutation::POLYNOMIAL) {
        stream(ss, "\n\t\tWidth: ", m_param_m);
    } else {
        stream(ss, "\n\t\tDistribution index: ", m_param_m);
    }
    stream(ss, "\n\tSelection:");
    stream(ss, "\n\t\tType: ", detail::sga_selection_map.right.at(m_selection));
    if (m_selection == detail::sga_selection::TRUNCATED) stream(ss, "\n\t\tTruncation size: ", m_param_s);
    if (m_selection == detail::sga_selection::TOURNAMENT) stream(ss, "\n\t\tTournament size: ", m_param_s);
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
void sga::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_cr, m_eta_c, m_m, m_param_m, m_param_s, m_mutation, m_selection, m_crossover, m_e,
                    m_seed, m_verbosity, m_log);
}

std::vector<vector_double::size_type> sga::perform_selection(const std::vector<vector_double> &F) const
{
    assert(m_param_s <= F.size());
    std::vector<vector_double::size_type> retval(F.size());
    std::vector<vector_double::size_type> best_idxs(F.size());
    std::iota(best_idxs.begin(), best_idxs.end(), vector_double::size_type(0u));
    switch (m_selection) {
        case (detail::sga_selection::TRUNCATED): {
            std::sort(best_idxs.begin(), best_idxs.end(), [&F](vector_double::size_type a, vector_double::size_type b) {
                return detail::less_than_f(F[a][0], F[b][0]);
            });
            for (decltype(retval.size()) i = 0u; i < retval.size(); ++i) {
                retval[i] = best_idxs[i % m_param_s];
            }
            break;
        }
        case (detail::sga_selection::TOURNAMENT): {
            std::uniform_int_distribution<std::vector<vector_double::size_type>::size_type> dist;
            // We make one tournament for each of the offspring to be generated
            for (decltype(retval.size()) j = 0u; j < retval.size(); ++j) {
                // Fisher Yates algo http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
                // to select m_param_s individial at random
                for (decltype(m_param_s) i = 0u; i < m_param_s; ++i) {
                    dist.param(
                        std::uniform_int_distribution<std::vector<vector_double::size_type>::size_type>::param_type(
                            i, best_idxs.size() - 1u));
                    auto index = dist(m_e);
                    std::swap(best_idxs[index], best_idxs[i]);
                }
                // Find the index of the individual with minimum fitness among the randomly selected ones
                auto winner = best_idxs[0];
                for (decltype(m_param_s) i = 1u; i < m_param_s; ++i) {
                    if (F[best_idxs[i]] < F[winner]) {
                        winner = best_idxs[i];
                    }
                }
                retval[j] = winner;
            }
            break;
        }
    }
    return retval;
}

void sga::perform_crossover(std::vector<vector_double> &X, const std::pair<vector_double, vector_double> &bounds,
                            vector_double::size_type dim_i) const
{
    auto dim = X[0].size();
    assert(X.size() > 1u);
    assert(std::all_of(X.begin(), X.end(), [dim](const vector_double &item) { return item.size() == dim; }));
    std::vector<vector_double::size_type> all_idx(X.size()); // stores indexes to then select one at random
    std::iota(all_idx.begin(), all_idx.end(), vector_double::size_type(0u));
    std::uniform_real_distribution<> drng(0., 1.);
    // We need different loops if the crossover type is "sbx"" as this method creates two offsprings per
    // selected couple.
    if (m_crossover == detail::sga_crossover::SBX) {
        assert(X.size() % 2u == 0u);
        std::shuffle(X.begin(), X.end(), m_e);
        for (decltype(X.size()) i = 0u; i < X.size(); i += 2) {
            auto children = sbx_crossover_impl(X[i], X[i + 1], bounds, dim_i);
            X[i] = children.first;
            X[i + 1] = children.second;
        }
    } else {
        auto XCOPY = X;
        std::uniform_int_distribution<std::vector<vector_double::size_type>::size_type> rnd_gene_idx(0, dim - 1u);
        std::uniform_int_distribution<std::vector<vector_double::size_type>::size_type> rnd_skip_first_idx(
            1, all_idx.size() - 1);
        // Start of main loop through the X
        for (decltype(X.size()) i = 0u; i < X.size(); ++i) {
            // 1 - we select a mating partner
            std::swap(all_idx[0], all_idx[i]);
            auto partner_idx = rnd_skip_first_idx(m_e);
            // 2 - We rename these chromosomes for code clarity
            auto &child = X[i];
            const auto &parent2 = XCOPY[all_idx[partner_idx]];
            // 3 - We perform crossover according to the selected method
            switch (m_crossover) {
                case (detail::sga_crossover::EXPONENTIAL): {
                    auto n = rnd_gene_idx(m_e);
                    decltype(dim) L = 0u;
                    do {
                        child[n] = parent2[n];
                        n = (n + 1u) % dim;
                        L++;
                    } while ((drng(m_e) < m_cr) && (L < dim));
                    break;
                }
                case (detail::sga_crossover::BINOMIAL): {
                    auto n = rnd_gene_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {    /* performs D binomial trials */
                        if ((drng(m_e) < m_cr) || L + 1 == dim) { /* changes at least one parameter */
                            child[n] = parent2[n];
                        }
                        n = (n + 1) % dim;
                    }
                    break;
                }
                case (detail::sga_crossover::SINGLE): {
                    if (drng(m_e) < m_cr) {
                        auto n = rnd_gene_idx(m_e);
                        for (decltype(dim) j = n; j < dim; ++j) {
                            child[j] = parent2[j];
                        }
                    }
                    break;
                }
                // LCOV_EXCL_START
                default: {
                    assert(false); // the code should never reach this point
                    break;
                }
                    // LCOV_EXCL_STOP
            }
        }
    }
}

void sga::perform_mutation(std::vector<vector_double> &X, const std::pair<vector_double, vector_double> &bounds,
                           vector_double::size_type dimi) const
{
    // Asserting the correct behaviour of input parameters
    assert(X.size() > 1u);
    auto dim = X[0].size();
    assert(std::all_of(X.begin(), X.end(), [dim](const vector_double &item) { return item.size() == dim; }));
    assert(bounds.first.size() == bounds.second.size());
    assert(bounds.first.size() == X[0].size());

    // Renaming some dimensions
    auto dimc = dim - dimi;
    // Problem bounds
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    // Random distributions
    std::uniform_real_distribution<> drng(0., 1.);  // to generate a number in [0, 1)
    std::uniform_real_distribution<> drngs(-1, 1.); // to generate a number in [-1, 1)
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<int> rnd_lb_ub; // to generate a random int between int bounds
    // This will contain the indexes of the genes to be mutated
    std::vector<vector_double::size_type> to_be_mutated(dim);
    std::iota(to_be_mutated.begin(), to_be_mutated.end(), vector_double::size_type(0u));
    // Then we start tha main loop through the population
    for (decltype(X.size()) i = 0u; i < X.size(); ++i) {
        // We select the indexes to be mutated (the first N of to_be_mutated)
        std::shuffle(to_be_mutated.begin(), to_be_mutated.end(), m_e);
        auto N = std::binomial_distribution<vector_double::size_type>(dim, m_m)(m_e);
        // We ensure at least one is mutated if m_m > 0
        // if (m_m > 0. and N == 0u) N = 1;
        // We apply the mutation scheme
        switch (m_mutation) {
            case (detail::sga_mutation::UNIFORM): {
                // Start of main loop through the chromosome
                for (decltype(N) j = 0u; j < N; ++j) {
                    auto gene_idx = to_be_mutated[j];
                    if (gene_idx < dimc) {
                        X[i][gene_idx] = uniform_real_from_range(lb[gene_idx], ub[gene_idx], m_e);
                    } else {
                        rnd_lb_ub.param(std::uniform_int_distribution<int>::param_type(static_cast<int>(lb[gene_idx]),
                                                                                       static_cast<int>(ub[gene_idx])));
                        X[i][gene_idx] = static_cast<double>(rnd_lb_ub(m_e));
                    }
                }
                break;
            }
            case (detail::sga_mutation::GAUSSIAN): {
                // Start of main loop through the chromosome
                for (decltype(N) j = 0u; j < N; ++j) {
                    auto gene_idx = to_be_mutated[j];
                    auto std = (ub[gene_idx] - lb[gene_idx]) * m_param_m;
                    if (gene_idx < dimc) {
                        X[i][gene_idx] += normal(m_e) * std;
                    } else {
                        X[i][gene_idx] += std::round(normal(m_e) * std);
                    }
                }
                break;
            }
            case (detail::sga_mutation::POLYNOMIAL): { // https://www.iitk.ac.in/kangal/papers/k2012016.pdf
                // Start of main loop through the chromosome
                for (decltype(N) j = 0u; j < N; ++j) {
                    auto gene_idx = to_be_mutated[j];
                    if (gene_idx < dimc) {
                        double u = drng(m_e);
                        if (u <= 0.5) {
                            auto delta_l = std::pow(2. * u, 1. / (1. + m_param_m)) - 1.;
                            X[i][gene_idx] += delta_l * (X[i][gene_idx] - lb[gene_idx]);
                        } else {
                            auto delta_r = 1 - std::pow(2. * (1. - u), 1. / (1. + m_param_m));
                            X[i][gene_idx] += delta_r * (ub[gene_idx] - X[i][gene_idx]);
                        }
                    } else {
                        rnd_lb_ub.param(std::uniform_int_distribution<int>::param_type(static_cast<int>(lb[gene_idx]),
                                                                                       static_cast<int>(ub[gene_idx])));
                        X[i][gene_idx] = static_cast<double>(rnd_lb_ub(m_e));
                    }
                }
                break;
            }
        }
        // We fix chromosomes possibly created outside the bounds to stick to the bounds
        detail::force_bounds_stick(X[i], lb, ub);
    }
}

std::pair<vector_double, vector_double> sga::sbx_crossover_impl(const vector_double &parent1,
                                                                const vector_double &parent2,
                                                                const std::pair<vector_double, vector_double> &bounds,
                                                                vector_double::size_type Di) const
{
    // Decision vector dimensions
    auto D = parent1.size();
    auto Dc = D - Di;
    // Problem bounds
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    // declarations
    double y1, y2, yl, yu, rand01, beta, alpha, betaq, c1, c2;
    vector_double::size_type site1, site2;
    // Initialize the child decision vectors
    auto child1 = parent1;
    auto child2 = parent2;
    // Random distributions
    std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)

    // This implements a Simulated Binary Crossover SBX and applies it to the non integer part of the decision
    // vector
    if (drng(m_e) <= m_cr) {
        for (decltype(Dc) i = 0u; i < Dc; i++) {
            if ((drng(m_e) <= 0.5) && (std::abs(parent1[i] - parent2[i])) > 1e-14 && lb[i] != ub[i]) {
                if (parent1[i] < parent2[i]) {
                    y1 = parent1[i];
                    y2 = parent2[i];
                } else {
                    y1 = parent2[i];
                    y2 = parent1[i];
                }
                yl = lb[i];
                yu = ub[i];
                rand01 = drng(m_e);
                beta = 1. + (2. * (y1 - yl) / (y2 - y1));
                alpha = 2. - std::pow(beta, -(m_eta_c + 1.));
                if (rand01 <= (1. / alpha)) {
                    betaq = std::pow((rand01 * alpha), (1. / (m_eta_c + 1.)));
                } else {
                    betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (m_eta_c + 1.)));
                }
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
                beta = 1. + (2. * (yu - y2) / (y2 - y1));
                alpha = 2. - std::pow(beta, -(m_eta_c + 1.));
                if (rand01 <= (1. / alpha)) {
                    betaq = std::pow((rand01 * alpha), (1. / (m_eta_c + 1.)));
                } else {
                    betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (m_eta_c + 1.)));
                }
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));
                if (c1 < lb[i]) c1 = lb[i];
                if (c2 < lb[i]) c2 = lb[i];
                if (c1 > ub[i]) c1 = ub[i];
                if (c2 > ub[i]) c2 = ub[i];
                if (drng(m_e) <= .5) {
                    child1[i] = c1;
                    child2[i] = c2;
                } else {
                    child1[i] = c2;
                    child2[i] = c1;
                }
            }
        }
    }
    // This implements two-point binary crossover and applies it to the integer part of the chromosome
    for (decltype(Dc) i = Dc; i < D; ++i) {
        // in this loop we are sure Di is at least 1
        std::uniform_int_distribution<vector_double::size_type> ra_num(0, Di - 1u);
        if (drng(m_e) <= m_cr) {
            site1 = ra_num(m_e);
            site2 = ra_num(m_e);
            if (site1 > site2) {
                std::swap(site1, site2);
            }
            for (decltype(site1) j = 0u; j < site1; ++j) {
                child1[j] = parent1[j];
                child2[j] = parent2[j];
            }
            for (decltype(site2) j = site1; j < site2; ++j) {
                child1[j] = parent2[j];
                child2[j] = parent1[j];
            }
            for (decltype(Di) j = site2; j < Di; ++j) {
                child1[j] = parent1[j];
                child2[j] = parent2[j];
            }
        } else {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
    }
    return std::make_pair(std::move(child1), std::move(child2));
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::sga)
