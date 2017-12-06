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

#ifndef PAGMO_ALGORITHMS_BEE_COLONY_HPP
#define PAGMO_ALGORITHMS_BEE_COLONY_HPP

#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{
/// Artificial Bee Colony Algorithm
/**
 * \image html BeeColony.gif "One funny bee" width=3cm
 *
 * Artificial Bee Colony is an optimization algorithm based on the intelligent foraging behaviour of honey bee swarm,
 * proposed by Karaboga in 2005.
 *
 * The implementation provided for PaGMO is based on the pseudo-code provided in Mernik et al. (2015) - Algorithm 2.
 * pagmo::bee_colony is suitable for box-constrained single-objective continuous optimization.
 *
 * See: http://mf.erciyes.edu.tr/abc/ for the official ABC web site
 *
 * See: https://link.springer.com/article/10.1007/s10898-007-9149-x for the paper that introduces Artificial Bee Colony
 *
 * See: http://www.sciencedirect.com/science/article/pii/S0020025514008378 for the pseudo-code
 */
class bee_colony
{
public:
    /// Single entry of the log (gen, fevals, best, cur_best)
    typedef std::tuple<unsigned, unsigned long long, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs a bee_colony algorithm
     *
     * @param gen number of generations. Note that the total number of fitness evaluations will be 2*gen
     * @param limit maximum number of trials for abandoning a source
     * @param seed seed used by the internal random number generator (default is random)
     *
     * @throws std::invalid_argument if limit equals 0
     */
    bee_colony(unsigned gen = 1u, unsigned limit = 20u, unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_limit(limit), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
        if (limit == 0u) {
            pagmo_throw(std::invalid_argument, "The limit must be greater than 0.");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * Evolves the population for a maximum number of generations
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained or stochastic
     * @throws std::invalid_argument if the population size is smaller than 2
     */
    population evolve(population pop) const
    {
        const auto &prob = pop.get_problem();
        auto dim = prob.get_nx();
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
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
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic. " + get_name() + " cannot deal with it");
        }
        if (NP < 2u) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 2 individuals in the population, "
                                                   + std::to_string(NP) + " detected");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Some vectors used during evolution are declared.
        vector_double newsol(dim); // contains the mutated candidate
        auto X = pop.get_x();
        auto fit = pop.get_f();
        std::vector<unsigned> trial(NP, 0u);
        std::uniform_real_distribution<double> phirng(-1., 1.); // to generate a number in [-1, 1)
        std::uniform_real_distribution<double> rrng(0., 1.);    // to generate a number in [0, 1)
        std::uniform_int_distribution<vector_double::size_type> comprng(
            0u, dim - 1u); // to generate a random index for the component to mutate
        std::uniform_int_distribution<vector_double::size_type> dvrng(
            0u, NP - 2u); // to generate a random index for the second decision vector

        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // 1 - Employed bees phase
            std::vector<unsigned>::size_type mi = 0u;
            for (decltype(NP) i = 1u; i < NP; ++i) {
                if (trial[i] > trial[mi]) {
                    mi = i;
                }
            }
            bool scout = false;
            if (trial[mi] >= m_limit) {
                scout = true;
            }
            for (decltype(NP) i = 0u; i < NP; ++i) {
                if (trial[i] < m_limit || i != mi) {
                    newsol = X[i];
                    // selects a random component of the decision vector
                    auto comp2change = comprng(m_e);
                    // selects a random decision vector in the population other than the current
                    auto rdv = dvrng(m_e);
                    if (rdv >= i) {
                        ++rdv;
                    }
                    // mutate new solution
                    newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[rdv][comp2change]);
                    // if the generated parameter value is out of boundaries, shift it into the boundaries
                    if (newsol[comp2change] < lb[comp2change]) {
                        newsol[comp2change] = lb[comp2change];
                    }
                    if (newsol[comp2change] > ub[comp2change]) {
                        newsol[comp2change] = ub[comp2change];
                    }
                    // if the new solution is better than the old one replace it and reset its trial counter
                    auto newfitness = prob.fitness(newsol);
                    if (newfitness[0] < fit[i][0]) {
                        fit[i][0] = newfitness[0];
                        X[i][comp2change] = newsol[comp2change];
                        pop.set_xf(i, newsol, newfitness);
                        trial[i] = 0;
                    } else {
                        ++trial[i];
                    }
                }
            }
            // 2 - Scout bee phase
            if (scout) {
                for (auto j = 0u; j < dim; ++j) {
                    X[mi][j] = uniform_real_from_range(lb[j], ub[j], m_e);
                }
                pop.set_x(mi, X[mi]); // this causes a fitness evaluation
                trial[mi] = 0;
            }
            // 3 - Onlooker bee phase
            // compute probabilities
            vector_double p(NP);
            auto sump = 0.;
            for (decltype(NP) i = 0u; i < NP; ++i) {
                if (fit[i][0] >= 0.) {
                    p[i] = 1. / (1. + fit[i][0]);
                } else {
                    p[i] = 1. - fit[i][0];
                }
                sump += p[i];
            }
            for (decltype(NP) i = 0u; i < NP; ++i) {
                p[i] /= sump;
            }
            vector_double::size_type s = 0u;
            decltype(NP) t = 0u;
            // for each onlooker bee
            while (t < NP) {
                // probabilistic selection of a food source
                auto r = rrng(m_e);
                if (r < p[s]) {
                    ++t;
                    newsol = X[s];
                    // selects a random component of the decision vector
                    auto comp2change = comprng(m_e);
                    // selects a random decision vector in the population other than the current
                    auto rdv = dvrng(m_e);
                    if (rdv >= s) {
                        ++rdv;
                    }
                    // mutate new solution
                    newsol[comp2change] += phirng(m_e) * (newsol[comp2change] - X[rdv][comp2change]);
                    // if the generated parameter value is out of boundaries, shift it into the boundaries
                    if (newsol[comp2change] < lb[comp2change]) {
                        newsol[comp2change] = lb[comp2change];
                    }
                    if (newsol[comp2change] > ub[comp2change]) {
                        newsol[comp2change] = ub[comp2change];
                    }
                    // if the new solution is better than the old one replace it and reset its trial counter
                    auto newfitness = prob.fitness(newsol);
                    if (newfitness[0] < fit[s][0]) {
                        fit[s][0] = newfitness[0];
                        X[s][comp2change] = newsol[comp2change];
                        pop.set_xf(s, newsol, newfitness);
                        trial[s] = 0;
                    } else {
                        ++trial[s];
                    }
                }
                s = (s + 1) % NP;
            }
            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    auto best_idx = pop.best_idx();
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:",
                              std::setw(15), "Current Best:\n");
                    }
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                          pop.champion_f()[0], std::setw(15), pop.get_f()[best_idx][0], '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(gen, prob.get_fevals() - fevals0, pop.champion_f()[0], pop.get_f()[best_idx][0]);
                }
            }
        }
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
     * Example (verbosity 100):
     * @code{.unparsed}
     *     Gen:        Fevals:          Best: Current Best:
     *        1             40         261363         261363
     *      101           4040        112.237        267.969
     *      201           8040        20.8885        265.122
     *      301          12040        20.6076        20.6076
     *      401          16040         18.252        140.079
     * @endcode
     * Gen is the generation number, Fevals the number of function evaluation used, , Best is the best fitness found,
     * Current best is the best fitness currently in the population.
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
    /// Gets the number of generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned get_gen() const
    {
        return m_gen;
    }
    /// Algorithm name
    /**
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "ABC: Artificial Bee Colony";
    }
    /// Extra informations
    /**
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tMaximum number of generations: ", m_gen);
        stream(ss, "\n\tLimit: ", m_limit);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        stream(ss, "\n\tSeed: ", m_seed);
        return ss.str();
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a bee_colony::log_line_type containing: Gen, Fevals, Current best, Best as
     * described in bee_colony::set_verbosity().
     *
     * @return an <tt> std::vector</tt> of bee_colony::log_line_type containing the logged values Gen, Fevals, Current
     * best, Best
     */
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
        ar(m_gen, m_limit, m_e, m_seed, m_verbosity, m_log);
    }

private:
    unsigned m_gen;
    unsigned m_limit;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::bee_colony)

#endif
