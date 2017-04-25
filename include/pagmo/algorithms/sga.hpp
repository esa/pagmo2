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

#ifndef PAGMO_ALGORITHMS_SGA_HPP
#define PAGMO_ALGORITHMS_SGA_HPP

#include <algorithm> // std::sort
#include <boost/bimap.hpp>
#include <iomanip>
#include <numeric> // std::iota
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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
// Usual template trick to have static members in header only libraries
// see: http://stackoverflow.com/questions/18860895/how-to-initialize-static-members-in-the-header
// All this scaffolding is to establish a one to one correspondance between enums and genetic operator types
// represented as strings.
template <typename = void>
struct sga_statics {
    enum class selection { ROULETTE, TOURNAMENT, BESTN };
    enum class crossover { EXPONENTIAL, BINOMIAL, SINGLE, SBX };
    enum class mutation { GAUSSIAN, UNIFORM, POLYNOMIAL };
    using selection_map_t = boost::bimap<std::string, selection>;
    using crossover_map_t = boost::bimap<std::string, crossover>;
    using mutation_map_t = boost::bimap<std::string, mutation>;
    const static selection_map_t m_selection_map;
    const static crossover_map_t m_crossover_map;
    const static mutation_map_t m_mutation_map;
};
// Helper init functions
inline typename sga_statics<>::selection_map_t init_selection_map()
{
    typename sga_statics<>::selection_map_t retval;
    using value_type = typename sga_statics<>::selection_map_t::value_type;
    retval.insert(value_type("roulette", sga_statics<>::selection::ROULETTE));
    retval.insert(value_type("tournament", sga_statics<>::selection::TOURNAMENT));
    retval.insert(value_type("bestN", sga_statics<>::selection::BESTN));
    return retval;
}
inline typename sga_statics<>::crossover_map_t init_crossover_map()
{
    typename sga_statics<>::crossover_map_t retval;
    using value_type = typename sga_statics<>::crossover_map_t::value_type;
    retval.insert(value_type("exponential", sga_statics<>::crossover::EXPONENTIAL));
    retval.insert(value_type("binomial", sga_statics<>::crossover::BINOMIAL));
    retval.insert(value_type("sbx", sga_statics<>::crossover::SBX));
    retval.insert(value_type("single-point", sga_statics<>::crossover::SINGLE));
    return retval;
}
inline typename sga_statics<>::mutation_map_t init_mutation_map()
{
    typename sga_statics<>::mutation_map_t retval;
    using value_type = typename sga_statics<>::mutation_map_t::value_type;
    retval.insert(value_type("gaussian", sga_statics<>::mutation::GAUSSIAN));
    retval.insert(value_type("uniform", sga_statics<>::mutation::UNIFORM));
    retval.insert(value_type("polynomial", sga_statics<>::mutation::POLYNOMIAL));
    return retval;
}
// We now init the various members
template <typename T>
const typename sga_statics<T>::selection_map_t sga_statics<T>::m_selection_map = init_selection_map();
template <typename T>
const typename sga_statics<T>::crossover_map_t sga_statics<T>::m_crossover_map = init_crossover_map();
template <typename T>
const typename sga_statics<T>::mutation_map_t sga_statics<T>::m_mutation_map = init_mutation_map();
} // end namespace detail

/// A Simple Genetic Algorithm
/**
 * \image html sga.jpg "The DNA Helix" width=3cm
 *
 * Approximately in the same decade as Evolutionary Strategies (see pagmo::sea) were studied, a different group
 * led by John Holland, and later by his student David Goldberg, introduced and studied an algorithmic framework called
 * "genetic algorithms" that were, essentially, leveraging on the same idea but introducing also crossover as a genetic
 * operator. This led to a few decades of confusion and discussions on what was an evolutionary startegy and what a
 * genetic algorithm and on whether the crossover was a useful operator or mutation only algorithms were to be
 * preferred.
 *
 * In pagmo we provide a rather classical implementation of a genetic algorithm, letting the user choose between
 * choosen selection schemes, crossover types, mutation types and reinsertion scheme.
 *
 * **NOTE** This algorithm will work only for box bounded problems.
 *
 * **NOTE** Specifying the parameter \p int_dim a part of the decision vector (at the end) will be treated as integers
 *
 */
class sga : private detail::sga_statics<>
{
public:
    /// Single entry of the log (gen, fevals, best, cur_best)
    // typedef std::tuple<unsigned, unsigned long long, double, double> log_line_type;
    /// The log
    // typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs a simple genetic algorithm.
     *
     * @param gen number of generations.
     * @param cr crossover probability.
     * @param eta_c distribution index for "sbx" crossover. This is an inactive parameter if other types of crossovers
     * are selected.
     * @param m mutation probability.
     * @param param_m distribution index (in polynomial mutation), otherwise width of the mutation.
     * @param elitism number of parents that gets carried over to the next generation.
     * @param param_s when "bestN" selection is used this indicates the percentage of best individuals to use. when
     * "tournament" selection is used this indicates the size of the tournament.
     * This is an inactive parameter if other types of selection are selected.
     * @param mutation the mutation strategy. One of "gaussian", "polynomial" or "uniform".
     * @param selection the selection strategy. One of "roulette", "tournament", "bestN".
     * @param crossover the crossover strategy. One of "exponential", "binomial", "single-point" or "sbx"
     * @param int_dim the number of element in the chromosome to be treated as integers.
     *
     * @throws std::invalid_argument if \p cr not in [0,1), \p eta_c not in [1, 100), \p m not in [0,1], \p elitism < 1
     * \p mutation not one of "gaussian", "uniform" or "polynomial", \p selection not one of "roulette" or "bestN"
     * \p crossover not one of "exponential", "binomial", "sbx" or "single-point", if \p param_m is not in [0,1] and
     * \p mutation is not "polynomial" or \p mutation is not in [1,100] and \p mutation is polynomial.
     */
    sga(unsigned gen = 1u, double cr = .95, double eta_c = 10., double m = 0.02, double param_m = 0.5,
        unsigned elitism = 5u, unsigned param_s = 5u, std::string mutation = "gaussian",
        std::string selection = "roulette", std::string crossover = "exponential",
        vector_double::size_type int_dim = 0u, unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_m(m), m_param_m(param_m), m_elitism(elitism), m_param_s(param_s),
          m_int_dim(int_dim), m_e(seed), m_seed(seed), m_verbosity(0u) //, m_log()
    {
        if (cr >= 1. || cr < 0.) {
            pagmo_throw(std::invalid_argument, "The crossover probability must be in the [0,1[ range, while a value of "
                                                   + std::to_string(cr) + " was detected");
        }
        if (eta_c < 1. || eta_c >= 100.) {
            pagmo_throw(std::invalid_argument,
                        "The distribution index for SBX crossover must be in [1, 100[, while a value of "
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
        if (selection != "roulette" && selection != "bestN" && selection != "tournament") {
            pagmo_throw(
                std::invalid_argument,
                R"(The selection type must either be "roulette" or "bestN" or "tournament": unknown type requested: )"
                    + selection);
        }
        if (crossover != "exponential" && crossover != "binomial" && crossover != "sbx"
            && crossover != "single-point") {
            pagmo_throw(
                std::invalid_argument,
                R"(The crossover type must either be "exponential" or "binomial" or "sbx" or "single-point": unknown type requested: )"
                    + crossover);
        }
        // param_m represents the distribution index if polynomial mutation is selected
        if (mutation == "polynomial" && (param_m < 1. || param_m > 100.)) {
            pagmo_throw(
                std::invalid_argument,
                "Polynomial mutation was selected, the mutation parameter must be in [1, 100], while a value of "
                    + std::to_string(param_m) + " was detected");
        }

        // otherwise param_m represents the width of the mutation relative to the box bounds
        if (mutation != "polynomial" && (param_m < 0 || param_m > 1.)) {
            pagmo_throw(std::invalid_argument, "The mutation parameter must be in [0,1], while a value of "
                                                   + std::to_string(param_m) + " was detected");
        }
        // We can now init the data members representing the various choices made using std::string
        m_selection = m_selection_map.left.at(selection);
        m_crossover = m_crossover_map.left.at(crossover);
        m_mutation = m_mutation_map.left.at(mutation);
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * Evolves the population for a maximum number of generations
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained
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
        if (NP < 2u) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 2 individuals in the population, "
                                                   + std::to_string(NP) + " detected");
        }
        if (m_elitism > pop.size()) {
            pagmo_throw(std::invalid_argument,
                        "The elitism must be smaller than the population size, while a value of: "
                            + std::to_string(m_elitism) + " was detected in a population of size: "
                            + std::to_string(pop.size()));
        }
        if (m_param_s > pop.size()) {
            pagmo_throw(std::invalid_argument,
                        "The parameter for selection must be smaller than the population size, while a value of: "
                            + std::to_string(m_param_s) + " was detected in a population of size: "
                            + std::to_string(pop.size()));
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------

        // TODO check bestN and elitism

        // No throws, all valid: we clear the logs
        // m_log.clear();

        for (decltype(m_gen) i = 1u; i <= m_gen; ++i) {
            // 0 - if the problem is stochastic we change seed and re-evaluate the entire population
            if (prob.is_stochastic()) {
                pop.get_problem().set_seed(std::uniform_int_distribution<unsigned int>()(m_e));
                // re-evaluate the whole population w.r.t. the new seed
                for (decltype(pop.size()) j = 0u; j < pop.size(); ++j) {
                    pop.set_xf(j, pop.get_x()[j], prob.fitness(pop.get_x()[j]));
                }
            }
            auto XNEW = pop.get_x();
            auto FNEW = pop.get_f();
            // 1 - Selection.
            auto selected_idx = perform_selection(XNEW, FNEW);
            for (decltype(XNEW.size()) i = 0u; i < XNEW.size(); ++i) {
                XNEW[i] = pop.get_x()[selected_idx[i]];
            }
            // 2 - Crossover
            perform_crossover(XNEW);
            // 3 - Mutation
            perform_mutation(XNEW);
            // 4 - Evaluate the new population
            for (decltype(XNEW.size()) i = 0u; i < XNEW.size(); ++i) {
                FNEW[i] = prob.fitness(XNEW[i]);
            }
            // 5 - Reinsertion
            // We sort the original population
            std::vector<vector_double::size_type> best_parents(pop.get_f().size());
            std::iota(best_parents.begin(), best_parents.end(), vector_double::size_type(0u));
            std::sort(best_parents.begin(), best_parents.end(),
                      [pop](vector_double::size_type a, vector_double::size_type b) {
                          return pop.get_f()[a][0] < pop.get_f()[b][0];
                      });
            // We sort the new population
            std::vector<vector_double::size_type> best_offsprings(FNEW.size());
            std::iota(best_offsprings.begin(), best_offsprings.end(), vector_double::size_type(0u));
            std::sort(
                best_offsprings.begin(), best_offsprings.end(),
                [FNEW](vector_double::size_type a, vector_double::size_type b) { return FNEW[a][0] < FNEW[b][0]; });
            // We re-insert m_elitism best parents and the remaining best children
            population pop_copy(pop);
            for (decltype(m_elitism) i = 0u; i < m_elitism; ++i) {
                pop.set_xf(i, pop_copy.get_x()[best_parents[i]], pop_copy.get_f()[best_parents[i]]);
            }
            for (decltype(pop.size()) i = m_elitism; i < pop.size(); ++i) {
                pop.set_xf(i, XNEW[best_offsprings[i]], FNEW[best_offsprings[i]]);
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
    /// Algorithm name
    /**
    * @return a string containing the algorithm name
    */
    std::string get_name() const
    {
        return "Genetic Algorithm";
    }
    /// Extra informations
    /**
    * @return a string containing extra informations on the algorithm
    */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tNumber of generations: ", m_gen);
        stream(ss, "\n\tElitism: ", m_elitism);
        stream(ss, "\n\tCrossover:");
        stream(ss, "\n\t\tType: " + m_crossover_map.right.at(m_crossover));
        stream(ss, "\n\t\tProbability: ", m_cr);
        if (m_crossover == crossover::SBX) stream(ss, "\n\t\tDistribution index: ", m_eta_c);
        stream(ss, "\n\tMutation:");
        stream(ss, "\n\t\tType: ", m_mutation_map.right.at(m_mutation));
        stream(ss, "\n\t\tProbability: ", m_m);
        if (m_mutation != mutation::POLYNOMIAL) {
            stream(ss, "\n\t\tWidth: ", m_param_m);
        } else {
            stream(ss, "\n\t\tDistribution index: ", m_param_m);
        }
        stream(ss, "\n\tSelection:");
        stream(ss, "\n\t\tType: ", m_selection_map.right.at(m_selection));
        if (m_selection == selection::BESTN) stream(ss, "\n\t\tNumber of best selected: ", m_param_s);
        if (m_selection == selection::TOURNAMENT) stream(ss, "\n\t\tTournament size: ", m_param_s);
        stream(ss, "\n\tSize of the integer part: ", m_int_dim);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
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
    // const log_type &get_log() const
    //{
    //    return m_log;
    //}
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
        ar(m_gen, m_cr, m_eta_c, m_m, m_param_m, m_elitism, m_param_s, m_mutation, m_selection, m_crossover, m_int_dim,
           m_e, m_seed, m_verbosity);
    }

private:
    std::vector<vector_double::size_type> perform_selection(const std::vector<vector_double> &X,
                                                            const std::vector<vector_double> &F) const
    {
        std::vector<vector_double::size_type> retval(X.size());
        std::iota(retval.begin(), retval.end(), vector_double::size_type(0u));
        return retval;
    }
    void perform_crossover(const std::vector<vector_double> &X) const
    {
    }
    void perform_mutation(const std::vector<vector_double> &X) const
    {
    }
    unsigned m_gen;
    double m_cr;
    double m_eta_c;
    double m_m;
    double m_param_m;
    unsigned m_elitism;
    unsigned m_param_s;
    mutation m_mutation;
    selection m_selection;
    crossover m_crossover;
    vector_double::size_type m_int_dim;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    // mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::sga)

#endif
