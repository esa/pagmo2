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

#include <iomanip>
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
 * In pagmo we provide a rather classical implementation of a genetic algorithm, letting the user choose the selection
 * schemes, crossover types, mutation types and reinsertion scheme.
 *
 * **NOTE** This algorithm will work only for box bounded problems.
 *
 * **NOTE** Specifying the parameter \p int_dim a part of the decision vector (at the end) will be treated as integers
 *
 */
class sga
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
     * @param elitism generation frequency fot the reinsertion of the best individual.
     * @param bestN when "bestN"" selection is used this indicates the percentage of best individuals to use.
     * This is an inactive parameter if other types of selection are selected.
     * @param mutation the mutation strategy. One of "gaussian", "polynomial" or "uniform".
     * @param mutation the selection strategy. One of "roulette", "bestN".
     * @param mutation the crossover strategy. One of "exponential", "binomial" or "sbx"
     * @param int_dim the number of element in the chromosome to be treated as integers.
     *
     * @throws std::invalid_argument if \p cr not in [0,1), \p eta_c not in [1, 100), \p m not in [0,1], \p elitism < 1
     * \p mutation not one of "gaussian", "uniform" or "polynomial", \p selection not one of "roulette" or "bestN"
     * \p crossover not one of "exponential", "binomial", "sbx" or "single-point", if \p param_m is not in [0,1] and
     * \p mutation is not "polynomial" or \p mutation is not in [1,100] and \p mutation is polynomial.
     */
    sga(unsigned gen = 1u, double cr = .95, double eta_c = 10., double m = 0.02, double param_m = 0.5,
        unsigned elitism = 1u, double bestN = 0.2, std::string mutation = "gaussian",
        std::string selection = "roulette", std::string crossover = "exponential",
        vector_double::size_type int_dim = 0u, unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_m(m), m_param_m(param_m), m_elitism(elitism), m_bestN(bestN),
          m_mutation(mutation), m_selection(selection), m_crossover(crossover), m_int_dim(int_dim), m_e(seed),
          m_seed(seed), m_verbosity(0u) //, m_log()
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
        if (elitism < 1u) {
            pagmo_throw(std::invalid_argument, "elitism must be greater than zero");
        }
        if (!mutation.compare("gaussian") && !mutation.compare("uniform") && !mutation.compare("polynomial")) {
            pagmo_throw(
                std::invalid_argument,
                R"(The mutation type must either be "gaussian" or "uniform" or "polynomial": unknown type requested: )"
                    + mutation);
        }
        if (!selection.compare("roulette") && !selection.compare("bestN")) {
            pagmo_throw(std::invalid_argument,
                        R"(The selection type must either be "roulette" or "bestN": unknown type requested: )"
                            + selection);
        }
        if (!crossover.compare("exponential") && !crossover.compare("binomial") && !crossover.compare("sbx")
            && !crossover.compare("single-point")) {
            pagmo_throw(
                std::invalid_argument,
                R"(The crossover type must either be "exponential" or "binomial" or "sbx" or "single-point": unknown type requested: )"
                    + crossover);
        }
        // param_m represents the distribution index if polynomial mutation is selected
        if (mutation.compare("polynomial") && (param_m < 1. || param_m > 100.)) {
            pagmo_throw(
                std::invalid_argument,
                "Polynomial mutation was selected, the mutation parameter must be in [1, 100], while a value of "
                    + std::to_string(param_m) + " was detected");
        }
        // otherwise it represents a width
        if (!mutation.compare("polynomial") && (param_m < 0 || param_m > 1.)) {
            pagmo_throw(std::invalid_argument, "The mutation parameter must be in [0,1], while a value of "
                                                   + std::to_string(param_m) + " was detected");
        }
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
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        // m_log.clear();

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
        stream(ss, "\tElitism: ", m_elitism);
        stream(ss, "\n\tCrossover:");
        stream(ss, "\n\t\tType: " + m_crossover);
        stream(ss, "\n\t\tProbability: " + m_cr);
        if (m_crossover.compare("sbx")) stream(ss, "\n\t\tDistribution index: " + m_eta_c);
        stream(ss, "\n\tMutation:");
        stream(ss, "\n\t\tType: " + m_mutation);
        stream(ss, "\n\t\tProbability: " + m_m);
        if (m_mutation.compare("polynomial")) {
            stream(ss, "\n\t\tWidth: " + m_param_m);
        } else {
            stream(ss, "\n\t\tDistribution index: " + m_param_m);
        }
        stream(ss, "\n\tSelection:");
        stream(ss, "\n\t\tType: " + m_selection);
        if (m_selection.compare("bestN")) stream(ss, "\n\t\tBest pop fraction: " + m_bestN);
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
        ar(m_gen, m_cr, m_eta_c, m_m, m_param_m, m_elitism, m_bestN, m_mutation, m_selection, m_crossover, m_int_dim,
           m_e, m_seed, m_verbosity)
    }

    unsigned gen = 1u, double cr = .95, double eta_c = 10., double m = 0.02, double param_m = 0.5,
        unsigned elitism = 1u, double bestN = 0.2, std::string mutation = "gaussian",
        std::string selection = "roulette", std::string crossover = "exponential",
        vector_double::size_type int_dim = 0u, unsigned seed = pagmo::random_device::next())

private:
    unsigned m_gen;
    double m_cr;
    double m_eta_c;
    double m_m;
    double m_param_m;
    unsigned m_elitism;
    double m_bestN;
    std::string m_mutation;
    std::string m_selection;
    std::string m_crossover;
    vector_double::size_type m_int_dim;
    unsigned m_seed;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::sga)

#endif
