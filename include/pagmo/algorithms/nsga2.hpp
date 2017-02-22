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

#ifndef PAGMO_ALGORITHMS_NSGA2_HPP
#define PAGMO_ALGORITHMS_NSGA2_HPP

#include <algorithm> // std::shuffle, std::transform
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp" // needed for the cereal macro
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../problem.hpp"
#include "../problems/decompose.hpp"
#include "../rng.hpp"
#include "../utils/multi_objective.hpp" // crowding_distance, etc..

namespace pagmo
{
/// Nondominated Sorting genetic algorithm II (NSGA-II)
/**
 * \image html moead.png "Solving by decomposition" width=3cm

 * NSGA-II is a solid multi-objective algorithm, widely used in many real-world applications.
 * While today it can be considered as an outdated approach, nsga2 has still a great value, if not
 * as a solid benchmark to test against.
 * NSGA-II genererates offsprings using a specific type of crossover and mutation and then selects the next
 * generation according to nondominated-sorting and crowding distance comparison.
 *
 * The version implemented in pagmo can be applied to box-bounded multiple-objective optimization. It also
 * deals with integer chromosomes treating the last /p int_dim entries in the decision vector as integers.
 *
 * See:  Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic
 * algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
 */
class nsga2
{
public:
    /// Single entry of the log (gen, fevals, adf, ideal_point)
    /// typedef std::tuple<unsigned int, unsigned long long, double, vector_double> log_line_type;
    /// The log
    /// typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
    * Constructs the NSGA II user defined algorithm
    *
    * @param[in] gen Number of generations to evolve.
    * @param[in] cr Crossover probability.
    * @param[in] eta_c Distribution index for crossover.
    * @param[in] m Mutation probability.
    * @param[in] eta_m Distribution index for mutation.
    * @param int_dim the dimension of the decision vector to be considered as integer (the last int_dim entries will be
    * treated as integers when mutation and crossover are applied)
    * @throws std::invalid_argument if crossover probability is not \f$ \in [0,1[\f$, mutation probability or mutation
    * width is
    * not \f$ \in [0,1]\f$.
    */
    nsga2(unsigned int gen = 1u, double cr = 0.95, double eta_c = 10., double m = 0.01, double eta_m = 50.,
          vector_double::size_type int_dim = 0u, unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_m(m), m_eta_m(eta_m), m_int_dim(int_dim), m_e(seed), m_seed(seed),
          m_verbosity(0u) // , m_log()
    {
        if (cr >= 1. || cr < 0.) {
            pagmo_throw(std::invalid_argument, "The crossover probability must be in the [0,1[ range, while a value of "
                                                   + std::to_string(cr) + " was detected");
        }
        if (m < 0. || m > 1.) {
            pagmo_throw(std::invalid_argument, "The mutation probability must be in the [0,1] range, while a value of "
                                                   + std::to_string(cr) + " was detected");
        }
        if (eta_c < 1. || eta_c >= 100.) {
            pagmo_throw(std::invalid_argument,
                        "The distribution index for crossover must be in [1, 100], while a value of "
                            + std::to_string(eta_c) + " was detected");
        }
        if (eta_m < 1. || eta_m >= 100.) {
            pagmo_throw(std::invalid_argument,
                        "The distribution index for mutation must be in [1, 100], while a value of "
                            + std::to_string(eta_m) + " was detected");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for the requested number of generations.
     *
     * @param pop population to be evolved
     * @return evolved population
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto NP = pop.size();

        auto fevals0 = prob.get_fevals(); // discount for the fevals already made
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (!NP) {
            pagmo_throw(std::invalid_argument, get_name() + " cannot work on an empty population");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them.");
        }
        if (prob.get_nf() < 2u) {
            pagmo_throw(std::invalid_argument, "The number of objectives detected in the instance of '"
                                                   + prob.get_name() + "' is " + std::to_string(prob.get_nf()) + ". "
                                                   + get_name() + " necessitates a problem with multiple objectives");
        }
        if (m_int_dim > dim) {
            pagmo_throw(
                std::invalid_argument,
                "The problem dimension is: " + std::to_string(dim)
                    + ", while this instance of NSGA-II has been instantiated requesting an integer dimension of: "
                    + std::to_string(m_int_dim));
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------
        return pop;
    }
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    };
    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned int get_seed() const
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
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:        Fevals:           ADF:        ideal1:        ideal2:
     *   1              0        24.9576       0.117748        2.77748
     *   2             40        19.2461      0.0238826        2.51403
     *   3             80        12.4375      0.0238826        2.51403
     *   4            120        9.08406     0.00389182        2.51403
     *   5            160        7.10407       0.002065        2.51403
     *   6            200        6.11242     0.00205598        2.51403
     *   7            240        8.79749     0.00205598        2.25538
     *   8            280        7.23155    7.33914e-05        2.25538
     *   9            320        6.83249    7.33914e-05        2.25538
     *  10            360        6.55125    7.33914e-05        2.25538
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used. ADF is the Average
     * Decomposed Fitness, that is the average across all decomposed problem of the single objective decomposed fitness
     * along the corresponding direction. The ideal point of the current population follows cropped to its 5th
     * component.
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }
    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
        return "NSGA-II";
    }
    /// Extra informations
    /**
     * Returns extra information on the algorithm.
     *
     * @return an <tt> std::string </tt> containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tGenerations: ", m_gen);
        stream(ss, "\n\tCrossover probability: ", m_cr);
        stream(ss, "\n\tDistribution index for crossover: ", m_eta_c);
        stream(ss, "\n\\tMutation probability: ", m_m);
        stream(ss, "\n\tDistribution index for mutation: ", m_eta_m);
        stream(ss, "\n\tSize of the integer part: ", m_int_dim);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        return ss.str();
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt> std::vector </tt> is a moead::log_line_type containing: Gen, Fevals, ADR, ideal_point
     * as described in moead::set_verbosity
     * @return an <tt> std::vector </tt> of moead::log_line_type containing the logged values Gen, Fevals, ADR,
     * ideal_point
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
        ar(m_gen, m_cr, m_eta_c, m_m, m_eta_m, m_e, m_int_dim, m_seed, m_verbosity);
    }

private:
    unsigned int m_gen;
    double m_cr;
    double m_eta_c;
    double m_m;
    double m_eta_m;
    vector_double::size_type m_int_dim;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    // mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::nsga2)

#endif
