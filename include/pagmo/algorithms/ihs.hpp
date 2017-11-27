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

#ifndef PAGMO_ALGORITHMS_IHS_HPP
#define PAGMO_ALGORITHMS_IHS_HPP

#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{

/// Imporved Harmony Search
/**
 * \image html ihs.gif
 *
 * Harmony search (HS) is a metaheuristic algorithm said to mimick the improvisation process of musicians.
 * In the metaphor, each musician (i.e., each variable) plays (i.e., generates) a note (i.e., a value)
 * for finding a best harmony (i.e., the global optimum) all together.
 *
 * The algorithm has been heavily criticized in the scientific literature, not for its performances
 * rather for the use of a metaphor that does not add anything to existing ones. The HS 
 * algorithm essentially applies mutation and crossover operators to a background population and as such
 * should have been developed in the context of Evolutionary Strategies or Genetic Algorithms and studied
 * in that context. The use of the musicians metaphor only obscures its internal functioning
 * making theoretical results from ES and GA erroneously seem as unapplicable to HS. 
 *
 * This code implements the so-called improved harmony search algorithm (IHS), in which the probability
 * of picking the variables from the decision vector and the amount of mutation to which they are subject
 * vary (respectively linearly and exponentially) at each call of the evolve() method.
 *
 * In this algorithm the number of fitness function evaluations is equal to the number of iterations.
 * All the individuals in the input population participate in the evolution. A new individual is generated
 * at every iteration, substituting the current worst individual of the population if better.
 *
 * This algorithm is suitable for continuous, constrained, mixed-integer and multi-objective optimisation.
 *
 * \verbatim embed:rst:leading-asterisk
 *
 * .. warning::
 *
 *    This algorithm is suitable for continuous, constrained, mixed-integer and
 *    multi-objective optimisation.
 *
 *
 * .. seealso::
 *
 *    http://en.wikipedia.org/wiki/Harmony_search for an introduction on harmony search.
 *
 * .. seealso::
 *
 *    http://dx.doi.org/10.1016/j.amc.2006.11.033 for the paper that introduces and explains improved harmony search.
 *
 * \endverbatim
 */
class ihs
{
public:
    /// Single entry of the log (gen, fevals, best, improvement, mutations)
    // typedef std::tuple<unsigned int, unsigned long long, double, double, vector_double::size_type> log_line_type;
    /// The log
    // typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs ihs
     *
     * @param gen Number of generations to consider. Each generation will compute the objective function once.
     * @param phmcr probability of choosing from memory (similar to a crossover probability)
     * @param ppar_min minimum pitch adjustment rate. (similar to a mutation rate)
     * @param ppar_max maximum pitch adjustment rate. (similar to a mutation rate)
     * @param bw_min minimum distance bandwidth. (similar to a mutation width)
     * @param bw_max maximum distance bandwidth. (similar to a mutation width)
     * @param seed seed used by the internal random number generator

     * @throws value_error if phmcr is not in the ]0,1[ interval, ppar min/max are not in the ]0,1[
     * interval, min/max quantities are less than/greater than max/min quantities, bw_min is negative.
     */
    ihs(unsigned gen = 1u, double phmcr = 0.85, double ppar_min = 0.35, double ppar_max = 0.99, double bw_min = 1E-5,
        double bw_max = 1., unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_phmcr(phmcr), m_ppar_min(ppar_min), m_ppar_max(ppar_max), m_bw_min(bw_min), m_bw_max(bw_max),
          m_e(seed), m_seed(seed), m_verbosity(0u) //, m_log()
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

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained
     */
    population evolve(population pop) const
    {
        // We store some useful properties
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        const auto dim = prob.get_nx();       // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto fevals0 = prob.get_fevals(); // disount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        if (!pop.size()) {
            pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        // m_log.clear();
        
        return pop;
    };
    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - 1: will only print and log when the population is improved
     * - >1: will print and log one line each \p level generations.
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:        Fevals:          Best:   Improvement:     Mutations:
     * 632           3797        1464.31        51.0203              1
     * 633           3803        1463.23        13.4503              1
     * 635           3815        1562.02        31.0434              3
     * 667           4007         1481.6        24.1889              1
     * 668           4013        1487.34        73.2677              2
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness
     * function currently in the population, Improvement is the improvement made by the last mutation and Mutations
     * is the number of mutated components of the decision vector
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
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned int seed)
    {
        m_e.seed(seed);
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
    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "Improved Harmony Search";
    }
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
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

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a ihs::log_line_type containing: Gen, Fevals, Best, Improvement, Mutations as described
     * in ihs::set_verbosity
     * @return an <tt>std::vector</tt> of ihs::log_line_type containing the logged values Gen, Fevals, Best,
     * Improvement, Mutations
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
        ar(m_gen, m_phmcr, m_ppar_min, m_ppar_max, m_bw_min, m_bw_max, m_e, m_seed, m_verbosity);
    }

private:
    unsigned m_gen;
    double m_phmcr;
    double m_ppar_min;
    double m_ppar_max;
    double m_bw_min;
    double m_bw_max;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    // mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::ihs)

#endif
