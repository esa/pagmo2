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

#ifndef PAGMO_ALGORITHMS_SGA_HPP
#define PAGMO_ALGORITHMS_SGA_HPP

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{

namespace detail
{

enum class sga_selection { TOURNAMENT, TRUNCATED };

enum class sga_crossover { EXPONENTIAL, BINOMIAL, SINGLE, SBX };

enum class sga_mutation { GAUSSIAN, UNIFORM, POLYNOMIAL };

} // namespace detail

/// A Simple Genetic Algorithm
/**
 * \image html sga.jpg "The DNA Helix" width=3cm
 *
 * \verbatim embed:rst:leading-asterisk
 * .. versionadded:: 2.2
 * \endverbatim
 *
 * Approximately during the same decades as Evolutionary Strategies (see pagmo::sea) were studied, a different group
 * led by John Holland, and later by his student David Goldberg, introduced and studied an algorithmic framework called
 * "genetic algorithms" that were, essentially, leveraging on the same idea but introducing also crossover as a genetic
 * operator. This led to a few decades of confusion and discussions on what was an evolutionary startegy and what a
 * genetic algorithm and on whether the crossover was a useful operator or mutation only algorithms were to be
 * preferred.
 *
 * In pagmo we provide a rather classical implementation of a genetic algorithm, letting the user choose between
 * some selected crossover types, selection schemes and mutation types.
 *
 * The pseudo code of our version is:
 * @code{.unparsed}
 * > Start from a pagmo::population (pop) of dimension N
 * > while i < gen
 * > > Selection: create a new population (pop2) with N individuals selected from pop (with repetition allowed)
 * > > Crossover: create a new population (pop3) with N individuals obtained applying crossover to pop2
 * > > Mutation:  create a new population (pop4) with N individuals obtained applying mutation to pop3
 * > > Evaluate all new chromosomes in pop4
 * > > Reinsertion: set pop to contain the best N individuals taken from pop and pop4
 * @endcode
 *
 * The various blocks of pagmo genetic algorithm are listed below:
 *
 * *Selection*: two selection methods are provided: "tournament" and "truncated". Tournament selection works by
 * selecting each offspring as the one having the minimal fitness in a random group of size \p param_s. The truncated
 * selection, instead, works selecting the best \p param_s chromosomes in the entire population over and over.
 * We have deliberately not implemented the popular roulette wheel selection as we are of the opinion that such
 * a system does not generalize much being highly sensitive to the fitness scaling.
 *
 * *Crossover*: four different crossover schemes are provided: "single", "exponential", "binomial", "sbx". The
 * single point crossover, called "single", works selecting a random point in the parent chromosome and inserting the
 * partner chromosome thereafter. The exponential crossover is taken from the algorithm differential evolution,
 * implemented, in pagmo, as pagmo::de. It essentially selects a random point in the parent chromosome and inserts,
 * in each successive gene, the partner values with probability \p cr up to when it stops. The binomial crossover
 * inserts each gene from the partner with probability \p cr. The simulated binary crossover (called "sbx"), is taken
 * from the NSGA-II algorithm, implemented in pagmo as pagmo::nsga2, and makes use of an additional parameter called
 * distribution index \p eta_c.
 *
 * *Mutation*: three different mutations schemes are provided: "uniform", "gaussian" and "polynomial". Uniform mutation
 * simply randomly samples from the bounds. Gaussian muattion samples around each gene using a normal distribution
 * with standard deviation proportional to the \p m_param_m and the bounds width. The last scheme is the polynomial
 * mutation.
 *
 * *Reinsertion*: the only reinsertion strategy provided is what we call pure elitism. After each generation
 * all parents and children are put in the same pool and only the best are passed to the next generation.
 *
 * \verbatim embed:rst:leading-asterisk
 *
 * .. warning::
 *
 *    The algorithm is not suitable for multi-objective problems, nor for
 *    constrained optimization.
 *
 * .. note::
 *
 *    Most genetic operators use the lower and upper bound information. Hence, unbounded problems will produce undefined
 *    behaviours.
 *
 * .. seealso::
 *
 *    Oliveto, Pietro S., Jun He, and Xin Yao. "Time complexity of evolutionary algorithms for
 *    combinatorial optimization: A decade of results." International Journal of Automation and Computing
 *    4.3 (2007): 281-293.
 *
 * .. seealso::
 *
 *    http://www.scholarpedia.org/article/Evolution_strategies
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC sga
{
public:
    /// Single entry of the log (gen, fevals, best, improvement)
    typedef std::tuple<unsigned, unsigned long long, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs a simple genetic algorithm.
     *
     * @param gen number of generations.
     * @param cr crossover probability.
     * @param eta_c distribution index for "sbx" crossover. This is an inactive parameter if other types of crossovers
     * are selected.
     * @param m mutation probability.
     * @param param_m distribution index ("polynomial" mutation), gaussian width ("gaussian" mutation) or inactive
     * ("uniform" mutation)
     * @param param_s when "truncated" selection is used this indicates the number of best individuals to use. When
     * "tournament" selection is used this indicates the size of the tournament.
     * @param mutation the mutation strategy. One of "gaussian", "polynomial" or "uniform".
     * @param selection the selection strategy. One of "tournament", "truncated".
     * @param crossover the crossover strategy. One of "exponential", "binomial", "single" or "sbx"
     * @param seed seed used by the internal random number generator
     *
     * @throws std::invalid_argument if \p cr not in [0,1], \p eta_c not in [1, 100], \p m not in [0,1],
     * \p mutation not one of "gaussian", "uniform" or "polynomial", \p selection not one of "roulette" or "truncated"
     * \p crossover not one of "exponential", "binomial", "sbx" or "single", if \p param_m is not in [0,1] and
     * \p mutation is not "polynomial" or \p mutation is not in [1,100] and \p mutation is polynomial.
     */
    sga(unsigned gen = 1u, double cr = .90, double eta_c = 1., double m = 0.02, double param_m = 1.,
        unsigned param_s = 2u, std::string crossover = "exponential", std::string mutation = "polynomial",
        std::string selection = "tournament", unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

    // Sets the seed
    void set_seed(unsigned);

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
     * - 1: will only print and log when the population is improved
     * - >1: will print and log one line each \p level generations.
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:        Fevals:          Best:   Improvement:
     *    1             20        6605.75         415.95
     *    3             60        6189.79        500.359
     *    4             80        5689.44        477.663
     *    5            100        5211.77        218.231
     *    6            120        4993.54        421.684
     *    8            160        4571.86        246.532
     *   10            200        4325.33        166.685
     *   11            220        4158.64        340.382
     *   14            280        3818.26        294.232
     *   15            300        3524.03        55.0358
     *   16            320        3468.99        452.544
     *   17            340        3016.45        16.7273
     *   19            380        2999.72         150.68
     *   21            420        2849.04        301.156
     *   22            440        2547.88        1.25038
     *   23            460        2546.63        192.561
     *   25            500        2354.07        22.6248
     * @endcode
     * Gen is the generation number, Fevals the number of fitness evaluations , Best is the best fitness found,
     * Improvement is the improvement of the new population of offspring with respect to the parents.
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
        return "SGA: Genetic Algorithm";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a sga::log_line_type containing: Gen, Fevals, Current best, Best as
     * described in sga::set_verbosity().
     *
     * @return an <tt> std::vector</tt> of sga::log_line_type containing the logged values Gen, Fevals, Best
     * improvement
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL std::vector<vector_double::size_type> perform_selection(const std::vector<vector_double> &F) const;
    PAGMO_DLL_LOCAL void perform_crossover(std::vector<vector_double> &X,
                                           const std::pair<vector_double, vector_double> &bounds,
                                           vector_double::size_type dim_i) const;
    PAGMO_DLL_LOCAL void perform_mutation(std::vector<vector_double> &X,
                                          const std::pair<vector_double, vector_double> &bounds,
                                          vector_double::size_type dimi) const;
    PAGMO_DLL_LOCAL std::pair<vector_double, vector_double>
    sbx_crossover_impl(const vector_double &parent1, const vector_double &parent2,
                       const std::pair<vector_double, vector_double> &bounds, vector_double::size_type Di) const;

    unsigned m_gen;
    double m_cr;
    double m_eta_c;
    double m_m;
    double m_param_m;
    unsigned m_param_s;
    detail::sga_mutation m_mutation;
    detail::sga_selection m_selection;
    detail::sga_crossover m_crossover;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::sga)

#endif
