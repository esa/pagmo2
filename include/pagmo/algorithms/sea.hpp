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

#ifndef PAGMO_ALGORITHMS_SEA_HPP
#define PAGMO_ALGORITHMS_SEA_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{

/// (N+1)-ES Simple Evolutionary Algorithm
/**
 * \image html sea.png
 *
 * Evolutionary strategies date back to the mid 1960s when P. Bienert,
 * I. Rechenberg, and H.-P. Schwefel at the Technical University of Berlin, Germany,
 * developed the first bionics-inspired schemes for evolving optimal shapes of
 * minimal drag bodies in a wind tunnel using Darwin's evolution principle.
 *
 * This c++ class represents the simplest evolutionary strategy, where a
 * population of \f$ \lambda \f$ individuals at each generation produces one offspring
 * by mutating its best individual uniformly at random within the bounds. Should the
 * offspring be better than the worst individual in the population it will substitute it.
 *
 * \verbatim embed:rst:leading-asterisk
 *
 * .. warning::
 *
 *    The algorithm is not suitable for multi-objective problems, nor for
 *    constrained or stochastic optimization
 *
 * .. note::
 *
 *    The mutation is uniform within box-bounds. Hence, unbounded problems will produce undefined
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
class PAGMO_DLL_PUBLIC sea
{
public:
    /// Single entry of the log (gen, fevals, best, improvement, mutations)
    typedef std::tuple<unsigned, unsigned long long, double, double, vector_double::size_type> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs sea
     *
     * @param gen Number of generations to consider. Each generation will compute the objective function once
     * @param seed seed used by the internal random number generator
     */
    sea(unsigned gen = 1u, unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

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
    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "SEA: (N+1)-EA Simple Evolutionary Algorithm";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a sea::log_line_type containing: Gen, Fevals, Best, Improvement, Mutations as described
     * in sea::set_verbosity
     * @return an <tt>std::vector</tt> of sea::log_line_type containing the logged values Gen, Fevals, Best,
     * Improvement, Mutations
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    unsigned m_gen;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::sea)

#endif
