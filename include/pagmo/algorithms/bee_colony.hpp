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

#ifndef PAGMO_ALGORITHMS_BEE_COLONY_HPP
#define PAGMO_ALGORITHMS_BEE_COLONY_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

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
 * See: https://abc.erciyes.edu.tr/ for the official ABC web site
 *
 * See: https://link.springer.com/article/10.1007/s10898-007-9149-x for the paper that introduces Artificial Bee Colony
 *
 * See: https://www.sciencedirect.com/science/article/pii/S0020025514008378 for the pseudo-code
 */
class PAGMO_DLL_PUBLIC bee_colony
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
    bee_colony(unsigned gen = 1u, unsigned limit = 20u, unsigned seed = pagmo::random_device::next());

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

    // Extra info
    std::string get_extra_info() const;

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

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    unsigned m_gen;
    unsigned m_limit;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::bee_colony)

#endif
