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

#ifndef PAGMO_ALGORITHMS_GWO_HPP
#define PAGMO_ALGORITHMS_GWO_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{
/// Grey Wolf Optimizer Algorithm
/**
 *
 * \image html GreyWolf.gif "One Grey Wolf" width=3cm
 * Grey Wolf Optimizer is an optimization algorithm based on the leadership hierarchy and hunting mechanism of
 * greywolves, proposed by Seyedali Mirjalilia, Seyed Mohammad Mirjalilib, Andrew Lewis in 2014.
 *
 * This algorithm is a classic example of a highly criticizable line of search that led in the first decades of
 * our millenia to the development of an entire zoo of metaphors inspiring optimzation heuristics. In our opinion they,
 * as is the case for the grey wolf optimizer, are often but small variations of already existing heuristics rebranded
 * with unnecessray and convoluted biological metaphors. In the case of GWO this is particularly evident as the position
 * update rule is shokingly trivial and can also be easily seen as a product of an evolutionary metaphor or a particle
 * swarm one. Such an update rule is also not particulary effective and results in a rather poor performance most of
 * times. Reading the original peer-reviewed paper, where the poor algoritmic perfromance is hidden by the
 * methodological flaws of the benchmark presented, one is left with a bitter opinion of the whole peer-review system.
 *
 * The implementation provided for PaGMO is based on the pseudo-code provided in the original Seyedali and Andrew (2014)
 * paper. pagmo::gwo is suitable for box-constrained single-objective continuous optimization.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. seealso::
 *
 *    https://www.sciencedirect.com/science/article/pii/S0965997813001853 for the paper that introduces Grey Wolf
 *    Optimizer and the pseudo-code
 *
 *    https://github.com/7ossam81/EvoloPy/blob/master/GWO.py for the Python implementation
 *
 * \endverbatim
 *
 */
class PAGMO_DLL_PUBLIC gwo
{
public:
    /// Single entry of the log (gen, alpha, beta, delta)
    typedef std::tuple<unsigned, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs a Grey Wolf Optimizer
     *
     * @param gen number of generations.
     *
     * @param seed seed used by the internal random number generator (default is random)
     *
     */

    gwo(unsigned gen = 1u, unsigned seed = pagmo::random_device::next());

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
     * Example (verbosity 1):
     * @code{.unparsed}
     *  Gen:      Alpha:          Beta:         Delta:
     *   1         5.7861        12.7206        19.6594
     *   2       0.404838        4.60328        9.51591
     *   3      0.0609075        3.83717        4.30162
     *   4      0.0609075       0.830047        1.77049
     *   5       0.040997        0.12541       0.196164

     * @endcode
     * Gen, is the generation number, Alpha is the fitness score for alpha, Beta is the fitness
     * score for beta, delta is the fitness score for delta
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

    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned get_gen() const
    {
        return m_gen;
    }

    /// Algorithm name
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "GWO: Grey Wolf Optimizer";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a gwo::log_line_type containing: gen, alpha, beta, delta as described
     * in gwo::set_verbosity
     * @return an <tt>std::vector</tt> of gwo::log_line_type containing the logged values gen, alpha, beta, delta
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
    unsigned m_seed;
    mutable detail::random_engine_type m_e;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::gwo)

#endif
