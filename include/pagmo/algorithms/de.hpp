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

#ifndef PAGMO_ALGORITHMS_DE_HPP
#define PAGMO_ALGORITHMS_DE_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{

/// Differential Evolution Algorithm
/**
 * \image html de.png "Differential Evolution block diagram."
 *
 * Differential Evolution is an heuristic optimizer developed by Rainer Storn and Kenneth Price.
 *
 * ''A breakthrough happened, when Ken came up with the idea of using vector differences for perturbing
 * the vector population. Since this seminal idea a lively discussion between Ken and Rainer and endless
 * ruminations and computer simulations on both parts yielded many substantial improvements which
 * make DE the versatile and robust tool it is today'' (from the official web pages....)
 *
 * The implementation provided for PaGMO is based on the code provided in the official
 * DE web site. pagmo::de is suitable for box-constrained single-objective continuous optimization.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The feasibility correction, that is the correction applied to an allele when some mutation puts it outside
 *    the allowed box-bounds, is here done by creating a random number in the bounds.
 *
 * .. seealso::
 *
 *    The official DE web site: http://www1.icsi.berkeley.edu/~storn/code.html
 *
 *    The paper that introduces Differential Evolution https://link.springer.com/article/10.1023%2FA%3A1008202821328
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC de
{
public:
    /// Single entry of the log (gen, fevals, best, dx, df)
    typedef std::tuple<unsigned, unsigned long long, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Constructs de
     *
     * The following variants (mutation variants) are available to create a new candidate individual:
     * @code{.unparsed}
     * 1 - best/1/exp                               2. - rand/1/exp
     * 3 - rand-to-best/1/exp                       4. - best/2/exp
     * 5 - rand/2/exp                               6. - best/1/bin
     * 7 - rand/1/bin                               8. - rand-to-best/1/bin
     * 9 - best/2/bin                               10. - rand/2/bin
     * @endcode
     *
     * @param gen number of generations.
     * @param F weight coefficient (dafault value is 0.8)
     * @param CR crossover probability (dafault value is 0.9)
     * @param variant mutation variant (dafault variant is 2: /rand/1/exp)
     * @param ftol stopping criteria on the f tolerance (default is 1e-6)
     * @param xtol stopping criteria on the x tolerance (default is 1e-6)
     * @param seed seed used by the internal random number generator (default is random)

     * @throws std::invalid_argument if F, CR are not in [0,1]
     * @throws std::invalid_argument if variant is not one of 1 .. 10
     */
    de(unsigned gen = 1u, double F = 0.8, double CR = 0.9, unsigned variant = 2u, double ftol = 1e-6,
       double xtol = 1e-6, unsigned seed = pagmo::random_device::next());

    // Evolve.
    population evolve(population) const;
    // Set the seed.
    void set_seed(unsigned);
    /// Get the seed
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
     * Gen:        Fevals:          Best:            dx:            df:
     * 5001         100020    3.62028e-05      0.0396687      0.0002866
     * 5101         102020    1.16784e-05      0.0473027    0.000249057
     * 5201         104020    1.07883e-05      0.0455471    0.000243651
     * 5301         106020    6.05099e-06      0.0268876    0.000103512
     * 5401         108020    3.60664e-06      0.0230468    5.78161e-05
     * 5501         110020     1.7188e-06      0.0141655    2.25688e-05
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness
     * function currently in the population, dx is the population flatness evaluated as the distance between
     * the decisions vector of the best and of the worst individual, df is the population flatness evaluated
     * as the distance between the fitness of the best and of the worst individual.
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    };
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
        return "DE: Differential Evolution";
    }
    // Extra info.
    std::string get_extra_info() const;
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a de::log_line_type containing: Gen, Fevals, Best, dx, df as described
     * in de::set_verbosity
     * @return an <tt>std::vector</tt> of de::log_line_type containing the logged values Gen, Fevals, Best, dx, df
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
    double m_F;
    double m_CR;
    unsigned m_variant;
    double m_Ftol;
    double m_xtol;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::de)

#endif
