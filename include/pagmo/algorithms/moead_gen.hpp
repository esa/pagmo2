/* Copyright 2017-2021 PaGMO development team

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

#ifndef PAGMO_ALGORITHMS_MOEAD_GEN_HPP
#define PAGMO_ALGORITHMS_MOEAD_GEN_HPP

#include <string>
#include <tuple>
#include <vector>

#include <boost/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>

namespace pagmo
{
/// Generational Multi Objective Evolutionary Algorithms by Decomposition (the DE variant)
/**
 * \image html moead.png "Solving by decomposition" width=3cm
 *
 * Here the MOEA/D-DE algorithm is adapted into a version which only updates the population on a generational basis.
 * Rather than constructing a new candidate and evaluating its fitness one-by-one across the population, the new candidate population is created 
 * at once at the beginning of a generation, and then the fitness of each candidate is evaluated either in parallel with a bfe or in series.
 * While this sacrifices the immediate sensitivity of the original MOEA/D-DE to changes in the population, it does allow for easy parallelizability.
 * This variant can employ a bfe to make several fitness evaluations simultaneously, and should be used for expensive fitness functions.
 *
 * See the MOEA/D-DE algorithm documentation for more details on the generic algorithm.
 */
class PAGMO_DLL_PUBLIC moead_gen
{
public:
    /// Single entry of the log (gen, fevals, adf, ideal_point)
    typedef std::tuple<unsigned, unsigned long long, double, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs MOEA/D-DE
     *
     * @param gen number of generations
     * @param weight_generation method used to generate the weights, one of "grid", "low discrepancy" or "random"
     * @param decomposition decomposition method: one of "weighted", "tchebycheff" or "bi"
     * @param neighbours size of the weight's neighborhood
     * @param CR crossover parameter in the Differential Evolution operator
     * @param F parameter for the Differential Evolution operator
     * @param eta_m distribution index used by the polynomial mutation
     * @param realb chance that the neighbourhood is considered at each generation, rather than the whole population
     * (only if preserve_diversity is true)
     * @param limit maximum number of copies reinserted in the population  (only if m_preserve_diversity is true)
     * @param preserve_diversity when true activates the two diversity preservation mechanisms described in Li, Hui,
     * and Qingfu Zhang paper
     * @param seed seed used by the internal random number generator (default is random)
     * @throws value_error if gen is negative, weight_generation is not one of the allowed types, realb,cr or f are not
     * in [1.0] or m_eta is < 0, if neighbours is <2
     */
    moead_gen(unsigned gen = 1u, std::string weight_generation = "grid", std::string decomposition = "tchebycheff",
          population::size_type neighbours = 20u, double CR = 1.0, double F = 0.5, double eta_m = 20.,
          double realb = 0.9, unsigned limit = 2u, bool preserve_diversity = true,
          unsigned seed = pagmo::random_device::next());

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

    // Sets the bfe
    void set_bfe(const bfe &b);

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
        return "Generational MOEAD: MOEA/D - DE";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a moead_gen::log_line_type containing: Gen, Fevals, ADR, ideal_point
     * as described in moead_gen::set_verbosity
     * @return an <tt>std::vector</tt> of moead_gen::log_line_type containing the logged values Gen, Fevals, ADR,
     * ideal_point
     */
    const log_type &get_log() const
    {
        return m_log;
    }

private:
    // Object serialization
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned);

    PAGMO_DLL_LOCAL std::vector<population::size_type>
    select_parents(population::size_type n, const std::vector<std::vector<population::size_type>> &neigh_idx,
                   bool whole_population) const;

    unsigned m_gen;
    std::string m_weight_generation;
    std::string m_decomposition;
    population::size_type m_neighbours;
    double m_CR;
    double m_F;
    double m_eta_m;
    double m_realb;
    unsigned m_limit;
    bool m_preserve_diversity;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    boost::optional<bfe> m_bfe;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::moead_gen)

#endif
