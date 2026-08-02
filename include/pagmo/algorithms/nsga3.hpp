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

/*
 *  Implements the NSGA-III multi-objective evolutionary algorithm
 *  as described in http://dx.doi.org/10.1109/TEVC.2013.2281535
 *
 *  Paul Slavin <paul.slavin@manchester.ac.uk>
 */

#ifndef PAGMO_ALGORITHMS_NSGA3_HPP
#define PAGMO_ALGORITHMS_NSGA3_HPP

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include <boost/optional.hpp>

#include <pagmo/algorithm.hpp>         // PAGMO_S11N_ALGORITHM_EXPORT_KEY
#include <pagmo/bfe.hpp>               // bfe
#include <pagmo/detail/visibility.hpp> // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>        // population
#include <pagmo/rng.hpp>               // random_device, random_engine_type
#include <pagmo/s11n.hpp>              // detail::archive
#include <pagmo/types.hpp>             // vector_double

namespace pagmo
{

/// Nondominated Sorting genetic algorithm III (NSGA-III)
/**
 * NSGA-III is a many-objective evolutionary algorithm. It keeps the non-dominated
 * sorting of NSGA-II but replaces the crowding distance, whose ability to
 * discriminate degrades quickly as the number of objectives grows, with a set of
 * structured reference directions. At every generation the objectives are
 * adaptively normalized, each individual is associated with the reference direction
 * whose ray it lies closest to, and the individuals which fill the least crowded
 * directions are preferred. Diversity is therefore maintained explicitly rather than
 * as a by-product of a density estimate, which is what allows the algorithm to scale
 * to a large number of objectives.
 *
 * The version implemented in pagmo can be applied to box-bounded, unconstrained,
 * deterministic multiple-objective optimization. Like nsga2 it also deals with
 * integer chromosomes, treating the last \p nix entries of the decision vector as
 * integers.
 *
 * **Reference directions.** The directions are placed on the unit simplex by the
 * systematic approach of Das and Dennis: with \p divisions divisions along each
 * objective, an \f$M\f$-objective problem receives
 * \f$H = \binom{M + p - 1}{p}\f$ directions. That count grows quickly with \f$M\f$,
 * so Deb and Jain add a second, inner layer for many-objective problems: a Das and
 * Dennis layer built with \p divisions_inner divisions whose every coordinate is
 * then mapped through
 * \f[
 *     c \rightarrow \frac{c + 1/M}{2},
 * \f]
 * which shrinks it by one half about the centroid of the simplex while keeping it on
 * that simplex. Setting \p divisions_inner to zero uses the outer layer alone. The
 * two layers are concatenated deterministically, outer first, and a direction of the
 * inner layer coinciding with one already present is dropped. The settings of Table I
 * of the original paper are reproduced by:
 *
 * <table>
 * <caption>Deb and Jain, Table I</caption>
 * <tr><th>Objectives<th>divisions<th>divisions_inner<th>Directions<th>Population
 * <tr><td>3 <td>12<td>0<td> 91<td> 92
 * <tr><td>5 <td> 6<td>0<td>210<td>212
 * <tr><td>8 <td> 3<td>2<td>156<td>156
 * <tr><td>10<td> 3<td>2<td>275<td>276
 * <tr><td>15<td> 2<td>1<td>135<td>136
 * </table>
 *
 * A configuration whose direction count is too large to be built is rejected before
 * anything is allocated, rather than exhausting memory: eight objectives with
 * \p divisions set to 8, for instance, would already need 5040 directions, which is
 * the situation the inner layer exists to avoid.
 *
 * **Population requirements.** The population size must be at least 5 and a multiple
 * of 4, as for nsga2, and it must be at least as large as the number of reference
 * directions. Equality is explicitly permitted: the eight-objective row of Table I
 * above uses a population of exactly 156 for 156 directions.
 *
 * **Memory.** Section IV-C of the original paper takes the ideal point over the
 * selected sets of every generation so far, and builds the normalizing hyperplane
 * from the extreme points ever found since the start of the run, while Algorithm 2
 * is written in terms of the current generation alone. Both behaviours are
 * available: with \p use_memory set to true the ideal point and the extreme points
 * are retained across generations, and with it left false they are recomputed from
 * scratch every generation. The retained extreme points are stored in the original
 * objective coordinates, so that they remain meaningful as the ideal point moves.
 *
 * **Deviations from Deb and Jain.** The following are deliberate and are the only
 * ones:
 * - Mating selection is selectable through \p random_mating. The default, true,
 *   is the behaviour of the original paper, whose Section IV-F states that no
 *   explicit selection operator is used and that parents are picked at random.
 *   Setting it to false instead holds the binary tournament on non-domination rank
 *   and crowding distance which is the pagmo convention established by nsga2; that
 *   is a materially different mechanism and is not what the paper describes. The
 *   tournament can help noticeably on multimodal problems, where the absence of any
 *   selection pressure at mating slows convergence: on DTLZ1 with three objectives
 *   it reached a p-distance below 0.08 across four seeds, against up to 1.08 for the
 *   random pairing. On the unimodal DTLZ2 the two are indistinguishable.
 * - \p use_memory defaults to false, which follows Algorithm 2 literally rather
 *   than the running quantities of Section IV-C.
 * - The default mutation probability is a constant, whereas Table II of the paper
 *   recommends \f$1/n\f$ for a chromosome of length \f$n\f$; that value depends on
 *   the problem and so cannot be a default. Passing it explicitly is advisable.
 *
 * This implementation is based on the work of Paul Slavin in
 * <a href="https://github.com/esa/pagmo2/pull/569">pagmo2 pull request #569</a>.
 *
 * See: Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
 * Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I:
 * Solving Problems With Box Constraints. IEEE Transactions on Evolutionary
 * Computation, 18(4), 577-601. https://doi.org/10.1109/TEVC.2013.2281535
 */
class PAGMO_DLL_PUBLIC nsga3
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs the NSGA-III user defined algorithm. The defaults for the genetic
     * operators are those of Table II of Deb and Jain, and the default reference
     * direction set is the single layer with 12 divisions of their Table I, which
     * gives 91 directions for a three-objective problem.
     *
     * @param gen number of generations to evolve.
     * @param cr crossover probability.
     * @param eta_c distribution index for crossover.
     * @param mut mutation probability.
     * @param eta_mut distribution index for mutation.
     * @param divisions number of divisions of the outer layer of reference
     * directions along each objective.
     * @param divisions_inner number of divisions of the inner layer of reference
     * directions; zero disables the inner layer.
     * @param random_mating if true, mating parents are picked at random as in
     * Section IV-F of the original paper; if false, they are picked by binary
     * tournament on non-domination rank and crowding distance, as in nsga2.
     * @param seed seed used by the internal random number generator (default is random).
     * @param use_memory if true, the ideal point and the extreme points are retained
     * across generations, as described in Section IV-C of the original paper.
     *
     * @throws std::invalid_argument if \p cr or \p mut is not finite or not in
     * \f$[0,1]\f$, if \p eta_c or \p eta_mut is not finite or not in \f$[1,100]\f$,
     * if \p divisions is zero, or if \p divisions_inner exceeds \p divisions.
     */
    nsga3(unsigned gen = 1u, double cr = 1.0, double eta_c = 30.0, double mut = 0.10, double eta_mut = 20.0,
          std::size_t divisions = 12u, std::size_t divisions_inner = 0u, bool random_mating = true,
          unsigned seed = pagmo::random_device::next(), bool use_memory = false);

    // Algorithm evolve method
    population evolve(population) const;

    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned seed)
    {
        m_reng.seed(seed);
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
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:        Fevals:        ideal1:        ideal2:        ideal3:
     *   1              0       0.113086       0.153994      0.0682423
     *   2             92       0.113086       0.153994      0.0682423
     *   3            184      0.0866138       0.107934      0.0682423
     *   4            276      0.0866138      0.0917604      0.0682423
     *   5            368      0.0361252      0.0917604      0.0577711
     * @endcode
     * Gen is the generation number, Fevals the number of function evaluations used.
     * The ideal point of the current population follows, cropped to its 5th component.
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

    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
        return "NSGA-III: Non-dominated Sorting Genetic Algorithm III";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each
     * element of the returned <tt>std::vector</tt> is a nsga3::log_line_type
     * containing: Gen, Fevals, ideal_point as described in nsga3::set_verbosity
     *
     * @return an <tt>std::vector</tt> of nsga3::log_line_type containing the logged
     * values Gen, Fevals, ideal_point
     */
    const log_type &get_log() const
    {
        return m_log;
    }

private:
    /*  State retained across generations when memory is enabled.
     *  Both members are expressed in the *original* objective coordinates,
     *  so that they remain meaningful as the ideal point moves.
     */
    struct nsga3_memory {
        std::vector<std::vector<double>> v_extreme;
        std::vector<double> v_ideal;
        template <typename Archive>
        void serialize(Archive &ar, unsigned)
        {
            detail::archive(ar, v_extreme, v_ideal);
        }
    };

    // Object serialization
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned);

    unsigned m_gen;
    double m_cr;                   // crossover probability
    double m_eta_c;                // distribution index for crossover
    double m_mut;                  // mutation probability
    double m_eta_mut;              // distribution index for mutation
    std::size_t m_divisions;       // reference direction divisions, outer layer
    std::size_t m_divisions_inner; // reference direction divisions, inner layer; 0 disables it
    bool m_random_mating;          // pick mating parents at random rather than by tournament
    unsigned m_seed;               // seed for PRNG initialisation
    bool m_use_memory;             // preserve extremes and ideal across generations
    mutable nsga3_memory m_memory{};
    mutable detail::random_engine_type m_reng; // defaults to std::mt19937
    mutable log_type m_log;
    unsigned m_verbosity;
    boost::optional<bfe> m_bfe;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nsga3)

#endif
