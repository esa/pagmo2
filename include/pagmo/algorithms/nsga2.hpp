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

#ifndef PAGMO_ALGORITHMS_NSGA2_HPP
#define PAGMO_ALGORITHMS_NSGA2_HPP

#include <string>
#include <tuple>
#include <vector>

#include <boost/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{
/// Nondominated Sorting genetic algorithm II (NSGA-II)
/**
 * \image html nsga2.jpg "The NSGA-II flowchart" width=3cm

 * NSGA-II is a solid multi-objective algorithm, widely used in many real-world applications.
 * While today it can be considered as an outdated approach, nsga2 has still a great value, if not
 * as a solid benchmark to test against.
 * NSGA-II genererates offsprings using a specific type of crossover and mutation and then selects the next
 * generation according to nondominated-sorting and crowding distance comparison.
 *
 * The version implemented in pagmo can be applied to box-bounded multiple-objective optimization. It also
 * deals with integer chromosomes treating the last \p int_dim entries in the decision vector as integers.
 *
 * See:  Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic
 * algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
 */
class PAGMO_DLL_PUBLIC nsga2
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs the NSGA II user defined algorithm.
     *
     * @param[in] gen Number of generations to evolve.
     * @param[in] cr Crossover probability.
     * @param[in] eta_c Distribution index for crossover.
     * @param[in] m Mutation probability.
     * @param[in] eta_m Distribution index for mutation.
     * @param seed seed used by the internal random number generator (default is random)
     * @throws std::invalid_argument if \p cr is not \f$ \in [0,1[\f$, \p m is not \f$ \in [0,1]\f$, \p eta_c is not in
     * [1,100[ or \p eta_m is not in [1,100[.
     */
    nsga2(unsigned gen = 1u, double cr = 0.95, double eta_c = 10., double m = 0.01, double eta_m = 50.,
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
     * Gen:        Fevals:        ideal1:        ideal2:        ideal3:
     *   1              0      0.0257554       0.267768       0.974592
     *   2             52      0.0257554       0.267768       0.908174
     *   3            104      0.0257554       0.124483       0.822804
     *   4            156      0.0130094       0.121889       0.650099
     *   5            208     0.00182705      0.0987425       0.650099
     *   6            260      0.0018169      0.0873995       0.509662
     *   7            312     0.00154273      0.0873995       0.492973
     *   8            364     0.00154273      0.0873995       0.471251
     *   9            416    0.000379582      0.0873995       0.471251
     *  10            468    0.000336743      0.0855247       0.432144
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used. The ideal point of the current
     * population follows cropped to its 5th component.
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
        return "NSGA-II:";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a nsga2::log_line_type containing: Gen, Fevals, ideal_point
     * as described in nsga2::set_verbosity
     * @return an <tt>std::vector</tt> of nsga2::log_line_type containing the logged values Gen, Fevals,
     * ideal_point
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL vector_double::size_type
    tournament_selection(vector_double::size_type idx1, vector_double::size_type idx2,
                         const std::vector<vector_double::size_type> &non_domination_rank,
                         std::vector<double> &crowding_d) const;
    PAGMO_DLL_LOCAL void crossover(vector_double &child1, vector_double &child2, vector_double::size_type parent1_idx,
                                   vector_double::size_type parent2_idx, const pagmo::population &pop) const;
    PAGMO_DLL_LOCAL void mutate(vector_double &child, const pagmo::population &pop) const;

    unsigned m_gen;
    double m_cr;
    double m_eta_c;
    double m_m;
    double m_eta_m;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    boost::optional<bfe> m_bfe;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nsga2)

#endif
