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

#ifndef PAGMO_ALGORITHMS_NSPSO_HPP
#define PAGMO_ALGORITHMS_NSPSO_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{
//\image html nspso.jpg "" width=3cm
/// Non-dominated Sorting Particle Swarm Optimizer (NSPSO)
/**
 *
 * Non-dominated Sorting Particle Swarm Optimizer (NSPSO) is a modified version of PSO for multi-objective optimization.
 * It extends the basic ideas of PSO by making a better use of personal bests and offspring for non-dominated
 *comparison. In order to increase the diversity of the pareto front it is possible to choose between 3 different
 *niching methods: crowding distance, niche count and maxmin.
 *
 * See: "Xiaodong Li - A Non-dominated Sorting Particle Swarm Optimizer for Multiobjective Optimization"
 * See: "Xiaodong Li - Better Spread and Convergence: Particle Swarm Multiobjective Optimization Using the Maximin
 *Fitness Function" See: "Carlos M. Fonseca, Peter J. Fleming - Genetic Algorithms for Multiobjective Optimization:
 *Formulation, Discussion and Generalization"
 **/

class PAGMO_DLL_PUBLIC nspso
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs the NSPSO user defined algorithm (multi objective PSO).
     *
     * @param[in] gen Number of generations to evolve.
     * @param[in] min_w Minimum particles' inertia weight (the inertia weight is decreased throughout the run between
     * max_w and min_w).
     * @param[in] max_w Maximum particles' inertia weight (the inertia weight is decreased throughout the run between
     * max_w and min_w).
     * @param[in] c1 Magnitude of the force, applied to the particle's velocity, in the direction of its previous best
     * position.
     * @param[in] c2 Magnitude of the force, applied to the particle's velocity, in the direction of its global best
     * (i.e., leader).
     * @param[in] chi Velocity scaling factor.
     * @param[in] v_coeff Velocity coefficient (determining the maximum allowed particle velocity).
     * @param[in] leader_selection_range The leader of each particle is selected among the best
     * leader_selection_range%individuals.
     * @param[in] diversity_mechanism The diversity mechanism used to mantain diversity on the Pareto front.
     * @param seed seed used by the internal random number generator (default is random)
     * @throws std::invalid_argument if .
     */
    nspso(unsigned gen = 1u, double min_w = 0.95, double max_w = 10., double c1 = 0.01, double c2 = 0.5,
          double chi = 0.5, double v_coeff = 0.5, unsigned leader_selection_range = 2u,
          std::string diversity_mechanism = "crowding distance", unsigned seed = pagmo::random_device::next());

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

    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
        return "NSPSO";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a nspso::log_line_type containing: Gen, Fevals, ideal_point
     * as described in nspso::set_verbosity
     * @return an <tt>std::vector</tt> of nspso::log_line_type containing the logged values Gen, Fevals,
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
    PAGMO_DLL_LOCAL double minfit(vector_double::size_type, vector_double::size_type,
                                  const std::vector<vector_double> &) const;
    PAGMO_DLL_LOCAL void compute_maxmin(vector_double &, const std::vector<vector_double> &) const;
    PAGMO_DLL_LOCAL void compute_niche_count(std::vector<vector_double::size_type> &,
                                             const std::vector<vector_double> &, double) const;
    PAGMO_DLL_LOCAL double euclidian_distance(const vector_double &, const vector_double &) const;

    struct nspso_individual {
        vector_double cur_x;
        vector_double best_x;
        vector_double cur_v;
        vector_double cur_f;
        vector_double best_f;
    };

    unsigned m_gen;
    double m_min_w;
    double m_max_w;
    double m_c1;
    double m_c2;
    double m_chi;
    double m_v_coeff;
    unsigned m_leader_selection_range;
    std::string m_diversity_mechanism;
    // paricles' velocities
    mutable std::vector<vector_double> m_velocity;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nspso)

#endif
