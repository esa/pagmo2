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

#include <boost/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
namespace pagmo
{
// Non-dominated Sorting Particle Swarm Optimizer (NSPSO)
// Non-dominated Sorting Particle Swarm Optimizer (NSPSO) is a modified version of PSO for multi-objective optimization.
// It extends the basic ideas of PSO by making a better use of personal bests and offspring for non-dominated
// comparison. In order to increase the diversity of the pareto front it is possible to choose between 3 different
// niching methods: crowding distance, niche count and maxmin.
// See:
// -"Xiaodong Li - A Non-dominated Sorting Particle Swarm Optimizer for Multiobjective Optimization"
// -"Xiaodong Li - Better Spread and Convergence: Particle Swarm Multiobjective Optimization Using the Maximin Fitness
// Function"
// -"Carlos M. Fonseca, Peter J. Fleming - Genetic Algorithms for Multiobjective Optimization: Formulation, Discussion
// and Generalization"

class PAGMO_DLL_PUBLIC nspso
{
public:
    // Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
    // The log
    typedef std::vector<log_line_type> log_type;

    // Constructor
    // Constructs the NSPSO user defined algorithm (multi objective PSO).
    nspso(unsigned gen = 1u, double omega = 0.6, double c1 = 2.0, double c2 = 2.0, double chi = 1.0,
          double v_coeff = 0.5, unsigned leader_selection_range = 60u,
          std::string diversity_mechanism = "crowding distance", bool memory = false,
          unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

    // Sets the seed
    void set_seed(unsigned);

    // Gets the
    unsigned get_seed() const
    {
        return m_seed;
    }

    // Sets the algorithm verbosity
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    }

    // Gets the verbosity level
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }

    // Gets the generations
    unsigned get_gen() const
    {
        return m_gen;
    }

    // Sets the bfe
    void set_bfe(const bfe &b);

    // Algorithm name
    std::string get_name() const
    {
        return "NSPSO";
    }

    // Extra info
    std::string get_extra_info() const;

    // Get log
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

    unsigned m_gen;
    double m_omega;
    double m_c1;
    double m_c2;
    double m_chi;
    double m_v_coeff;
    unsigned m_leader_selection_range;
    std::string m_diversity_mechanism;
    bool m_memory;
    // paricles' velocities
    mutable std::vector<vector_double> m_velocity;
    mutable std::vector<vector_double> m_best_fit;
    mutable std::vector<vector_double> m_best_dvs;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    boost::optional<bfe> m_bfe;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nspso)

#endif
