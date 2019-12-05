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

#ifndef PAGMO_ALGORITHMS_PSO_GEN_HPP
#define PAGMO_ALGORITHMS_PSO_GEN_HPP

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

/// Particle Swarm Optimization
/**
 *
 * As opposed to the main PSO algorithm implemented in pagmo, this version of Particle Swarm Optimization is
 * generational. In other words, the velocity is first calculated for all particles, then the position is updated.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    This PSO is suitable for stochastic optimization problems. The random seed is changed at the end of each
 *    generation.
 *
 * .. warning::
 *
 *    The algorithm is not suitable for multi-objective problems, nor for
 *    constrained optimization.
 *
 * .. seealso::
 *
 *    http://www.particleswarm.info/ for a repository of information related to PSO
 *
 * .. seealso::
 *
 *    :cpp:class:`pagmo::pso` for the implementation of non-generational PSO
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC pso_gen
{
public:
    /// Single entry of the log (Gen, Fevals, gbest, Mean Vel., Mean lbest, Avg. Dist.)
    typedef std::tuple<unsigned, unsigned long long, double, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Allows to specify in detail all the parameters of the algorithm.
     *
     * @param gen number of generations
     * @param omega particles' inertia weight, or alternatively, the constriction coefficient (definition depends on
     * the variant used)
     * @param eta1 magnitude of the force, applied to the particle's velocity, in the direction of its previous best
     * position
     * @param eta2 magnitude of the force, applied to the particle's velocity, in the direction of the best position
     * in its neighborhood
     * @param max_vel maximum allowed particle velocity (as a fraction of the box bounds)
     * @param variant algorithm variant to use (one of 1 .. 6)
     * @param neighb_type swarm topology to use (one of 1 .. 4) [gbest, lbest, Von Neumann, adaptive random]
     * @param neighb_param the neighbourhood parameter. If the lbest topology is selected (neighb_type=2),
     * it represents each particle's indegree (also outdegree) in the swarm topology. Particles have neighbours up
     * to a radius of k = neighb_param / 2 in the ring. If the Randomly-varying neighbourhood topology
     * is selected (neighb_type=4), it represents each particle's maximum outdegree in the swarm topology.
     * The minimum outdegree is 1 (the particle always connects back to itself). If neighb_type is 1 or 3
     * this parameter is ignored.
     * @param memory when true the particle velocities are not reset between successive calls to evolve
     * @param seed seed used by the internal random number generator (default is random)
     *
     * @throws std::invalid_argument if omega is not in the [0,1] interval, eta1, eta2 are not in the [0,1] interval,
     * vcoeff is not in ]0,1], variant is not one of 1 .. 6, neighb_type is not one of 1 .. 4, neighb_param is zero
     */
    pso_gen(unsigned gen = 1u, double omega = 0.7298, double eta1 = 2.05, double eta2 = 2.05, double max_vel = 0.5,
            unsigned variant = 5u, unsigned neighb_type = 2u, unsigned neighb_param = 4u, bool memory = false,
            unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method (juice implementation of the algorithm)
    population evolve(population) const;

    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0u: no verbosity
     * - >=1u: will print and log one line each \p level generations
     *
     * Example (verbosity 50u):
     * @code{.unparsed}
     * Gen:        Fevals:         gbest:     Mean Vel.:    Mean lbest:    Avg. Dist.:
     *    1             40        2.01917       0.298551        1855.03       0.394038
     *    51           1040     0.00436298      0.0407766         1.0704         0.1288
     *   101           2040    0.000228898      0.0110884       0.282699      0.0488969
     *   151           3040    5.53426e-05     0.00231688       0.106807      0.0167147
     *   201           4040    3.88181e-06    0.000972132      0.0315856     0.00988859
     *   251           5040    1.25676e-06    0.000330553     0.00146805     0.00397989
     *   301           6040    3.76784e-08    0.000118192    0.000738972      0.0018789
     *   351           7040    2.35193e-09    5.39387e-05    0.000532189     0.00253805
     *   401           8040    3.24364e-10     2.2936e-05    9.02879e-06    0.000178279
     *   451           9040    2.31237e-10    5.01558e-06    8.12575e-07    9.77163e-05
     * @endcode
     *
     * Gen is the generation number, Fevals the number of fitness evaluation made, gbest the global best,
     * Mean Vel. the average mean normalized velocity of particles, Mean lbest the average of the local best
     * fitness of particles and Avg. Dist. the average normalized distance among particles. Normalization is made
     * with respect to the problem bounds.
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
        return "GPSO: Generational Particle Swarm Optimization";
    }
    // Extra info
    std::string get_extra_info() const;
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a pso_gen::log_line_type containing: Gen, Fevals, gbest,
     * Mean Vel., Mean lbest, Avg. Dist. as described in pso_gen::set_verbosity
     * @return an <tt>std::vector</tt> of pso_gen::log_line_type containing the logged values
     */
    const log_type &get_log() const
    {
        return m_log;
    }
    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL vector_double
    particle__get_best_neighbor(population::size_type pidx, std::vector<std::vector<vector_double::size_type>> &neighb,
                                const std::vector<vector_double> &lbX, const std::vector<vector_double> &lbfit) const;
    PAGMO_DLL_LOCAL void initialize_topology__gbest(const population &pop, vector_double &gbX, vector_double &gbfit,
                                                    std::vector<std::vector<vector_double::size_type>> &neighb) const;
    PAGMO_DLL_LOCAL void initialize_topology__lbest(std::vector<std::vector<vector_double::size_type>> &neighb) const;
    PAGMO_DLL_LOCAL void initialize_topology__von(std::vector<std::vector<vector_double::size_type>> &neighb) const;
    PAGMO_DLL_LOCAL void
    initialize_topology__adaptive_random(std::vector<std::vector<vector_double::size_type>> &neighb) const;

    // Generations
    unsigned m_max_gen;
    // Inertia (or constriction) coefficient
    double m_omega;
    double m_eta1;
    double m_eta2;
    // Maximum particle velocity
    double m_max_vel;
    // Algoritmic variant
    unsigned m_variant;
    // Particle topology (only relevant for some variants)
    unsigned m_neighb_type;
    // Neighbourhood parameter (only relevant for some variants)
    unsigned m_neighb_param;
    // memory
    bool m_memory;
    // paricles' velocities
    mutable std::vector<vector_double> m_V;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    boost::optional<bfe> m_bfe;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::pso_gen)

#endif
