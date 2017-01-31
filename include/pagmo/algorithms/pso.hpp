/* Copyright 2017 PaGMO development team

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

#ifndef PAGMO_ALGORITHMS_PSO_HPP
#define PAGMO_ALGORITHMS_PSO_HPP

#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include "../algorithm.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../rng.hpp"

namespace pagmo
{

/// Particle Swarm optimization
/**
 * \image html sea.png
 *
 * Particle swarm optimization (PSO) is a population based algorithm inspired by the foraging behaviour of swarms.
 * In PSO each point has memory of the position where it achieved the best performance \f$\mathbf x^l_i\f$ (local
 *memory)
 * and of the swarm best decision vector \f$ \mathbf x^g \f$ (global memory) and uses this information to update
 * its position using the equation:
 * \f[
 *	\mathbf v_{i+1} = \omega \mathbf v_i + \eta_1 \mathbf r_1 \cdot \left( \mathbf x_i - \mathbf x^l_i \right)
 *	+ \eta_2 \mathbf r_2 \cdot \left(  \mathbf x_i - \mathbf x^g \right)
 * \f]
 * \f[
 *	\mathbf x_{i+1} = \mathbf x_i + \mathbf v_i
 * \f]
 *
 * The user can specify the values for \f$\omega, \eta_1, \eta_2\f$ and the magnitude of the maximum velocity
 * allowed. this last value is evaluated for each search direction as the product of \f$ vcoeff\f$ and the
 * search space width along that direction. The user can also specify one of four variants where the velocity
 * update rule differs on the definition of the random vectors \f$r_1\f$ and \f$r_2\f$:
 *
 * \li Variant 1: \f$\mathbf r_1 = [r_{11}, r_{12}, ..., r_{1n}]\f$, \f$\mathbf r_2 = [r_{21}, r_{21}, ..., r_{2n}]\f$
 * \li Variant 2: \f$\mathbf r_1 = [r_{11}, r_{12}, ..., r_{1n}]\f$, \f$\mathbf r_2 = [r_{11}, r_{11}, ..., r_{1n}]\f$
 * \li Variant 3: \f$\mathbf r_1 = [r_1, r_1, ..., r_1]\f$, \f$\mathbf r_2 = [r_2, r_2, ..., r_2]\f$
 * \li Variant 4: \f$\mathbf r_1 = [r_1, r_1, ..., r_1]\f$, \f$\mathbf r_2 = [r_1, r_1, ..., r_1]\f$
 * \li Variant 5: \f$\mathbf r_1 = [r_1, r_1, ..., r_1]\f$, \f$\mathbf r_2 = [r_1, r_1, ..., r_1]\f$
 *
 *
 * **NOTE** The algorithm does not work for multi-objective problems, nor for
 * constrained or stochastic optimization
 *
 * @see http://www.particleswarm.info/ for a repository of information related to PSO
 * @see http://dx.doi.org/10.1007/s11721-007-0002-0 for a survey
 * @see http://www.engr.iupui.edu/~shi/Coference/psopap4.html for the first paper on this algorithm
 */
class pso
{
public:
    /// Single entry of the log (fevals, best, current, avg_range, temperature)
    // typedef std::tuple<unsigned long long, double, double, double, double> log_line_type;
    /// The log
    // typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Allows to specify in detail all the parameters of the algorithm.
     *
     * @param[in] gen number of generations
     * @param[in] omega particles' inertia weight, or alternatively, the constriction coefficient (definition depends on
     * the variant used)
     * @param[in] eta1 magnitude of the force, applied to the particle's velocity, in the direction of its previous best
     * position
     * @param[in] eta2 magnitude of the force, applied to the particle's velocity, in the direction of the best position
     * in its neighborhood
     * @param[in] max_vel maximum allowed particle velocity (as a fraction of the box bounds)
     * @param[in] variant algorithm variant to use (one of 1 .. 6)
     * @param[in] neighb_type swarm topology to use (one of 1 .. 4) [gbest, lbest, von, adaptive random]
     * @param[in] neighb_param the neighbourhood parameter. If the lbest topology is selected (neighb_type=2),
     * it represents each particle's indegree (also outdegree) in the swarm topology. Particles have neighbours up
     * to a radius of k = neighb_param / 2 in the ring. If the Randomly-varying neighbourhood topology
     * is selected (neighb_type=4), it represents each particle's maximum outdegree in the swarm topology.
     * The minimum outdegree is 1 (the particle always connects back to itself). If neighb_type is 1 or 3
     * this parameter is ignored.
     *
     * @throws std::invalid_argument if m_omega is not in the [0,1] interval, eta1, eta2 are not in the [0,1] interval,
     * vcoeff is not in ]0,1], variant is not one of 1 .. 6, neighb_type is not one of 1 .. 4, neighb_param is zero
     */
    pso(unsigned int gen = 1u, double omega = 0.7298, double eta1 = 2.05, double eta2 = 2.05, double max_vel = 0.5,
        unsigned int variant = 5u, unsigned int neighb_type = 2u, unsigned int neighb_param = 4u,
        unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_omega(omega), m_eta1(eta1), m_eta2(eta2), m_max_vel(max_vel), m_variant(variant),
          m_neighb_type(neighb_type), m_neighb_param(neighb_param), m_e(seed), m_seed(seed), m_verbosity(0u) //, m_log()
    {
        if (m_omega < 0. || m_omega > 1.) {
            // variants using Inertia weight
            pagmo_throw(
                std::invalid_argument,
                "The particles' inertia (or the constriction factor) must be in the [0,1] range, while a value of "
                    + std::to_string(m_variant) + " was detected");
        }
        if (m_eta1 < 0. || m_eta2 < 0. || m_eta1 > 4. || m_eta2 > 4.) {
            pagmo_throw(std::invalid_argument, "The eta parameters must be in the [0,4] range, while eta1 = "
                                                   + std::to_string(m_eta1) + ", eta2 = " + std::to_string(m_eta2)
                                                   + " was detected");
        }
        if (m_max_vel <= 0. || m_max_vel > 1.) {
            pagmo_throw(std::invalid_argument, "The maximum particle velocity (as a fraction of the bounds) should be "
                                               "in the (0,1] range, while a value of "
                                                   + std::to_string(m_max_vel) + " was detected");
        }
        if (m_variant < 1u || m_variant > 6u) {
            pagmo_throw(std::invalid_argument, "The PSO variant must be in [1,6], while a value of "
                                                   + std::to_string(m_variant) + " was detected");
        }
        if (m_neighb_type < 1u || m_neighb_type > 4u) {
            pagmo_throw(std::invalid_argument, "The swarm topology variant must be in [1,4], while a value of "
                                                   + std::to_string(m_neighb_type) + " was detected");
        }
        if (m_neighb_param < 1u) {
            pagmo_throw(std::invalid_argument, "The neighborhood parameter must be in (0, inf), while a value of "
                                                   + std::to_string(m_neighb_param) + " was detected");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained
     */
    population evolve(population pop) const
    {
        // We store some useful properties
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();             // not const as used type for counters
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto fevals0 = prob.get_fevals(); // discount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.get_nf() != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        if (pop.size() < 1u) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 1 individual in the population, "
                                                   + std::to_string(pop.size()) + " detected");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        // m_log.clear();

        return pop;
    };
    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >=1: will print and log one line at minimum every \p level function evaluations.
     *
     * Example (verbosity 5000):
     * @code{.unparsed}
     * Fevals:          Best:       Current:    Mean range:   Temperature:
     *  ...
     *  45035      0.0700823       0.135928     0.00116657      0.0199526
     *  50035      0.0215442      0.0261641    0.000770297           0.01
     *  55035     0.00551839      0.0124842    0.000559839     0.00501187
     *  60035     0.00284761     0.00703856    0.000314098     0.00251189
     *  65035     0.00264808      0.0114764    0.000314642     0.00125893
     *  70035      0.0011007     0.00293813    0.000167859    0.000630957
     *  75035    0.000435798     0.00184352    0.000126954    0.000316228
     *  80035    0.000287984    0.000825294    8.91823e-05    0.000158489
     *  85035     9.5885e-05    0.000330647    6.49981e-05    7.94328e-05
     *  90035     4.7986e-05    0.000148512    4.24692e-05    3.98107e-05
     *  95035    2.43633e-05    2.43633e-05    2.90025e-05    1.99526e-05
     * @endcode
     *
     * Fevals is the number of function evaluation used, Best is the best fitness
     * function found, Current is the last fitness sampled, Mean range is the Mean
     * search range across the decision vector components, Temperature is the current temperature.
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    };
    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned int get_seed() const
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
        return "Particle Swarm Optimization";
    }
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tGenerations: ", m_gen);
        stream(ss, "\n\tomega: ", m_omega);
        stream(ss, "\n\teta1: ", m_eta1);
        stream(ss, "\n\teta2: ", m_eta2);
        stream(ss, "\n\tMaximum velocity: ", m_max_vel);
        stream(ss, "\n\tVariant: ", m_variant);
        stream(ss, "\n\tTopology: ", m_neighb_type);
        if (m_neighb_type == 2 || m_neighb_type == 4) {
            stream(ss, "\n\tTopology parameter: ", m_neighb_param);
        }
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        return ss.str();
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt> std::vector </tt> is a simulated_annealing::log_line_type containing: Fevals, Best, Current, Mean range
     * Temperature as described in simulated_annealing::set_verbosity
     * @return an <tt> std::vector </tt> of simulated_annealing::log_line_type containing the logged values Gen, Fevals,
     * Best, Improvement, Mutations
     */
    // const log_type &get_log() const
    //{
    //    return m_log;
    //}
    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDA and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen, m_omega, m_eta1, m_eta2, m_max_vel, m_variant, m_neighb_type, m_neighb_param, m_e, m_seed,
           m_verbosity); //, m_log);
    }

private:
    // Generations
    unsigned int m_gen;
    // Inertia (or constriction) coefficient
    double m_omega;
    double m_eta1;
    double m_eta2;
    // Maximum particle velocity
    double m_max_vel;
    // Algoritmic variant
    unsigned int m_variant;
    // Particle topology (only relevant for some variants)
    unsigned int m_neighb_type;
    // Neighbourhood parameter (only relevant for some variants)
    unsigned int m_neighb_param;

    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    // mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::pso)

#endif
