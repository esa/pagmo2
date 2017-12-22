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

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

/// Particle Swarm Optimization
/**
 * \image html pso.png width=5cm
 *
 * Particle swarm optimization (PSO) is a population based algorithm inspired by the foraging behaviour of swarms.
 * In PSO each point has memory of the position where it achieved the best performance \f$\mathbf x^l_i\f$ (local
 * memory)
 * and of the best decision vector \f$ \mathbf x^g \f$ in a certain neighbourhood, and uses this information to update
 * its position using the equations (constriction coefficient):
 * \f[
 * \begin{array}{l}
 *	\mathbf v_{i+1} = \omega \left( \mathbf v_i + \eta_1 \mathbf r_1 \cdot \left( \mathbf x_i - \mathbf x^l_i \right)
 *	+ \eta_2 \mathbf r_2 \cdot \left(  \mathbf x_i - \mathbf x^g \right) \right)
 * \\
 *	\mathbf x_{i+1} = \mathbf x_i + \mathbf v_i
 *	\end{array}
 * \f]
 * or (inertia weight):
 * \f[
 * \begin{array}{l}
 *	\mathbf v_{i+1} = \omega \mathbf v_i + \eta_1 \mathbf r_1 \cdot \left( \mathbf x_i - \mathbf x^l_i \right)
 *	+ \eta_2 \mathbf r_2 \cdot \left(  \mathbf x_i - \mathbf x^g \right)
 * \\
 *	\mathbf x_{i+1} = \mathbf x_i + \mathbf v_i
 *	\end{array}
 * \f]
 *
 * The user can specify the values for \f$\omega, \eta_1, \eta_2\f$ and the magnitude of the maximum velocity
 * allowed (normalized with respect ot the bounds). The user can specify one of five variants where the velocity
 * update rule differs on the definition of the random vectors \f$r_1\f$ and \f$r_2\f$:
 *
 * \li Variant 1: \f$\mathbf r_1 = [r_{11}, r_{12}, ..., r_{1n}]\f$, \f$\mathbf r_2 = [r_{21}, r_{21}, ..., r_{2n}]\f$
 *... (inertia weight)
 * \li Variant 2: \f$\mathbf r_1 = [r_{11}, r_{12}, ..., r_{1n}]\f$, \f$\mathbf r_2 = [r_{11}, r_{11}, ..., r_{1n}]\f$
 *... (inertia weight)
 * \li Variant 3: \f$\mathbf r_1 = [r_1, r_1, ..., r_1]\f$, \f$\mathbf r_2 = [r_2, r_2, ..., r_2]\f$ ... (inertia
 *weight)
 * \li Variant 4: \f$\mathbf r_1 = [r_1, r_1, ..., r_1]\f$, \f$\mathbf r_2 = [r_1, r_1, ..., r_1]\f$ ... (inertia
 *weight)
 * \li Variant 5: \f$\mathbf r_1 = [r_{11}, r_{12}, ..., r_{1n}]\f$, \f$\mathbf r_2 = [r_{21}, r_{21}, ..., r_{2n}]\f$
 *... (constriction coefficient)
 * \li Variant 6: Fully Informed Particle Swarm (FIPS)
 *
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The default variant in PaGMO is n. 5 corresponding to the canonical PSO and thus using the constriction
 *    coefficient velocity update formula
 *
 * .. warning::
 *
 *    The algorithm is not suitable for multi-objective problems, nor for
 *    constrained or stochastic optimization
 *
 * .. seealso::
 *
 *    http://www.particleswarm.info/ for a repository of information related to PSO
 *
 * .. seealso::
 *
 *    http://dx.doi.org/10.1007/s11721-007-0002-0 for a survey
 *
 * \endverbatim
 */
class pso
{
public:
    /// Single entry of the log (Gen, Fevals, gbest, Mean Vel., Mean lbest, Avg. Dist.)
    typedef std::tuple<unsigned int, unsigned long long, double, double, double, double> log_line_type;
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
    pso(unsigned int gen = 1u, double omega = 0.7298, double eta1 = 2.05, double eta2 = 2.05, double max_vel = 0.5,
        unsigned int variant = 5u, unsigned int neighb_type = 2u, unsigned int neighb_param = 4u, bool memory = false,
        unsigned int seed = pagmo::random_device::next())
        : m_max_gen(gen), m_omega(omega), m_eta1(eta1), m_eta2(eta2), m_max_vel(max_vel), m_variant(variant),
          m_neighb_type(neighb_type), m_neighb_param(neighb_param), m_memory(memory), m_V(), m_e(seed), m_seed(seed),
          m_verbosity(0u), m_log()
    {
        if (m_omega < 0. || m_omega > 1.) {
            // variants using Inertia weight
            pagmo_throw(
                std::invalid_argument,
                "The particles' inertia (or the constriction factor) must be in the [0,1] range, while a value of "
                    + std::to_string(m_variant) + " was detected");
        }
        if (m_eta1 < 0. || m_eta2 < 0. || m_eta1 > 4. || m_eta2 > 4.) {
            pagmo_throw(std::invalid_argument,
                        "The eta parameters must be in the [0,4] range, while eta1 = " + std::to_string(m_eta1)
                            + ", eta2 = " + std::to_string(m_eta2) + " was detected");
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
                                              // allowed
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
        if (!pop.size()) {
            pagmo_throw(std::invalid_argument, get_name() + " does not work on an empty population");
        }
        // ---------------------------------------------------------------------------------------------------------
        // No throws, all valid: we clear the logs
        m_log.clear();

        auto swarm_size = pop.size();
        // Some vectors used are allocated here.
        vector_double dummy(dim, 0.); // used for initialisation purposes

        std::vector<vector_double> X(swarm_size, dummy); // particles' current positions
        std::vector<vector_double> fit(swarm_size);      // particles' current fitness values

        std::vector<vector_double> lbX(swarm_size, dummy); // particles' previous best positions
        std::vector<vector_double> lbfit(swarm_size);      // particles' fitness values at their previous best positions

        // swarm topology (iterators over indexes of each particle's neighbors in the swarm)
        std::vector<std::vector<decltype(swarm_size)>> neighb(swarm_size);
        // search space position of particles' best neighbor
        vector_double best_neighb(dim, 0.);
        // fitness at the best found search space position (tracked only when using topologies 1 or 4)
        vector_double best_fit;
        // flag indicating whether the best solution's fitness improved (tracked only when using topologies 1 or 4)
        bool best_fit_improved;

        vector_double minv(dim), maxv(dim); // Maximum and minimum velocity allowed

        double vwidth; // Temporary variable
        double new_x;  // Temporary variable

        std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)

        // Initialise the minimum and maximum velocity
        for (decltype(dim) i = 0u; i < dim; ++i) {
            vwidth = (ub[i] - lb[i]) * m_max_vel;
            minv[i] = -1. * vwidth;
            maxv[i] = vwidth;
        }

        // Copy the particle positions and their fitness
        for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
            X[i] = pop.get_x()[i];
            lbX[i] = pop.get_x()[i];

            fit[i] = pop.get_f()[i];
            lbfit[i] = pop.get_f()[i];
        }

        // Initialize the particle velocities if necessary
        if ((m_V.size() != swarm_size) || (!m_memory)) {
            m_V = std::vector<vector_double>(swarm_size, dummy);
            for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    m_V[i][j] = uniform_real_from_range(minv[j], maxv[j], m_e);
                }
            }
        }

        // Initialize the Swarm's topology
        switch (m_neighb_type) {
            case 1:
                initialize_topology__gbest(pop, best_neighb, best_fit, neighb);
                break;
            case 3:
                initialize_topology__von(neighb);
                break;
            case 4:
                initialize_topology__adaptive_random(neighb);
                // need to track improvements in best found fitness, to know when to rewire
                best_fit = pop.get_f()[pop.best_idx()];
                break;
            case 2:
            default:
                initialize_topology__lbest(neighb);
        }
        // auxiliary variables specific to the Fully Informed Particle Swarm variant
        double acceleration_coefficient = m_eta1 + m_eta2;
        double sum_forces;

        double r1 = 0.;
        double r2 = 0.;

        /* --- Main PSO loop ---
         */
        // For each generation
        for (decltype(m_max_gen) gen = 1u; gen <= m_max_gen; ++gen) {
            best_fit_improved = false;
            // For each particle in the swarm
            for (decltype(swarm_size) p = 0u; p < swarm_size; ++p) {

                // identify the current particle's best neighbour
                // . not needed if m_neighb_type == 1 (gbest): best_neighb directly tracked in this function
                // . not needed if m_variant == 6 (FIPS): all neighbours are considered, no need to identify the best
                // one
                if (m_neighb_type != 1u && m_variant != 6u)
                    best_neighb = particle__get_best_neighbor(p, neighb, lbX, lbfit);

                /*-------PSO canonical (with inertia weight) ---------------------------------------------*/
                /*-------Original algorithm used in the first PaGMO paper (~2007) ------------------------*/
                if (m_variant == 1u) {
                    for (decltype(dim) d = 0u; d < dim; ++d) {
                        r1 = drng(m_e);
                        r2 = drng(m_e);
                        m_V[p][d] = m_omega * m_V[p][d] + m_eta1 * r1 * (lbX[p][d] - X[p][d])
                                    + m_eta2 * r2 * (best_neighb[d] - X[p][d]);
                    }
                }

                /*-------PSO canonical (with inertia weight) ---------------------------------------------*/
                /*-------and with equal random weights of social and cognitive components-----------------*/
                /*-------Check with Rastrigin-------------------------------------------------------------*/
                else if (m_variant == 2u) {
                    for (decltype(dim) d = 0u; d < dim; ++d) {
                        r1 = drng(m_e);
                        m_V[p][d] = m_omega * m_V[p][d] + m_eta1 * r1 * (lbX[p][d] - X[p][d])
                                    + m_eta2 * r1 * (best_neighb[d] - X[p][d]);
                    }
                }

                /*-------PSO variant (commonly mistaken in literature for the canonical)----------------*/
                /*-------Same random number for all components------------------------------------------*/
                else if (m_variant == 3u) {
                    r1 = drng(m_e);
                    r2 = drng(m_e);
                    for (decltype(dim) d = 0u; d < dim; ++d) {
                        m_V[p][d] = m_omega * m_V[p][d] + m_eta1 * r1 * (lbX[p][d] - X[p][d])
                                    + m_eta2 * r2 * (best_neighb[d] - X[p][d]);
                    }
                }

                /*-------PSO variant (commonly mistaken in literature for the canonical)----------------*/
                /*-------Same random number for all components------------------------------------------*/
                /*-------and with equal random weights of social and cognitive components---------------*/
                else if (m_variant == 4u) {
                    r1 = drng(m_e);
                    for (decltype(dim) d = 0u; d < dim; ++d) {
                        m_V[p][d] = m_omega * m_V[p][d] + m_eta1 * r1 * (lbX[p][d] - X[p][d])
                                    + m_eta2 * r1 * (best_neighb[d] - X[p][d]);
                    }
                }

                /*-------PSO variant with constriction coefficients------------------------------------*/
                /*  ''Clerc's analysis of the iterative system led him to propose a strategy for the
                 *  placement of "constriction coefficients" on the terms of the formulas; these
                 *  coefficients controlled the convergence of the particle and allowed an elegant and
                 *  well-explained method for preventing explosion, ensuring convergence, and
                 *  eliminating the arbitrary Vmax parameter. The analysis also takes the guesswork
                 *  out of setting the values of phi_1 and phi_2.''
                 *  ''this is the canonical particle swarm algorithm of today.''
                 *  [Poli et al., 2007] http://dx.doi.org/10.1007/s11721-007-0002-0
                 *  [Clerc and Kennedy, 2002] http://dx.doi.org/10.1109/4235.985692
                 *
                 *  This being the canonical PSO of today, this variant is set as the default in PaGMO.
                 *-------------------------------------------------------------------------------------*/
                else if (m_variant == 5u) {
                    for (decltype(dim) d = 0u; d < dim; ++d) {
                        r1 = drng(m_e);
                        r2 = drng(m_e);
                        m_V[p][d] = m_omega
                                    * (m_V[p][d] + m_eta1 * r1 * (lbX[p][d] - X[p][d])
                                       + m_eta2 * r2 * (best_neighb[d] - X[p][d]));
                    }
                }

                /*-------Fully Informed Particle Swarm-------------------------------------------------*/
                /*  ''Whereas in the traditional algorithm each particle is affected by its own
                 *  previous performance and the single best success found in its neighborhood, in
                 *  Mendes' fully informed particle swarm (FIPS), the particle is affected by all its
                 *  neighbors, sometimes with no influence from its own previous success.''
                 *  ''With good parameters, FIPS appears to find better solutions in fewer iterations
                 *  than the canonical algorithm, but it is much more dependent on the population topology.''
                 *  [Poli et al., 2007] http://dx.doi.org/10.1007/s11721-007-0002-0
                 *  [Mendes et al., 2004] http://dx.doi.org/10.1109/TEVC.2004.826074
                 *-------------------------------------------------------------------------------------*/
                else if (m_variant == 6u) {
                    for (decltype(dim) d = 0u; d < dim; ++d) {
                        sum_forces = 0.;
                        for (decltype(neighb[p].size()) n = 0u; n < neighb[p].size(); ++n) {
                            sum_forces += drng(m_e) * acceleration_coefficient * (lbX[neighb[p][n]][d] - X[p][d]);
                        }
                        m_V[p][d] = m_omega * (m_V[p][d] + sum_forces / static_cast<double>(neighb[p].size()));
                    }
                }

                // We now check that the velocity does not exceed the maximum allowed per component
                // and we perform the position update and the feasibility correction
                for (decltype(dim) d = 0u; d < dim; ++d) {

                    if (m_V[p][d] > maxv[d]) {
                        m_V[p][d] = maxv[d];
                    }

                    else if (m_V[p][d] < minv[d]) {
                        m_V[p][d] = minv[d];
                    }

                    // update position
                    new_x = X[p][d] + m_V[p][d];

                    // feasibility correction
                    // (velocity updated to that which would have taken the previous position
                    // to the newly corrected feasible position)
                    if (new_x < lb[d]) {
                        new_x = lb[d];
                        m_V[p][d] = 0.;
                        //					new_x = boost::uniform_real<double>(lb[d],ub[d])(m_drng);
                        //					V[p][d] = new_x - X[p][d];
                    } else if (new_x > ub[d]) {
                        new_x = ub[d];
                        m_V[p][d] = 0.;
                        //					new_x = boost::uniform_real<double>(lb[d],ub[d])(m_drng);
                        //					V[p][d] = new_x - X[p][d];
                    }
                    X[p][d] = new_x;
                }
                // We evaluate here the new individual fitness
                // as to be able to update the global best in real time
                fit[p] = prob.fitness(X[p]);

                if (fit[p] <= lbfit[p]) {
                    // update the particle's previous best position
                    lbfit[p] = fit[p];
                    lbX[p] = X[p];
                    // update the best position observed so far by any particle in the swarm
                    // (only performed if swarm topology is gbest)
                    if ((m_neighb_type == 1u || m_neighb_type == 4u) && (fit[p] <= best_fit)) {
                        best_neighb = X[p];
                        best_fit = fit[p];
                        best_fit_improved = true;
                    }
                }
            } // End of loop on the population members
            // reset swarm topology if no improvement was observed in the best found fitness value
            if (m_neighb_type == 4u && !best_fit_improved) initialize_topology__adaptive_random(neighb);
            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    // We compute the number of fitness evaluations made
                    auto feval_count = prob.get_fevals() - fevals0;
                    // We compute the average across the swarm of the best fitness encountered
                    vector_double local_fits(swarm_size);
                    for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
                        local_fits[i] = lbfit[i][0];
                    }
                    auto lb_avg = std::accumulate(local_fits.begin(), local_fits.end(), 0.)
                                  / static_cast<double>(local_fits.size());
                    // We compute the best fitness encounterd so far across generations and across the swarm
                    // TODO: distance returns a signed type that can be overflown by the local_fits::size_type
                    auto idx_best = std::distance(std::begin(local_fits),
                                                  std::min_element(std::begin(local_fits), std::end(local_fits)));
                    auto best = local_fits[static_cast<vector_double::size_type>(idx_best)];
                    // We compute a measure for the average particle velocity across the swarm
                    auto mean_velocity = 0.;
                    for (decltype(m_V.size()) i = 0u; i < m_V.size(); ++i) {
                        for (decltype(m_V[i].size()) j = 0u; j < m_V[i].size(); ++j) {
                            if (ub[j] > lb[j]) {
                                mean_velocity += std::abs(m_V[i][j] / (ub[j] - lb[j]));
                            } // else 0
                        }
                        mean_velocity /= static_cast<double>(m_V[i].size());
                    }
                    // We compute the average distance across particles (NOTE: N^2 complexity)
                    auto avg_dist = 0.;
                    for (decltype(X.size()) i = 0u; i < X.size(); ++i) {
                        for (decltype(X.size()) j = i + 1u; j < X.size(); ++j) {
                            auto x1 = X[i];
                            auto x2 = X[j];
                            double acc = 0.;
                            for (decltype(x1.size()) k = 0u; k < x1.size(); ++k) {
                                if (ub[j] > lb[j]) {
                                    acc += (x1[k] - x2[k]) * (x1[k] - x2[k]) / (ub[k] - lb[k]) / (ub[k] - lb[k]);
                                } // else 0
                            }
                            avg_dist += std::sqrt(acc);
                        }
                    }
                    avg_dist /= ((static_cast<double>(X.size()) - 1u) * static_cast<double>(X.size())) / 2.;
                    // We start printing
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                              "gbest:", std::setw(15), "Mean Vel.:", std::setw(15), "Mean lbest:", std::setw(15),
                              "Avg. Dist.:", '\n');
                    }
                    print(std::setw(7), gen, std::setw(15), feval_count, std::setw(15), best, std::setw(15),
                          mean_velocity, std::setw(15), lb_avg, std::setw(15), avg_dist, '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(gen, feval_count, best, mean_velocity, lb_avg, avg_dist);
                }
            }
        } // end of main PSO loop
        if (m_verbosity) {
            std::cout << "Exit condition -- generations = " << m_max_gen << std::endl;
        }

        // copy particles' positions & velocities back to the main population
        for (decltype(swarm_size) i = 0u; i < swarm_size; ++i) {
            pop.set_xf(i, lbX[i], lbfit[i]);
        }
        return pop;
    };
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
        m_e.seed(seed);
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
        return "PSO: Particle Swarm Optimization";
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
        stream(ss, "\tGenerations: ", m_max_gen);
        stream(ss, "\n\tOmega: ", m_omega);
        stream(ss, "\n\tEta1: ", m_eta1);
        stream(ss, "\n\tEta2: ", m_eta2);
        stream(ss, "\n\tMaximum velocity: ", m_max_vel);
        stream(ss, "\n\tVariant: ", m_variant);
        stream(ss, "\n\tTopology: ", m_neighb_type);
        if (m_neighb_type == 2u || m_neighb_type == 4u) {
            stream(ss, "\n\tTopology parameter: ", m_neighb_param);
        }
        stream(ss, "\n\tMemory: ", m_memory);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        return ss.str();
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a pso::log_line_type containing: Gen, Fevals, gbest,
     * Mean Vel., Mean lbest, Avg. Dist. as described in pso::set_verbosity
     * @return an <tt>std::vector</tt> of pso::log_line_type containing the logged values
     */
    const log_type &get_log() const
    {
        return m_log;
    }
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
        ar(m_max_gen, m_omega, m_eta1, m_eta2, m_max_vel, m_variant, m_neighb_type, m_neighb_param, m_e, m_seed,
           m_verbosity, m_log);
    }

private:
    /**
     *  @brief Get information on the best position already visited by any of a particle's neighbours
     *
     *  @param pidx index to the particle under consideration
     *  @param neighb definition of the swarm's topology
     *  @param lbX particles' previous best positions
     *  @param lbfit particles' fitness values at their previous best positions
     *  @return best position already visited by any of the considered particle's neighbours
     */
    vector_double particle__get_best_neighbor(population::size_type pidx,
                                              std::vector<std::vector<vector_double::size_type>> &neighb,
                                              const std::vector<vector_double> &lbX,
                                              const std::vector<vector_double> &lbfit) const
    {
        population::size_type bnidx; // neighbour index; best neighbour index

        switch (m_neighb_type) {
            case 1: // { gbest }
                // ERROR: execution should not reach this point, as the global best position is not tracked using the
                // neighb vector
                pagmo_throw(                                                                     // LCOV_EXCL_LINE
                    std::invalid_argument,                                                       // LCOV_EXCL_LINE
                    "particle__get_best_neighbor() invoked while using a gbest swarm topology"); // LCOV_EXCL_LINE
                break;
            case 2: // { lbest }
            case 3: // { von }
            case 4: // { adaptive random }
            default:
                // iterate over indexes of the particle's neighbours, and identify the best
                bnidx = neighb[pidx][0];
                for (decltype(neighb[pidx].size()) nidx = 1u; nidx < neighb[pidx].size(); ++nidx) {
                    if (lbfit[neighb[pidx][nidx]][0] <= lbfit[bnidx][0]) {
                        bnidx = neighb[pidx][nidx];
                    }
                }
                return lbX[bnidx];
        }
    }

    /**
     *  @brief Defines the Swarm topology as a fully connected graph, where particles are influenced by all other
     * particles in the swarm
     *
     *  ''The earliest reported particle swarm version [3], [4] used a kind of
     *  topology that is known as gbest. The source of social influence on each
     *  particle was the best performing individual in the entire population. This
     *  is equivalent to a sociogram or social network where every individual is
     *  connected to every other one.'' \n
     *  ''The gbest topology (i.e., the biggest neighborhood possible) has often
     *  been shown to converge on optima more quickly than lbest, but gbest is also
     *  more susceptible to the attraction of local optima since the population
     *  rushes unanimously toward the first good solution found.'' \n
     *  [Kennedy and Mendes, 2006] http://dx.doi.org/10.1109/TSMCC.2006.875410
     *
     *  @param pop pagmo::population being evolved
     *  @param[out] gbX best search space position already visited by the swarm
     *  @param[out] gbfit best fitness value in the swarm
     *  @param[out] neighb definition of the swarm's topology
     */
    void initialize_topology__gbest(const population &pop, vector_double &gbX, vector_double &gbfit,
                                    std::vector<std::vector<vector_double::size_type>> &neighb) const
    {
        // The best position already visited by the swarm will be tracked in pso::evolve() as particles are evaluated.
        // Here we define the initial values of the variables that will do that tracking.
        gbX = pop.get_x()[pop.best_idx()];
        gbfit = pop.get_f()[pop.best_idx()];

        /* The usage of a gbest swarm topology along with a FIPS (fully informed particle swarm) velocity update formula
         * is discouraged. However, because a user might still configure such a setup, we must ensure FIPS has access to
         * the list of indices of particles' neighbours:
         */
        if (m_variant == 6u) {
            for (decltype(neighb.size()) i = 0u; i < neighb.size(); i++) {
                neighb[0].push_back(i);
            }
            for (decltype(neighb.size()) i = 0u; i < neighb.size(); i++) {
                neighb[i] = neighb[0];
            }
        }
    }

    /**
     *  @brief Defines the Swarm topology as a ring, where particles are influenced by their immediate neighbors to
     *either side
     *
     *  ''The L-best-k topology consists of n nodes arranged in a ring, in which
     *  node i is connected to each node in {(i+j) mod n : j = +-1,+-2, . . . ,+-k}.'' \n
     *  [Mohais et al., 2005] http://dx.doi.org/10.1007/11589990_80
     *
     *  neighb_param represents each particle's indegree (also outdegree) in the swarm topology.
     *	Particles have neighbours up to a radius of k = neighb_param / 2 in the ring.
     *
     *  @param[out] neighb definition of the swarm's topology
     */
    void initialize_topology__lbest(std::vector<std::vector<vector_double::size_type>> &neighb) const
    {
        auto swarm_size = neighb.size();
        auto radius = m_neighb_param / 2u;
        for (decltype(swarm_size) pidx = 0u; pidx < swarm_size; ++pidx) {
            for (decltype(radius) j = radius; j > 0u; j--) {
                if (pidx < j) {
                    neighb[pidx].push_back(pidx - j + swarm_size);
                } else {
                    neighb[pidx].push_back(pidx - j);
                }
            }
            for (decltype(radius) j = 1u; j <= radius; j++) {
                if (pidx + j >= swarm_size) {
                    neighb[pidx].push_back(pidx + j - swarm_size);
                } else {
                    neighb[pidx].push_back(pidx + j);
                }
            }
        }
    }

    /*! @brief Von Neumann neighborhood
     *  (increments on particles' lattice coordinates that produce the coordinates of their neighbors)
     *
     *  The von Neumann neighbourhood of a point includes all the points at a Hamming distance of 1.
     *
     *  - http://en.wikipedia.org/wiki/Von_Neumann_neighborhood
     *  - http://mathworld.wolfram.com/vonNeumannNeighborhood.html
     *  - http://en.wikibooks.org/wiki/Cellular_Automata/Neighborhood
     */
    const int vonNeumann_neighb_diff[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    /**
     *  @brief Arranges particles in a lattice, where each interacts with its immediate 4 neighbors to the N, S, E and
     * W.
     *
     *  ''The population is arranged in a rectangular matrix, for instance, 5 x 4
     *  in a population of 20 individuals, and each individual is connected to
     *  the individuals above, below and on both of its sides, wrapping the edges'' \n
     *  [Kennedy and Mendes, 2006] http://dx.doi.org/10.1109/TSMCC.2006.875410
     *
     *  ''Given a population of size n, the von Neumann neighborhood was
     *  configured into r rows and c columns, where r is the smallest integer
     *  less than or equal to sqrt(n) that evenly divides n and c = n / r'' \n
     *  [Mohais et al., 2005] http://dx.doi.org/10.1007/11589990_80 \n
     *  (there's an error in the description above: "smallest integer" should
     *  instead be "highest integer")
     *
     *  @param[out] neighb definition of the swarm's topology
     */
    void initialize_topology__von(std::vector<std::vector<vector_double::size_type>> &neighb) const
    {
        int swarm_size = static_cast<int>(neighb.size());
        int cols, rows; // lattice structure
        int p_x, p_y;   // particle's coordinates in the lattice
        int n_x, n_y;   // particle neighbor's coordinates in the lattice

        rows = static_cast<int>(std::sqrt(swarm_size));
        while (swarm_size % rows != 0u)
            rows -= 1;
        cols = swarm_size / rows;

        for (decltype(swarm_size) pidx = 0u; pidx < swarm_size; pidx++) {
            p_x = pidx % cols;
            p_y = pidx / cols;

            for (unsigned int nidx = 0u; nidx < 4u; nidx++) {
                n_x = (p_x + vonNeumann_neighb_diff[nidx][0]) % cols;
                if (n_x < 0) n_x = cols + n_x;
                n_y = (p_y + vonNeumann_neighb_diff[nidx][1]) % rows;
                if (n_y < 0) n_y = rows + n_y;

                neighb[static_cast<unsigned int>(pidx)].push_back(static_cast<unsigned int>(n_y * cols + n_x));
            }
        }
    }

    /**
     *  @brief Arranges particles in a random graph having a parameterized maximum outdegree; the graph changes
     *adaptively over time
     *
     *	''At the very beginning, and after each unsuccessful iteration (no
     *	improvement of the best known fitness value), the graph of the information
     *	links is modified: each particle informs at random K particles (the same
     *	particle may be chosen several times), and informs itself. The parameter K
     *	is usually set to 3. It means that each particle informs at less one
     *	particle (itself), and at most K+1 particles (including itself). It also
     *	means that each particle can be informed by any number of particles between
     *	1 and S. However, the distribution of the possible number of "informants"
     *	is not uniform. On average, a particle is often informed by about K others,
     *	but it may be informed by a much larger number of particles with a small
     *	probability'' \n
     *  [Maurice Clerc, 2011] Standard Particle Swarm Optimisation, From 2006 to 2011 \n
     *	http://clerc.maurice.free.fr/pso/SPSO_descriptions.pdf
     *
     *  neighb_param represents each particle's maximum outdegree in the swarm topology.
     *	The minimum outdegree is 1 (the particle always connects back to itself).
     *
     *  @param[out] neighb definition of the swarm's topology
     */
    void initialize_topology__adaptive_random(std::vector<std::vector<vector_double::size_type>> &neighb) const
    {
        auto swarm_size = neighb.size();

        std::uniform_int_distribution<decltype(swarm_size)> dis(0u, swarm_size - 1u);

        // clear previously defined topology
        for (decltype(swarm_size) pidx = 0u; pidx < swarm_size; ++pidx) {
            neighb[pidx].clear();
        }
        // define new topology
        for (decltype(swarm_size) pidx = 0u; pidx < swarm_size; ++pidx) {
            // the particle always connects back to itself, thus guaranteeing a minimum indegree of 1
            neighb[pidx].push_back(pidx);

            for (decltype(m_neighb_param) j = 1u; j < m_neighb_param; ++j) {

                // auto nidx = drng(m_e) * swarm_size;
                neighb[dis(m_e)].push_back(pidx);
                // No check performed to see whether pidx is already in neighb[nidx],
                // leading to a performance penalty in particle__get_best_neighbor() when it occurs.
            }
        }
    }
    // Generations
    unsigned int m_max_gen;
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
    // memory
    bool m_memory;
    // paricles' velocities
    mutable std::vector<vector_double> m_V;

    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::pso)

#endif
