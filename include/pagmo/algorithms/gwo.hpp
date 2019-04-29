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

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/utils/generic.hpp>

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
 * as is the case for the grey wolf optimizer, are often but small variations of already existing heuristics rebranded with unnecessray and convoluted
 * biological metaphors. In the case of GWO this is particularly evident as the position update rule is shokingly
 * trivial and can also be easily seen as a product of an evolutionary metaphor or a particle swarm one. Such an update rule
 * is also not particulary effective and results in a rather poor performance most of times. Reading the original
 * peer-reviewed paper, where the poor algoritmic perfromance is hidden by the methodological flaws of the benchmark presented,
 * one is left with a bitter opinion of the whole peer-review system.
 *
 * The implementation provided for PaGMO is based on the pseudo-code provided in the original Seyedali and Andrew (2014) paper.
 * pagmo::gwo is suitable for box-constrained single-objective continuous optimization.
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
class gwo
{
public:
    /// Single entry of the log (gen, alpha, beta, delta)
    typedef std::tuple<unsigned int, double, double, double> log_line_type;
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

    gwo(unsigned int gen = 1u, unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_seed(seed), m_e(seed), m_verbosity(0u), m_log()
    {
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for a maximum number of generations, until maximum number
     * of generations is reached.
     *
     * @param pop population to be evolved
     * @return alpha agent's position
     * @throws std::invalid_argument if the problem is multi-objective or constrained or stochastic
     * @throws std::invalid_argument if the population size is not at least 3
     */

    population evolve(population pop) const
    {
        const auto &prob = pop.get_problem();
        auto dim = prob.get_nx();
        auto prob_f_dimension = prob.get_nf();
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto NP = pop.size(); // number of agent(wolf) equal to population size
        auto count = 1u;      // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob_f_dimension != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        if (pop.size() < 3u) {
            pagmo_throw(std::invalid_argument, get_name() + " needs at least 3 individuals in the population, "
                                                   + std::to_string(pop.size()) + " detected");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Some vectors used during evolution are declared.
        vector_double a_vector(3); // vector coefficient for encircling prey
        vector_double c_vector(3); // vector coefficient for encircling prey
        vector_double d_vector(3); // position vector for alpha, beta and delta
        vector_double x_vector(3); // vector used to calculate position of an agent in a single dimension
        double a;                  // coefficient which decrease linearly from 2 to 0 over generations
        double r1, r2;             // random coefficient between 0 and 1;
        auto agents_position = pop.get_x();
        auto init_fit = pop.get_f();
        std::vector<decltype(init_fit.size())> index_vec(init_fit.size()); // used to stored sorted index
        std::iota(index_vec.begin(), index_vec.end(), decltype(init_fit.size())(0));
        std::sort(index_vec.begin(), index_vec.end(), [&](decltype(init_fit.size()) j, decltype(init_fit.size()) k) {
            return detail::greater_than_f(init_fit[j][0], init_fit[k][0]);
        });
        double alpha_score = init_fit[index_vec[0]][0];
        double beta_score = init_fit[index_vec[1]][0];
        double delta_score = init_fit[index_vec[2]][0];
        vector_double alpha_pos = agents_position[index_vec[0]];
        vector_double beta_pos = agents_position[index_vec[1]];
        vector_double delta_pos = agents_position[index_vec[2]];
        std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)

        // Main gwo iterations
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {

            a = 2.0 - static_cast<double>(gen) * (2.0 / static_cast<double>(m_gen));
            // for each agents
            for (decltype(NP) i = 0u; i < NP; ++i) {

                // Encircling prey and attack
                for (decltype(dim) k = 0; k < dim; ++k) {
                    r1 = drng(m_e);
                    r2 = drng(m_e);

                    a_vector[0] = 2.0 * a * r1 - a; // Equation (3.3)
                    c_vector[0] = 2.0 * r2;         // Equation (3.4)

                    d_vector[0] = std::abs(c_vector[0] * alpha_pos[k] - agents_position[i][k]); // Equation (3.5)-part 1
                    x_vector[0] = alpha_pos[k] - a_vector[0] * d_vector[0];                     // Equation (3.6)-part 1

                    r1 = drng(m_e);
                    r2 = drng(m_e);

                    a_vector[1] = 2.0 * a * r1 - a;
                    c_vector[1] = 2.0 * r2;

                    d_vector[1] = std::abs(c_vector[1] * beta_pos[k] - agents_position[i][k]); // Equation (3.5)-part 2
                    x_vector[1] = beta_pos[k] - a_vector[1] * d_vector[1];                     // Equation (3.6)-part 2

                    r1 = drng(m_e);
                    r2 = drng(m_e);

                    a_vector[2] = 2.0 * a * r1 - a;
                    c_vector[2] = 2.0 * r2;

                    d_vector[2] = std::abs(c_vector[2] * delta_pos[k] - agents_position[i][k]); // Equation (3.5)-part 3
                    x_vector[2] = delta_pos[k] - a_vector[2] * d_vector[2];                     // Equation (3.6)-part 3

                    agents_position[i][k] = (x_vector[0] + x_vector[1] + x_vector[2]) / 3.0; // Equation (3.7)
                }
                // clip position value that goes beyond search space
                detail::force_bounds_stick(agents_position[i], lb, ub);
                auto newfitness = prob.fitness(agents_position[i]);
                pop.set_xf(i, agents_position[i], newfitness);
                // get updated fitness
                auto fit = pop.get_f()[i];
                // Update alpha, beta and delta
                if (fit[0] < alpha_score) {
                    alpha_score = fit[0];
                    alpha_pos = agents_position[i];
                }

                if (fit[0] > alpha_score && fit[0] < beta_score) {
                    beta_score = fit[0];
                    beta_pos = agents_position[i];
                }

                if (fit[0] > alpha_score && fit[0] > beta_score && fit[0] < delta_score) {
                    delta_score = fit[0];
                    delta_pos = agents_position[i];
                }

            } // End of one agent iteration

            /// Single entry of the log (gen, alpha, beta, delta)
            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Alpha:", std::setw(15),
                              "Beta:", std::setw(15), "Delta:", '\n');
                    }
                    print(std::setw(7), gen, std::setw(15), alpha_score, std::setw(15), beta_score, std::setw(15),
                          delta_score, '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(gen, alpha_score, beta_score, delta_score);
                }
            }

        } // end main gmo iterations

        return pop;

    } // end of evolve method

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
    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned int get_gen() const
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
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) + "\n\tVerbosity: " + std::to_string(m_verbosity)
               + "\n\tSeed: " + std::to_string(m_seed);
    }

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

    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen, m_seed, m_e, m_verbosity, m_log);
    }

private:
    unsigned int m_gen;
    unsigned int m_seed;
    mutable detail::random_engine_type m_e;
    unsigned int m_verbosity;
    mutable log_type m_log;

}; // class gwo
} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::gwo)

#endif