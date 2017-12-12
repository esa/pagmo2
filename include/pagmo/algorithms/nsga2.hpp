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

#ifndef PAGMO_ALGORITHMS_NSGA2_HPP
#define PAGMO_ALGORITHMS_NSGA2_HPP

#include <algorithm> // std::shuffle, std::transform
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>

#include <pagmo/algorithm.hpp> // needed for the cereal macro
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/multi_objective.hpp> // crowding_distance, etc..
#include <pagmo/utils/generic.hpp> // uniform_real_from_range

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
class nsga2
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned int, unsigned long long, vector_double> log_line_type;
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
          unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_cr(cr), m_eta_c(eta_c), m_m(m), m_eta_m(eta_m), m_e(seed), m_seed(seed), m_verbosity(0u),
          m_log()
    {
        if (cr >= 1. || cr < 0.) {
            pagmo_throw(std::invalid_argument, "The crossover probability must be in the [0,1[ range, while a value of "
                                                   + std::to_string(cr) + " was detected");
        }
        if (m < 0. || m > 1.) {
            pagmo_throw(std::invalid_argument, "The mutation probability must be in the [0,1] range, while a value of "
                                                   + std::to_string(cr) + " was detected");
        }
        if (eta_c < 1. || eta_c > 100.) {
            pagmo_throw(std::invalid_argument,
                        "The distribution index for crossover must be in [1, 100], while a value of "
                            + std::to_string(eta_c) + " was detected");
        }
        if (eta_m < 1. || eta_m > 100.) {
            pagmo_throw(std::invalid_argument,
                        "The distribution index for mutation must be in [1, 100], while a value of "
                            + std::to_string(eta_m) + " was detected");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for the requested number of generations.
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throw std::invalid_argument if pop.get_problem() is stochastic, single objective or has non linear constraints.
     * If \p int_dim is larger than the problem dimension. If the population size is smaller than 5 or not a multiple of
     * 4.
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        auto NP = pop.size();

        auto fevals0 = prob.get_fevals(); // discount for the fevals already made
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The problem appears to be stochastic " + get_name() + " cannot deal with it");
        }
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them.");
        }
        if (prob.get_nf() < 2u) {
            pagmo_throw(std::invalid_argument,
                        "This is a multiobjective algortihm, while number of objectives detected in " + prob.get_name()
                            + " is " + std::to_string(prob.get_nf()));
        }
        if (NP < 5u || (NP % 4 != 0u)) {
            pagmo_throw(std::invalid_argument,
                        "for NSGA-II at least 5 individuals in the population are needed and the "
                        "population size must be a multiple of 4. Detected input population size is: "
                            + std::to_string(NP));
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Declarations
        std::vector<vector_double::size_type> best_idx(NP), shuffle1(NP), shuffle2(NP);
        vector_double::size_type parent1_idx, parent2_idx;
        vector_double child1(dim), child2(dim);

        std::iota(shuffle1.begin(), shuffle1.end(), 0u);
        std::iota(shuffle2.begin(), shuffle2.end(), 0u);

        // Main NSGA-II loop
        for (decltype(m_gen) gen = 1u; gen <= m_gen; gen++) {
            // 0 - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    // We compute the ideal point
                    vector_double ideal_point = ideal(pop.get_f());
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:");
                        for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                            if (i >= 5u) {
                                print(std::setw(15), "... :");
                                break;
                            }
                            print(std::setw(15), "ideal" + std::to_string(i + 1u) + ":");
                        }
                        print('\n');
                    }
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0);
                    for (decltype(ideal_point.size()) i = 0u; i < ideal_point.size(); ++i) {
                        if (i >= 5u) {
                            break;
                        }
                        print(std::setw(15), ideal_point[i]);
                    }
                    print('\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(gen, prob.get_fevals() - fevals0, ideal_point);
                }
            }

            // At each generation we make a copy of the population into popnew
            population popnew(pop);

            // We create some pseudo-random permutation of the poulation indexes
            std::shuffle(shuffle1.begin(), shuffle1.end(), m_e);
            std::shuffle(shuffle2.begin(), shuffle2.end(), m_e);

            // 1 - We compute crowding distance and non dominated rank for the current population
            auto fnds_res = fast_non_dominated_sorting(pop.get_f());
            auto ndf = std::get<0>(fnds_res); // non dominated fronts [[0,3,2],[1,5,6],[4],...]
            vector_double pop_cd(NP);         // crowding distances of the whole population
            auto ndr = std::get<3>(fnds_res); // non domination rank [0,1,0,0,2,1,1, ... ]
            for (const auto &front_idxs : ndf) {
                if (front_idxs.size() == 1u) { // handles the case where the front has collapsed to one point
                    pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
                } else {
                    if (front_idxs.size() == 2u) { // handles the case where the front has collapsed to one point
                        pop_cd[front_idxs[0]] = std::numeric_limits<double>::infinity();
                        pop_cd[front_idxs[1]] = std::numeric_limits<double>::infinity();
                    } else {
                        std::vector<vector_double> front;
                        for (auto idx : front_idxs) {
                            front.push_back(pop.get_f()[idx]);
                        }
                        auto cd = crowding_distance(front);
                        for (decltype(cd.size()) i = 0u; i < cd.size(); ++i) {
                            pop_cd[front_idxs[i]] = cd[i];
                        }
                    }
                }
            }

            // 3 - We then loop thorugh all individuals with increment 4 to select two pairs of parents that will
            // each create 2 new offspring
            for (decltype(NP) i = 0u; i < NP; i += 4) {
                // We create two offsprings using the shuffled list 1
                parent1_idx = tournament_selection(shuffle1[i], shuffle1[i + 1], ndr, pop_cd);
                parent2_idx = tournament_selection(shuffle1[i + 2], shuffle1[i + 3], ndr, pop_cd);
                crossover(child1, child2, parent1_idx, parent2_idx, pop);
                mutate(child1, pop);
                mutate(child2, pop);
                // we use prob to evaluate the fitness so
                // that its feval counter is correctly updated
                auto f1 = prob.fitness(child1);
                auto f2 = prob.fitness(child2);
                popnew.push_back(child1, f1);
                popnew.push_back(child2, f2);

                // We repeat with the shuffled list 2
                parent1_idx = tournament_selection(shuffle2[i], shuffle2[i + 1], ndr, pop_cd);
                parent2_idx = tournament_selection(shuffle2[i + 2], shuffle2[i + 3], ndr, pop_cd);
                crossover(child1, child2, parent1_idx, parent2_idx, pop);
                mutate(child1, pop);
                mutate(child2, pop);
                // we use prob to evaluate the fitness so
                // that its feval counter is correctly updated
                f1 = prob.fitness(child1);
                f2 = prob.fitness(child2);
                popnew.push_back(child1, f1);
                popnew.push_back(child2, f2);
            } // popnew now contains 2NP individuals

            // This method returns the sorted N best individuals in the population according to the crowded comparison
            // operator
            best_idx = select_best_N_mo(popnew.get_f(), NP);
            // We insert into the population
            for (population::size_type i = 0; i < NP; ++i) {
                pop.set_xf(i, popnew.get_x()[best_idx[i]], popnew.get_f()[best_idx[i]]);
            }
        } // end of main NSGAII loop
        return pop;
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
    /// Extra informations
    /**
     * Returns extra information on the algorithm.
     *
     * @return an <tt> std::string </tt> containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tGenerations: ", m_gen);
        stream(ss, "\n\tCrossover probability: ", m_cr);
        stream(ss, "\n\tDistribution index for crossover: ", m_eta_c);
        stream(ss, "\n\tMutation probability: ", m_m);
        stream(ss, "\n\tDistribution index for mutation: ", m_eta_m);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        return ss.str();
    }
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
        ar(m_gen, m_cr, m_eta_c, m_m, m_eta_m, m_e, m_seed, m_verbosity, m_log);
    }

private:
    vector_double::size_type tournament_selection(vector_double::size_type idx1, vector_double::size_type idx2,
                                                  const std::vector<vector_double::size_type> &non_domination_rank,
                                                  std::vector<double> &crowding_d) const
    {
        if (non_domination_rank[idx1] < non_domination_rank[idx2]) return idx1;
        if (non_domination_rank[idx1] > non_domination_rank[idx2]) return idx2;
        if (crowding_d[idx1] > crowding_d[idx2]) return idx1;
        if (crowding_d[idx1] < crowding_d[idx2]) return idx2;
        std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)
        return ((drng(m_e) > 0.5) ? idx1 : idx2);
    }
    void crossover(vector_double &child1, vector_double &child2, vector_double::size_type parent1_idx,
                   vector_double::size_type parent2_idx, const pagmo::population &pop) const
    {
        // Decision vector dimensions
        auto D = pop.get_problem().get_nx();
        auto Di = pop.get_problem().get_nix();
        auto Dc = pop.get_problem().get_ncx();
        // Problem bounds
        const auto bounds = pop.get_problem().get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        // Parents decision vectors
        vector_double parent1 = pop.get_x()[parent1_idx];
        vector_double parent2 = pop.get_x()[parent2_idx];
        // declarations
        double y1, y2, yl, yu, rand01, beta, alpha, betaq, c1, c2;
        vector_double::size_type site1, site2;
        // Initialize the child decision vectors
        child1 = parent1;
        child2 = parent2;
        // Random distributions
        std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)

        // This implements a Simulated Binary Crossover SBX and applies it to the non integer part of the decision
        // vector
        if (drng(m_e) <= m_cr) {
            for (decltype(Dc) i = 0u; i < Dc; i++) {
                if ((drng(m_e) <= 0.5) && (std::abs(parent1[i] - parent2[i])) > 1e-14 && lb[i] != ub[i]) {
                    if (parent1[i] < parent2[i]) {
                        y1 = parent1[i];
                        y2 = parent2[i];
                    } else {
                        y1 = parent2[i];
                        y2 = parent1[i];
                    }
                    yl = lb[i];
                    yu = ub[i];
                    rand01 = drng(m_e);
                    beta = 1. + (2. * (y1 - yl) / (y2 - y1));
                    alpha = 2. - std::pow(beta, -(m_eta_c + 1.));
                    if (rand01 <= (1. / alpha)) {
                        betaq = std::pow((rand01 * alpha), (1. / (m_eta_c + 1.)));
                    } else {
                        betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (m_eta_c + 1.)));
                    }
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));

                    beta = 1. + (2. * (yu - y2) / (y2 - y1));
                    alpha = 2. - std::pow(beta, -(m_eta_c + 1.));
                    if (rand01 <= (1. / alpha)) {
                        betaq = std::pow((rand01 * alpha), (1. / (m_eta_c + 1.)));
                    } else {
                        betaq = std::pow((1. / (2. - rand01 * alpha)), (1. / (m_eta_c + 1.)));
                    }
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                    if (c1 < lb[i]) c1 = lb[i];
                    if (c2 < lb[i]) c2 = lb[i];
                    if (c1 > ub[i]) c1 = ub[i];
                    if (c2 > ub[i]) c2 = ub[i];
                    if (drng(m_e) <= .5) {
                        child1[i] = c1;
                        child2[i] = c2;
                    } else {
                        child1[i] = c2;
                        child2[i] = c1;
                    }
                }
            }
        }
        // This implements two-point binary crossover and applies it to the integer part of the chromosome
        for (decltype(Dc) i = Dc; i < D; ++i) {
            // in this loop we are sure Di is at least 1
            std::uniform_int_distribution<vector_double::size_type> ra_num(0, Di - 1u);
            if (drng(m_e) <= m_cr) {
                site1 = ra_num(m_e);
                site2 = ra_num(m_e);
                if (site1 > site2) {
                    std::swap(site1, site2);
                }
                for (decltype(site1) j = 0u; j < site1; ++j) {
                    child1[j] = parent1[j];
                    child2[j] = parent2[j];
                }
                for (decltype(site2) j = site1; j < site2; ++j) {
                    child1[j] = parent2[j];
                    child2[j] = parent1[j];
                }
                for (decltype(Di) j = site2; j < Di; ++j) {
                    child1[j] = parent1[j];
                    child2[j] = parent2[j];
                }
            } else {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
            }
        }
    }
    void mutate(vector_double &child, const pagmo::population &pop) const
    {
        // Decision vector dimensions
        auto D = pop.get_problem().get_nx();
        auto Dc = pop.get_problem().get_ncx();
        // Problem bounds
        const auto bounds = pop.get_problem().get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        // declarations
        double rnd, delta1, delta2, mut_pow, deltaq;
        double y, yl, yu, val, xy;
        // Random distributions
        std::uniform_real_distribution<> drng(0., 1.); // to generate a number in [0, 1)

        // This implements the real polinomial mutation and applies it to the non integer part of the decision vector
        for (decltype(Dc) j = 0u; j < Dc; ++j) {
            if (drng(m_e) <= m_m && lb[j] != ub[j]) {
                y = child[j];
                yl = lb[j];
                yu = ub[j];
                delta1 = (y - yl) / (yu - yl);
                delta2 = (yu - y) / (yu - yl);
                rnd = drng(m_e);
                mut_pow = 1. / (m_eta_m + 1.);
                if (rnd <= 0.5) {
                    xy = 1. - delta1;
                    val = 2. * rnd + (1. - 2. * rnd) * (std::pow(xy, (m_eta_m + 1.)));
                    deltaq = std::pow(val, mut_pow) - 1.;
                } else {
                    xy = 1. - delta2;
                    val = 2. * (1. - rnd) + 2. * (rnd - 0.5) * (std::pow(xy, (m_eta_m + 1.)));
                    deltaq = 1. - (std::pow(val, mut_pow));
                }
                y = y + deltaq * (yu - yl);
                if (y < yl) y = yl;
                if (y > yu) y = yu;
                child[j] = y;
            }
        }

        // This implements the integer mutation for an individual
        for (decltype(D) j = Dc; j < D; ++j) {
            if (drng(m_e) < m_m) {
                // We need to draw a random integer in [lb, ub]. Since these are floats we
                // cannot use integer distributions without risking overflows, hence we use a real
                // distribution
                auto mutated = std::floor(uniform_real_from_range(lb[j], ub[j] + 1, m_e));
                child[j] = mutated;
            }
        }
    }

    unsigned int m_gen;
    double m_cr;
    double m_eta_c;
    double m_m;
    double m_eta_m;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::nsga2)

#endif
