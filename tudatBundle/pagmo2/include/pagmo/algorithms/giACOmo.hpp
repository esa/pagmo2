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

#ifndef PAGMO_ALGORITHMS_giACOmo_HPP
#define PAGMO_ALGORITHMS_giACOmo_HPP

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
/// Extended ACO
/**
 * \image html ACO.jpg "The ACO flowchart" width=3cm [TO BE ADDED]

 * ACO is inspired by the natural mechanism with which real ant colonies forage food.
 * This algorithm has shown promising results in many trajectory optimization problems.
 * The first appearance of the algorithm happened in Dr. Marco Dorigo's thesis, in 1992.
 * ACO generates future generations of ants by using the a multi-kernel gaussian distribution
 * based on three parameters (i.e., pheromone values) which are computed depending on the quality
 * of each previous solution. The solutions are ranked through an oracle penalty method.
 *
 *
 * The version implemented in pagmo can be applied to box-bounded multiple-objective optimization.
 *
 * See:  M. Schlueter, et al. (2009). Extended ant colony optimization for non-convex
 * mixed integer non-linear programming. Computers & Operations Research.
 */
class giACOmo
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned int, unsigned long long, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
    * Constructs the ACO user defined algorithm.
    *
    * @param[in] gen Generations: number of generations to evolve.
    * @param[in] acc Accuracy parameter: for inequality and equality constraints .
    * @param[in] FSTOP Objective stopping criterion: when the objective value reaches this value, the algorithm is stopped [for multi-objective, this applies to the first obj. only].
    * @param[in] IMPSTOP Improvement stopping criterion: if a positive integer is assigned here, the algorithm will count the runs without improvements, if this number will exceed IMPSTOP value, the algorithm will be stopped.
    * @param[in] EVALSTOP Evaluation stopping criterion: same as previous one, but with function evaluations
    * @param[in] FOCUS Focus parameter: this parameter makes the search for the optimum greedier and more focused on local improvements (the higher the greedier). If the value is very high, the search is more focused around the current best solutions
    * @param[in] KER Kernel: number of solutions stored in the solution archive
    * @param[in] ORACLE Oracle parameter: this is the oracle parameter used in the penalty method
    * @param[in] PARETOMAX Max number of non-dominated solutions: this regulates the max number of Pareto points to be stored
    * @param[in] EPSILON Pareto precision: the smaller this parameter, the higher the chances to introduce a new solution in the Pareto front
    * @param seed seed used by the internal random number generator (default is random)
    * @throws std::invalid_argument if \p acc is not \f$ \in [0,1[\f$, \p FSTOP is not positive, \p IMPSTOP is not a
    * positive integer, \p EVALSTOP is not a positive integer, \p FOCUS is not \f$ \in [0,1[\f$, \p ANTS is not a positive integer,
    * \p KER is not a positive integer, \p ORACLE is not positive, \p PARETOMAX is not a positive integer, \p EPSILON is not \f$ \in [0,1[\f$
    */
    giACOmo(unsigned gen = 1u, double acc = 0.95, unsigned int FSTOP = 1, unsigned int IMPSTOP = 1, unsigned int EVALSTOP = 1,
          double FOCUS = 0.9,  unsigned int KER = 10, double ORACLE=1.0, unsigned int PARETOMAX = 10,
            double EPSILON = 0.9, unsigned seed = pagmo::random_device::next())
        : m_gen(gen), m_acc(acc), m_FSTOP(FSTOP), m_IMPSTOP(IMPSTOP), m_EVALSTOP(EVALSTOP), m_FOCUS(FOCUS),
          m_KER(KER), m_ORACLE(ORACLE), m_PARETOMAX(PARETOMAX), m_EPSILON(EPSILON), m_e(seed), m_seed(seed), m_verbosity(0u),
          m_log()
    {
        if (acc >= 1. || acc < 0.) {
            pagmo_throw(std::invalid_argument, "The accuracy parameter must be in the [0,1[ range, while a value of "
                                                   + std::to_string(acc) + " was detected");
        }
        if (FSTOP < 0.) {
            pagmo_throw(std::invalid_argument, "The objective stopping criterion must be in the [0,inf[ range, while a value of "
                                                   + std::to_string(FSTOP) + " was detected");
        }
        if (IMPSTOP < 0) {
            pagmo_throw(std::invalid_argument,
                        "The improvement stopping criterion must be in [0, inf[, while a value of "
                            + std::to_string(IMPSTOP) + " was detected");
        }
        if (EVALSTOP < 0) {
            pagmo_throw(std::invalid_argument,
                        "The evaluation stopping criterion must be in [0, inf[, while a value of "
                            + std::to_string(EVALSTOP) + " was detected");
        }

        if (FOCUS >= 1. || FOCUS < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The focus parameter must be in [0, inf[, while a value of "
                            + std::to_string(FOCUS) + " was detected");
        }

        if (KER < 0) {
            pagmo_throw(std::invalid_argument,
                        "The kernel parameter must be in [0, inf[, while a value of "
                            + std::to_string(KER) + " was detected");
        }

        if (ORACLE < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The oracle parameter must be in [0, inf[, while a value of "
                            + std::to_string(ORACLE) + " was detected");
        }

        if (PARETOMAX < 0) {
            pagmo_throw(std::invalid_argument,
                        "The max number of non-dominated solutions must be in [0, inf[, while a value of "
                            + std::to_string(PARETOMAX) + " was detected");
        }

        if (EPSILON >= 1. || EPSILON < 0.) {
            pagmo_throw(std::invalid_argument,
                        "The Pareto precision parameter must be in [0, inf[, while a value of "
                            + std::to_string(EPSILON) + " was detected");
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

        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;

        auto fevals0 = prob.get_fevals(); // discount for the fevals already made
        unsigned int count = 1u;          // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        // We start by checking that the problem is suitable for this
        // particular algorithm.

        if (!NP) {
            pagmo_throw(std::invalid_argument, get_name() + " cannot work on an empty population");
        }
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

        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Declarations


        // 0 - We initialize the first generation using a pseudo-random number generator:

        // Main ACO loop over generations:
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



            // 1 - Retrieve fitness values and use it to update and sort the solution archive


            // 2 - Compute pheromone values


            // 3 - Generate new generation of ants using the evolutionary operator



        } // end of main ACO loop
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
        return "giACOmo:";
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
        stream(ss, "\n\tAccuracy parameter: ", m_acc);
        stream(ss, "\n\tObjective stopping criterion: ", m_FSTOP);
        stream(ss, "\n\tImprovement stopping criterion: ", m_IMPSTOP);
        stream(ss, "\n\tEvaluation stopping criterion: ", m_EVALSTOP);
        stream(ss, "\n\tFocus parameter: ", m_FOCUS);
        stream(ss, "\n\tKernel: ", m_KER);
        stream(ss, "\n\tOracle parameter: ", m_ORACLE);
        stream(ss, "\n\tMax number of non-dominated solutions: ", m_PARETOMAX);
        stream(ss, "\n\tPareto precision: ", m_EPSILON);
        stream(ss, "\n\tDistribution index for mutation: ", m_EPSILON);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);

        return ss.str();
    }
    /// Get log

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
        ar(m_gen, m_acc, m_FSTOP, m_IMPSTOP, m_EVALSTOP, m_FOCUS, m_KER, m_ORACLE, m_PARETOMAX, m_EPSILON, m_e, m_seed, m_verbosity, m_log);
    }

private:


    unsigned int m_gen;
    double m_acc;
    int m_FSTOP;
    int m_IMPSTOP;
    int m_EVALSTOP;
    double m_FOCUS;
    int m_KER;
    double m_ORACLE;
    int m_PARETOMAX;
    double m_EPSILON;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;  
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::giACOmo)

#endif
