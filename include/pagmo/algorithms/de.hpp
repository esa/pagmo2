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

#ifndef PAGMO_ALGORITHMS_DE_HPP
#define PAGMO_ALGORITHMS_DE_HPP

#include <iomanip>
#include <numeric> //std::iota
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility> //std::swap

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{
/// Differential Evolution Algorithm
/**
 * \image html de.png "Differential Evolution block diagram."
 *
 * Differential Evolution is an heuristic optimizer developed by Rainer Storn and Kenneth Price.
 *
 * ''A breakthrough happened, when Ken came up with the idea of using vector differences for perturbing
 * the vector population. Since this seminal idea a lively discussion between Ken and Rainer and endless
 * ruminations and computer simulations on both parts yielded many substantial improvements which
 * make DE the versatile and robust tool it is today'' (from the official web pages....)
 *
 * The implementation provided for PaGMO is based on the code provided in the official
 * DE web site. pagmo::de is suitable for box-constrained single-objective continuous optimization.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The feasibility correction, that is the correction applied to an allele when some mutation puts it outside
 *    the allowed box-bounds, is here done by creating a random number in the bounds.
 *
 * .. seealso::
 *
 *    The official DE web site: http://www1.icsi.berkeley.edu/~storn/code.html
 *
 *    The paper that introduces Differential Evolution https://link.springer.com/article/10.1023%2FA%3A1008202821328
 *
 * \endverbatim
 */
class de
{
public:
    /// Single entry of the log (gen, fevals, best, dx, df)
    typedef std::tuple<unsigned int, unsigned long long, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Constructs de
     *
     * The following variants (mutation variants) are available to create a new candidate individual:
     * @code{.unparsed}
     * 1 - best/1/exp                               2. - rand/1/exp
     * 3 - rand-to-best/1/exp                       4. - best/2/exp
     * 5 - rand/2/exp                               6. - best/1/bin
     * 7 - rand/1/bin                               8. - rand-to-best/1/bin
     * 9 - best/2/bin                               10. - rand/2/bin
     * @endcode
     *
     * @param gen number of generations.
     * @param F weight coefficient (dafault value is 0.8)
     * @param CR crossover probability (dafault value is 0.9)
     * @param variant mutation variant (dafault variant is 2: /rand/1/exp)
     * @param ftol stopping criteria on the x tolerance (default is 1e-6)
     * @param xtol stopping criteria on the f tolerance (default is 1e-6)
     * @param seed seed used by the internal random number generator (default is random)

     * @throws std::invalid_argument if F, CR are not in [0,1]
     * @throws std::invalid_argument if variant is not one of 1 .. 10
     */
    de(unsigned int gen = 1u, double F = 0.8, double CR = 0.9, unsigned int variant = 2u, double ftol = 1e-6,
       double xtol = 1e-6, unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_F(F), m_CR(CR), m_variant(variant), m_Ftol(ftol), m_xtol(xtol), m_e(seed), m_seed(seed),
          m_verbosity(0u), m_log()
    {
        if (variant < 1u || variant > 10u) {
            pagmo_throw(std::invalid_argument,
                        "The Differential Evolution variant must be in [1, .., 10], while a value of "
                            + std::to_string(variant) + " was detected.");
        }
        if (CR < 0. || F < 0. || CR > 1. || F > 1.) {
            pagmo_throw(std::invalid_argument, "The F and CR parameters must be in the [0,1] range");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     *
     * Evolves the population for a maximum number of generations, until one of
     * tolerances set on the population flatness (x_tol, f_tol) are met.
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained or stochastic
     * @throws std::invalid_argument if the population size is not at least 5
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto NP = pop.size();
        auto prob_f_dimension = prob.get_nf();
        auto fevals0 = prob.get_fevals(); // disount for the already made fevals
        unsigned int count = 1u;          // regulates the screen output

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
        if (pop.size() < 5u) {
            pagmo_throw(std::invalid_argument, get_name() + " needs at least 5 individuals in the population, "
                                                   + std::to_string(pop.size()) + " detected");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Some vectors used during evolution are declared.
        vector_double tmp(dim);                              // contains the mutated candidate
        std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)
        std::uniform_int_distribution<vector_double::size_type> c_idx(
            0u, dim - 1u); // to generate a random index for the chromosome

        // We extract from pop the chromosomes and fitness associated
        auto popold = pop.get_x();
        auto fit = pop.get_f();
        auto popnew = popold;

        // Initialise the global bests
        auto best_idx = pop.best_idx();
        vector_double::size_type worst_idx = 0u;
        auto gbX = popnew[best_idx];
        auto gbfit = fit[best_idx];
        // the best decision vector of a generation
        auto gbIter = gbX;
        std::vector<vector_double::size_type> r(5); // indexes of 5 selected population members

        // Main DE iterations
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // Start of the loop through the population
            for (decltype(NP) i = 0u; i < NP; ++i) {
                /*-----We select at random 5 indexes from the population---------------------------------*/
                std::vector<vector_double::size_type> idxs(NP);
                std::iota(idxs.begin(), idxs.end(), vector_double::size_type(0u));
                for (auto j = 0u; j < 5u; ++j) { // Durstenfeld's algorithm to select 5 indexes at random
                    auto idx = std::uniform_int_distribution<vector_double::size_type>(0u, NP - 1u - j)(m_e);
                    r[j] = idxs[idx];
                    std::swap(idxs[idx], idxs[NP - 1u - j]);
                }

                /*-------DE/best/1/exp--------------------------------------------------------------------*/
                /*-------The oldest DE variant but still not bad. However, we have found several---------*/
                /*-------optimization problems where misconvergence occurs.-------------------------------*/
                if (m_variant == 1u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = gbIter[n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }

                /*-------DE/rand/1/exp-------------------------------------------------------------------*/
                /*-------This is one of my favourite strategies. It works especially well when the-------*/
                /*-------"gbIter[]"-schemes experience misconvergence. Try e.g. m_F=0.7 and m_CR=0.5---------*/
                /*-------as a first guess.---------------------------------------------------------------*/
                else if (m_variant == 2u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    decltype(dim) L = 0u;
                    do {
                        tmp[n] = popold[r[0]][n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }
                /*-------DE/rand-to-best/1/exp-----------------------------------------------------------*/
                /*-------This variant seems to be one of the best strategies. Try m_F=0.85 and m_CR=1.------*/
                /*-------If you get misconvergence try to increase NP. If this doesn't help you----------*/
                /*-------should play around with all three control variables.----------------------------*/
                else if (m_variant == 3u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = tmp[n] + m_F * (gbIter[n] - tmp[n]) + m_F * (popold[r[0]][n] - popold[r[1]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }
                /*-------DE/best/2/exp is another powerful variant worth trying--------------------------*/
                else if (m_variant == 4u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n]
                            = gbIter[n] + (popold[r[0]][n] + popold[r[1]][n] - popold[r[2]][n] - popold[r[3]][n]) * m_F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }
                /*-------DE/rand/2/exp seems to be a robust optimizer for many functions-------------------*/
                else if (m_variant == 5u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = popold[r[4]][n]
                                 + (popold[r[0]][n] + popold[r[1]][n] - popold[r[2]][n] - popold[r[3]][n]) * m_F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < m_CR) && (L < dim));
                }

                /*=======Essentially same strategies but BINOMIAL CROSSOVER===============================*/
                /*-------DE/best/1/bin--------------------------------------------------------------------*/
                else if (m_variant == 6u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {     /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = gbIter[n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand/1/bin-------------------------------------------------------------------*/
                else if (m_variant == 7u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {     /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[0]][n] + m_F * (popold[r[1]][n] - popold[r[2]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand-to-best/1/bin-----------------------------------------------------------*/
                else if (m_variant == 8u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {     /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = tmp[n] + m_F * (gbIter[n] - tmp[n]) + m_F * (popold[r[0]][n] - popold[r[1]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/best/2/bin--------------------------------------------------------------------*/
                else if (m_variant == 9u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {     /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = gbIter[n]
                                     + (popold[r[0]][n] + popold[r[1]][n] - popold[r[2]][n] - popold[r[3]][n]) * m_F;
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand/2/bin--------------------------------------------------------------------*/
                else if (m_variant == 10u) {
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {     /* perform Dc binomial trials */
                        if ((drng(m_e) < m_CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[4]][n]
                                     + (popold[r[0]][n] + popold[r[1]][n] - popold[r[2]][n] - popold[r[3]][n]) * m_F;
                        }
                        n = (n + 1u) % dim;
                    }
                }

                // Trial mutation now in tmp. force feasibility and see how good this choice really was.
                // a) feasibility
                // detail::force_bounds_reflection(tmp, lb, ub); // TODO: check if this choice is better
                detail::force_bounds_random(tmp, lb, ub, m_e);
                // b) how good?
                auto newfitness = prob.fitness(tmp); /* Evaluates tmp[] */
                if (newfitness[0] <= fit[i][0]) {    /* improved objective function value ? */
                    fit[i] = newfitness;
                    popnew[i] = tmp;
                    // updates the individual in pop (avoiding to recompute the objective function)
                    pop.set_xf(i, popnew[i], newfitness);

                    if (newfitness[0] <= gbfit[0]) {
                        /* if so...*/
                        gbfit = newfitness; /* reset gbfit to new low...*/
                        gbX = popnew[i];
                    }
                } else {
                    popnew[i] = popold[i];
                }
            } // End of one generation
            /* Save best population member of current iteration */
            gbIter = gbX;
            /* swap population arrays. New generation becomes old one */
            std::swap(popold, popnew);

            // Check the exit conditions
            double dx = 0., df = 0.;
            best_idx = pop.best_idx();
            worst_idx = pop.worst_idx();
            for (decltype(dim) i = 0u; i < dim; ++i) {
                dx += std::abs(pop.get_x()[worst_idx][i] - pop.get_x()[best_idx][i]);
            }
            if (dx < m_xtol) {
                if (m_verbosity > 0u) {
                    std::cout << "Exit condition -- xtol < " << m_xtol << std::endl;
                }
                return pop;
            }

            df = std::abs(pop.get_f()[worst_idx][0] - pop.get_f()[best_idx][0]);
            if (df < m_Ftol) {
                if (m_verbosity > 0u) {
                    std::cout << "Exit condition -- ftol < " << m_Ftol << std::endl;
                }
                return pop;
            }

            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    best_idx = pop.best_idx();
                    worst_idx = pop.worst_idx();
                    dx = 0.;
                    // The population flattness in chromosome
                    for (decltype(dim) i = 0u; i < dim; ++i) {
                        dx += std::abs(pop.get_x()[worst_idx][i] - pop.get_x()[best_idx][i]);
                    }
                    // The population flattness in fitness
                    df = std::abs(pop.get_f()[worst_idx][0] - pop.get_f()[best_idx][0]);
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                              "Best:", std::setw(15), "dx:", std::setw(15), "df:", '\n');
                    }
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                          pop.get_f()[best_idx][0], std::setw(15), dx, std::setw(15), df, '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(gen, prob.get_fevals() - fevals0, pop.get_f()[best_idx][0], dx, df);
                }
            }
        } // end main DE iterations
        if (m_verbosity) {
            std::cout << "Exit condition -- generations = " << m_gen << std::endl;
        }
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
     * Example (verbosity 100):
     * @code{.unparsed}
     * Gen:        Fevals:          Best:            dx:            df:
     * 5001         100020    3.62028e-05      0.0396687      0.0002866
     * 5101         102020    1.16784e-05      0.0473027    0.000249057
     * 5201         104020    1.07883e-05      0.0455471    0.000243651
     * 5301         106020    6.05099e-06      0.0268876    0.000103512
     * 5401         108020    3.60664e-06      0.0230468    5.78161e-05
     * 5501         110020     1.7188e-06      0.0141655    2.25688e-05
     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness
     * function currently in the population, dx is the population flatness evaluated as the distance between
     * the decisions vector of the best and of the worst individual, df is the population flatness evaluated
     * as the distance between the fitness of the best and of the worst individual.
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
        return "DE: Differential Evolution";
    }
    /// Extra informations
    /**
     * One of the optional methods of any user-defined algorithm (UDA).
     *
     * @return a string containing extra informations on the algorithm
     */
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) + "\n\tParameter F: " + std::to_string(m_F)
               + "\n\tParameter CR: " + std::to_string(m_CR) + "\n\tVariant: " + std::to_string(m_variant)
               + "\n\tStopping xtol: " + std::to_string(m_xtol) + "\n\tStopping ftol: " + std::to_string(m_Ftol)
               + "\n\tVerbosity: " + std::to_string(m_verbosity) + "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a de::log_line_type containing: Gen, Fevals, Best, dx, df as described
     * in de::set_verbosity
     * @return an <tt>std::vector</tt> of de::log_line_type containing the logged values Gen, Fevals, Best, dx, df
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
        ar(m_gen, m_F, m_CR, m_variant, m_Ftol, m_xtol, m_e, m_seed, m_verbosity, m_log);
    }

private:
    unsigned int m_gen;
    double m_F;
    double m_CR;
    unsigned int m_variant;
    double m_Ftol;
    double m_xtol;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::de)

#endif
