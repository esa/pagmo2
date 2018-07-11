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

#ifndef PAGMO_ALGORITHMS_DE1220_HPP
#define PAGMO_ALGORITHMS_DE1220_HPP

#include <iomanip>
#include <numeric> //std::iota
#include <random>
#include <sstream> //std::osstringstream
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

/// Static variables used in pagmo::de1220
template <typename T>
struct de1220_statics {
    /// Allowed mutation variants considered by default: {2u ,3u ,7u ,10u ,13u ,14u ,15u ,16u}
    static std::vector<unsigned int> allowed_variants;
};

template <typename T>
std::vector<unsigned int> de1220_statics<T>::allowed_variants = {2u, 3u, 7u, 10u, 13u, 14u, 15u, 16u};

/// A Differential Evolution Algorithm (1220, or pDE: our own DE flavour!!)
/**
 * \image html original.jpg "Our own DE flavour".
 *
 * Differential Evolution (pagmo::de, pagmo::sade) is one of the best meta-heuristics in PaGMO, so we
 * dared to propose our own algoritmic variant we call DE 1220 (a.k.a. pDE as in pagmo DE). Our variant
 * makes use of the pagmo::sade adaptation schemes for CR and F and adds self-adaptation for
 * the mutation variant. The only parameter left to be specified is thus population size.
 *
 * Similarly to what done in pagmo::sade for F and CR, in DE 1220 each
 * individual chromosome (index \f$i\f$) is augmented also with an integer \f$V_i\f$ that specifies
 * the mutation variant used to produce the next trial individual. Right before mutating the
 * chromosome the value of \f$V_i\f$ is adapted according to the equation:
 *
 * \f[
 * V_i =
 * \left\{\begin{array}{ll}
 * random & r_i < \tau \\
 * V_i & \mbox{otherwise}
 * \end{array}\right.
 * \f]
 *
 * where \f$\tau\f$ is set to be 0.1, \f$random\f$ selects a random mutation variant and \f$r_i\f$ is a random
 * uniformly distributed number in [0, 1]
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The feasibility correction, that is the correction applied to an allele when some mutation puts it outside
 *    the allowed box-bounds, is here done by creating a random number in the bounds.
 *
 * .. note::
 *
 *    The search range is defined relative to the box-bounds. Hence, unbounded problems
 *    will produce an error.
 *
 *
 * .. seealso::
 *
 *    :cpp:class:`pagmo::de`, :cpp:class:`pagmo::sade` For other available algorithms based on Differential Evolution
 *
 * \endverbatim
 *
 */

class de1220
{
public:
    /// Single entry of the log (gen, fevals, best, F, CR, Variant, dx, df)
    typedef std::tuple<unsigned int, unsigned long long, double, double, double, unsigned int, double, double>
        log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Constructs pDE (a.k.a. DE 1220)
     *
     * The same two self-adaptation variants used in pagmo::sade are used to self-adapt the
     * CR and F parameters:
     *
     * @code{.unparsed}
     * 1 - jDE (Brest et al.)                       2 - iDE (Elsayed at al.)
     * @endcode
     *
     * A subset of the following mutation variants is considered when adapting the mutation variant:
     *
     * @code{.unparsed}
     * 1 - best/1/exp                               2. - rand/1/exp
     * 3 - rand-to-best/1/exp                       4. - best/2/exp
     * 5 - rand/2/exp                               6. - best/1/bin
     * 7 - rand/1/bin                               8. - rand-to-best/1/bin
     * 9 - best/2/bin                               10. - rand/2/bin
     * 11. - rand/3/exp                             12. - rand/3/bin
     * 13. - best/3/exp                             14. - best/3/bin
     * 15. - rand-to-current/2/exp                  16. - rand-to-current/2/bin
     * 17. - rand-to-best-and-current/2/exp         18. - rand-to-best-and-current/2/bin
     * @endcode
     *
     * The first ten are the classical variants introduced in the orginal DE algorithm, the remaining ones are,
     * instead, introduced in the work by Elsayed et al.
     *
     * @param gen number of generations.
     * @param allowed_variants the subset of mutation variants to be considered (default is {2u ,3u ,7u ,10u ,13u
     ,14u ,15u ,16u})
     * @param variant_adptv parameter adaptation scheme to be used (one of 1..2)
     * @param ftol stopping criteria on the x tolerance (default is 1e-6)
     * @param xtol stopping criteria on the f tolerance (default is 1e-6)
     * @param memory when true the parameters CR anf F are not reset between successive calls to the evolve method
     * @param seed seed used by the internal random number generator (default is random)

     * @throws std::invalid_argument if \p variant_adptv is not in [0,1]
     * @throws std::invalid_argument if \p allowed_variants contains a number not in 1..18
     *
     * See: (jDE) - Brest, J., Greiner, S., Bošković, B., Mernik, M., & Zumer, V. (2006). Self-adapting control
     parameters in differential evolution: a comparative study on numerical benchmark problems. Evolutionary
     Computation, IEEE Transactions on, 10(6), 646-657. Chicago
     * See: (iDE) - Elsayed, S. M., Sarker, R. A., & Essam, D. L. (2011, June). Differential evolution with multiple
     strategies for solving CEC2011 real-world numerical optimization problems. In Evolutionary Computation (CEC), 2011
     IEEE Congress on (pp. 1041-1048). IEEE.
     */
    de1220(unsigned int gen = 1u, std::vector<unsigned int> allowed_variants = de1220_statics<void>::allowed_variants,
           unsigned int variant_adptv = 1u, double ftol = 1e-6, double xtol = 1e-6, bool memory = false,
           unsigned int seed = pagmo::random_device::next())
        : m_gen(gen), m_F(), m_CR(), m_variant(), m_allowed_variants(allowed_variants), m_variant_adptv(variant_adptv),
          m_ftol(ftol), m_xtol(xtol), m_memory(memory), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
        for (auto variant : allowed_variants) {
            if (variant < 1u || variant > 18u) {
                pagmo_throw(std::invalid_argument,
                            "All mutation variants considered must be in [1, .., 18], while a value of "
                                + std::to_string(variant) + " was detected.");
            }
        }
        if (variant_adptv < 1u || variant_adptv > 2u) {
            pagmo_throw(std::invalid_argument, "The variant for self-adaptation must be in [1,2], while a value of "
                                                   + std::to_string(variant_adptv) + " was detected.");
        }
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    /**
     * Evolves the population for a maximum number of generations, until one of
     * tolerances set on the population flatness (x_tol, f_tol) are met.
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if the problem is multi-objective or constrained or stochastic
     * @throws std::invalid_argument if the population size is not at least 7
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
        if (pop.size() < 7u) {
            pagmo_throw(std::invalid_argument, get_name() + " needs at least 7 individuals in the population, "
                                                   + std::to_string(pop.size()) + " detected");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();

        // Some vectors used during evolution are declared.
        vector_double tmp(dim);                              // contains the mutated candidate
        std::uniform_real_distribution<double> drng(0., 1.); // to generate a number in [0, 1)
        std::normal_distribution<double> n_dist(0., 1.);     // to generate a normally distributed number
        std::uniform_int_distribution<vector_double::size_type> c_idx(
            0u, dim - 1u); // to generate a random index in the chromosome
        std::uniform_int_distribution<vector_double::size_type> p_idx(0u, NP - 1u); // to generate a random index in pop
        std::uniform_int_distribution<vector_double::size_type> v_idx(0u, m_allowed_variants.size()
                                                                              - 1u); // to generate a random variant

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
        std::vector<vector_double::size_type> r(7); // indexes of 7 selected population members

        // Initialize the F and CR vectors
        if ((m_CR.size() != NP) || (m_F.size() != NP) || (m_variant.size() != NP) || (!m_memory)) {
            m_CR.resize(NP);
            m_F.resize(NP);
            m_variant.resize(NP);
            if (m_variant_adptv == 1u) {
                for (decltype(NP) i = 0u; i < NP; ++i) {
                    m_CR[i] = drng(m_e);
                    m_F[i] = drng(m_e) * 0.9 + 0.1;
                }
            } else if (m_variant_adptv == 2u) {
                for (decltype(NP) i = 0u; i < NP; ++i) {
                    m_CR[i] = n_dist(m_e) * 0.15 + 0.5;
                    m_F[i] = n_dist(m_e) * 0.15 + 0.5;
                }
            }
            for (auto &variant : m_variant) {
                variant = m_allowed_variants[v_idx(m_e)];
            }
        }
        // Initialize the global and iteration bests for F and CR
        double gbF = m_F[0];   // initialization to the 0 ind, will soon be forgotten
        double gbCR = m_CR[0]; // initialization to the 0 ind, will soon be forgotten
        unsigned int gbVariant = m_variant[0];
        double gbIterF = gbF;
        double gbIterCR = gbCR;
        unsigned int gbIterVariant;

        // We initialize the global best for F and CR as the first individual (this will soon be forgotten)

        // Main DE iterations
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // Start of the loop through the population
            for (decltype(NP) i = 0u; i < NP; ++i) {
                /*-----We select at random 5 indexes from the population---------------------------------*/
                std::vector<vector_double::size_type> idxs(NP);
                std::iota(idxs.begin(), idxs.end(), vector_double::size_type(0u));
                for (auto j = 0u; j < 7u; ++j) { // Durstenfeld's algorithm to select 7 indexes at random
                    auto idx = std::uniform_int_distribution<vector_double::size_type>(0u, NP - 1u - j)(m_e);
                    r[j] = idxs[idx];
                    std::swap(idxs[idx], idxs[NP - 1u - j]);
                }

                // Adapt amplification factor, crossover probability and mutation variant for DE 1220
                double F = 0., CR = 0.;
                unsigned int VARIANT = 0u;
                VARIANT = (drng(m_e) < 0.9) ? m_variant[i] : m_allowed_variants[v_idx(m_e)];
                if (m_variant_adptv == 1u) {
                    F = (drng(m_e) < 0.9) ? m_F[i] : drng(m_e) * 0.9 + 0.1;
                    CR = (drng(m_e) < 0.9) ? m_CR[i] : drng(m_e);
                }

                /*-------DE/best/1/exp--------------------------------------------------------------------*/
                if (VARIANT == 1u) {
                    if (m_variant_adptv == 2u) {
                        F = gbIterF + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]]);
                        CR = gbIterCR + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[r[2]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = gbIter[n] + F * (popold[r[1]][n] - popold[r[2]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }

                /*-------DE/rand/1/exp-------------------------------------------------------------------*/
                else if (VARIANT == 2u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]]);
                        CR = m_CR[r[0]] + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[r[2]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    decltype(dim) L = 0u;
                    do {
                        tmp[n] = popold[r[0]][n] + F * (popold[r[1]][n] - popold[r[2]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }
                /*-------DE/rand-to-best/1/exp-----------------------------------------------------------*/
                else if (VARIANT == 3u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[i] + n_dist(m_e) * 0.5 * (gbIterF - m_F[i])
                            + n_dist(m_e) * 0.5 * (m_F[r[0]] - m_F[r[1]]);
                        CR = m_CR[i] + n_dist(m_e) * 0.5 * (gbIterCR - m_CR[i])
                             + n_dist(m_e) * 0.5 * (m_CR[r[0]] - m_CR[r[1]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = tmp[n] + F * (gbIter[n] - tmp[n]) + F * (popold[r[0]][n] - popold[r[1]][n]);
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }
                /*-------DE/best/2/exp is another powerful variant worth trying--------------------------*/
                else if (VARIANT == 4u) {
                    if (m_variant_adptv == 2u) {
                        F = gbIterF + n_dist(m_e) * 0.5 * (m_F[r[0]] - m_F[r[1]])
                            + n_dist(m_e) * 0.5 * (m_F[r[2]] - m_F[r[3]]);
                        CR = gbIterCR + n_dist(m_e) * 0.5 * (m_CR[r[0]] - m_CR[r[1]])
                             + n_dist(m_e) * 0.5 * (m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = gbIter[n] + (popold[r[0]][n] - popold[r[1]][n]) * F
                                 + (popold[r[2]][n] - popold[r[3]][n]) * F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }
                /*-------DE/rand/2/exp seems to be a robust optimizer for many functions-------------------*/
                else if (VARIANT == 5u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[4]] + n_dist(m_e) * 0.5 * (m_F[r[0]] - m_F[r[1]])
                            + n_dist(m_e) * 0.5 * (m_F[r[2]] - m_F[r[3]]);
                        CR = m_CR[r[4]] + n_dist(m_e) * 0.5 * (m_CR[r[0]] - m_CR[r[1]])
                             + n_dist(m_e) * 0.5 * (m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = popold[r[4]][n] + (popold[r[0]][n] - popold[r[1]][n]) * F
                                 + (popold[r[2]][n] - popold[r[3]][n]) * F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }

                /*=======Essentially same strategies but BINOMIAL CROSSOVER===============================*/
                /*-------DE/best/1/bin--------------------------------------------------------------------*/
                else if (VARIANT == 6u) {
                    if (m_variant_adptv == 2u) {
                        F = gbIterF + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]]);
                        CR = gbIterCR + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[r[2]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = gbIter[n] + F * (popold[r[1]][n] - popold[r[2]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand/1/bin-------------------------------------------------------------------*/
                else if (VARIANT == 7u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]]);
                        CR = m_CR[r[0]] + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[r[2]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[0]][n] + F * (popold[r[1]][n] - popold[r[2]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand-to-best/1/bin-----------------------------------------------------------*/
                else if (VARIANT == 8u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[i] + n_dist(m_e) * 0.5 * (gbIterF - m_F[i])
                            + n_dist(m_e) * 0.5 * (m_F[r[0]] - m_F[r[1]]);
                        CR = m_CR[i] + n_dist(m_e) * 0.5 * (gbIterCR - m_CR[i])
                             + n_dist(m_e) * 0.5 * (m_CR[r[0]] - m_CR[r[1]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = tmp[n] + F * (gbIter[n] - tmp[n]) + F * (popold[r[0]][n] - popold[r[1]][n]);
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/best/2/bin--------------------------------------------------------------------*/
                else if (VARIANT == 9u) {
                    if (m_variant_adptv == 2u) {
                        F = gbIterF + n_dist(m_e) * 0.5 * (m_F[r[0]] - m_F[r[1]])
                            + n_dist(m_e) * 0.5 * (m_F[r[2]] - m_F[r[3]]);
                        CR = gbIterCR + n_dist(m_e) * 0.5 * (m_CR[r[0]] - m_CR[r[1]])
                             + n_dist(m_e) * 0.5 * (m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = gbIter[n] + (popold[r[0]][n] - popold[r[1]][n]) * F
                                     + (popold[r[2]][n] - popold[r[3]][n]) * F;
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand/2/bin--------------------------------------------------------------------*/
                else if (VARIANT == 10u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[4]] + n_dist(m_e) * 0.5 * (m_F[r[0]] - m_F[r[1]])
                            + n_dist(m_e) * 0.5 * (m_F[r[2]] - m_F[r[3]]);
                        CR = m_CR[r[4]] + n_dist(m_e) * 0.5 * (m_CR[r[0]] - m_CR[r[1]])
                             + n_dist(m_e) * 0.5 * (m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[4]][n] + (popold[r[0]][n] - popold[r[1]][n]) * F
                                     + (popold[r[2]][n] - popold[r[3]][n]) * F;
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand/3/exp --------------------------------------------------------------------*/
                else if (VARIANT == 11u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]])
                            + n_dist(m_e) * 0.5 * (m_F[r[3]] - m_F[r[4]]) + n_dist(m_e) * 0.5 * (m_F[r[5]] - m_F[r[6]]);
                        CR = m_CR[r[4]] + n_dist(m_e) * 0.5 * (m_CR[r[0]] + m_CR[r[1]] - m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = popold[r[0]][n] + (popold[r[1]][n] - popold[r[2]][n]) * F
                                 + (popold[r[3]][n] - popold[r[4]][n]) * F + (popold[r[5]][n] - popold[r[6]][n]) * F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }
                /*-------DE/rand/3/bin --------------------------------------------------------------------*/
                else if (VARIANT == 12u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]])
                            + n_dist(m_e) * 0.5 * (m_F[r[3]] - m_F[r[4]]) + n_dist(m_e) * 0.5 * (m_F[r[5]] - m_F[r[6]]);
                        CR = m_CR[r[4]] + n_dist(m_e) * 0.5 * (m_CR[r[0]] + m_CR[r[1]] - m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[0]][n] + (popold[r[1]][n] - popold[r[2]][n]) * F
                                     + (popold[r[3]][n] - popold[r[4]][n]) * F
                                     + (popold[r[5]][n] - popold[r[6]][n]) * F;
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/best/3/exp --------------------------------------------------------------------*/
                else if (VARIANT == 13u) {
                    if (m_variant_adptv == 2u) {
                        F = gbIterF + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]])
                            + n_dist(m_e) * 0.5 * (m_F[r[3]] - m_F[r[4]]) + n_dist(m_e) * 0.5 * (m_F[r[5]] - m_F[r[6]]);
                        CR = gbIterCR + n_dist(m_e) * 0.5 * (m_CR[r[0]] + m_CR[r[1]] - m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = gbIter[n] + (popold[r[1]][n] - popold[r[2]][n]) * F
                                 + (popold[r[3]][n] - popold[r[4]][n]) * F + (popold[r[5]][n] - popold[r[6]][n]) * F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }
                /*-------DE/best/3/bin --------------------------------------------------------------------*/
                else if (VARIANT == 14u) {
                    if (m_variant_adptv == 2u) {
                        F = gbIterF + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[r[2]])
                            + n_dist(m_e) * 0.5 * (m_F[r[3]] - m_F[r[4]]) + n_dist(m_e) * 0.5 * (m_F[r[5]] - m_F[r[6]]);
                        CR = gbIterCR + n_dist(m_e) * 0.5 * (m_CR[r[0]] + m_CR[r[1]] - m_CR[r[2]] - m_CR[r[3]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = gbIter[n] + (popold[r[1]][n] - popold[r[2]][n]) * F
                                     + (popold[r[3]][n] - popold[r[4]][n]) * F
                                     + (popold[r[5]][n] - popold[r[6]][n]) * F;
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand-to-current/2/exp --------------------------------------------------------------------*/
                else if (VARIANT == 15u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[i])
                            + n_dist(m_e) * 0.5 * (m_F[r[3]] - m_F[r[4]]);
                        CR = m_CR[r[0]] + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[i])
                             + n_dist(m_e) * 0.5 * (m_CR[r[3]] - m_CR[r[4]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = popold[r[0]][n] + (popold[r[1]][n] - popold[i][n]) * F
                                 + (popold[r[2]][n] - popold[r[3]][n]) * F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }
                /*-------DE/rand-to-current/2/bin --------------------------------------------------------------------*/
                else if (VARIANT == 16u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[i])
                            + n_dist(m_e) * 0.5 * (m_F[r[3]] - m_F[r[4]]);
                        CR = m_CR[r[0]] + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[i])
                             + n_dist(m_e) * 0.5 * (m_CR[r[3]] - m_CR[r[4]]);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[0]][n] + (popold[r[1]][n] - popold[i][n]) * F
                                     + (popold[r[2]][n] - popold[r[3]][n]) * F;
                        }
                        n = (n + 1u) % dim;
                    }
                }
                /*-------DE/rand-to-best-and-current/2/exp
                   --------------------------------------------------------------------*/
                else if (VARIANT == 17u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[i])
                            - n_dist(m_e) * 0.5 * (m_F[r[2]] - gbIterF);
                        CR = m_CR[r[0]] + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[i])
                             - n_dist(m_e) * 0.5 * (m_CR[r[3]] - gbIterCR);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    auto L = 0u;
                    do {
                        tmp[n] = popold[r[0]][n] + (popold[r[1]][n] - popold[i][n]) * F
                                 - (popold[r[2]][n] - gbIter[n]) * F;
                        n = (n + 1u) % dim;
                        ++L;
                    } while ((drng(m_e) < CR) && (L < dim));
                }
                /*-------DE/rand-to-best-and-current/2/bin
                   --------------------------------------------------------------------*/
                else if (VARIANT == 18u) {
                    if (m_variant_adptv == 2u) {
                        F = m_F[r[0]] + n_dist(m_e) * 0.5 * (m_F[r[1]] - m_F[i])
                            - n_dist(m_e) * 0.5 * (m_F[r[2]] - gbIterF);
                        CR = m_CR[r[0]] + n_dist(m_e) * 0.5 * (m_CR[r[1]] - m_CR[i])
                             - n_dist(m_e) * 0.5 * (m_CR[r[3]] - gbIterCR);
                    }
                    tmp = popold[i];
                    auto n = c_idx(m_e);
                    for (decltype(dim) L = 0u; L < dim; ++L) {   /* perform Dc binomial trials */
                        if ((drng(m_e) < CR) || L + 1u == dim) { /* change at least one parameter */
                            tmp[n] = popold[r[0]][n] + (popold[r[1]][n] - popold[i][n]) * F
                                     - (popold[r[2]][n] - gbIter[n]) * F;
                        }
                        n = (n + 1u) % dim;
                    }
                }

                /*==Trial mutation now in tmp. force feasibility and see how good this choice really was.==*/
                // a) feasibility
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    if ((tmp[j] < lb[j]) || (tmp[j] > ub[j])) {
                        tmp[j] = uniform_real_from_range(lb[j], ub[j], m_e);
                    }
                }
                // b) how good?
                auto newfitness = prob.fitness(tmp); /* Evaluates tmp[] */
                if (newfitness[0] <= fit[i][0]) {    /* improved objective function value ? */
                    fit[i] = newfitness;
                    popnew[i] = tmp;
                    // updates the individual in pop (avoiding to recompute the objective function)
                    pop.set_xf(i, popnew[i], newfitness);
                    // Update the adapted parameters
                    m_CR[i] = CR;
                    m_F[i] = F;
                    m_variant[i] = VARIANT;

                    if (newfitness[0] <= gbfit[0]) {
                        /* if so...*/
                        gbfit = newfitness; /* reset gbfit to new low...*/
                        gbX = popnew[i];
                        gbF = F;   /* these were forgotten in PaGMOlegacy */
                        gbCR = CR; /* these were forgotten in PaGMOlegacy */
                        gbVariant = VARIANT;
                    }
                } else {
                    popnew[i] = popold[i];
                }
            } // End of one generation
            /* Save best population member of current iteration */
            gbIter = gbX;
            gbIterF = gbF;
            gbIterCR = gbCR;
            gbIterVariant = gbVariant; // the gbIterVariant is not really needed and is only kept for consistency
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
            if (df < m_ftol) {
                if (m_verbosity > 0u) {
                    std::cout << "Exit condition -- ftol < " << m_ftol << std::endl;
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
                              "Best:", std::setw(15), "F:", std::setw(15), "CR:", std::setw(15),
                              "Variant:", std::setw(15), "dx:", std::setw(15), std::setw(15), "df:", '\n');
                    }
                    print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                          pop.get_f()[best_idx][0], std::setw(15), gbIterF, std::setw(15), gbIterCR, std::setw(15),
                          gbIterVariant, std::setw(15), dx, std::setw(15), df, '\n');
                    ++count;
                    // Logs
                    m_log.emplace_back(gen, prob.get_fevals() - fevals0, pop.get_f()[best_idx][0], gbIterF, gbIterCR,
                                       gbIterVariant, dx, df);
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
     * Example (verbosity 1):
     * @code{.unparsed}
     * Gen:        Fevals:          Best:             F:            CR:       Variant:            dx:            df:
    *     1             15        45.4245       0.480391       0.567908              4        10.9413        35061.1
    *     2             30        45.4245       0.480391       0.567908              4        10.9413        35061.1
    *     3             45        45.4245       0.480391       0.567908              4        10.9413        35061.1
    *     4             60        6.55036       0.194324      0.0732594              6        9.35874        4105.24
    *     5             75        6.55036       0.194324      0.0732594              6        6.57553         3558.4
    *     6             90        2.43304       0.448999       0.678681             14        3.71972        1026.26
    *     7            105        2.43304       0.448999       0.678681             14        11.3925        820.816
    *     8            120        1.61794       0.194324      0.0732594              6        11.0693        821.631
    *     9            135        1.61794       0.194324      0.0732594              6        11.0693        821.631
    *    10            150        1.61794       0.194324      0.0732594              6        11.0693        821.631
    *    11            165       0.643149       0.388876       0.680573              7        11.2983        822.606


     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness
     * function currently in the population, F is the F used to create the best so far, CR
     * the CR used to create the best so far, Variant is the mutation variant used to create the best so far,
     * dx is the population flatness evaluated as the distance between
     * the decisions vector of the best and of the worst individual and df is the population flatness evaluated
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
        return "sa-DE1220: Self-adaptive Differential Evolution 1220";
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
        stream(ss, "\n\tAllowed variants: ", m_allowed_variants);
        stream(ss, "\n\tSelf adaptation variant: ", m_variant_adptv);
        stream(ss, "\n\tStopping xtol: ", m_xtol);
        stream(ss, "\n\tStopping ftol: ", m_ftol);
        stream(ss, "\n\tMemory: ", m_memory);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        stream(ss, "\n\tSeed: ", m_seed);
        return ss.str();
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a de1220::log_line_type containing: Gen, Fevals, Best, F, CR, Variant, dx, df as
     * described
     * in de1220::set_verbosity
     * @return an <tt>std::vector</tt> of de1220::log_line_type containing the logged values Gen, Fevals, Best, F, CR,
     * Variant, dx, df
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
        ar(m_gen, m_F, m_CR, m_allowed_variants, m_variant_adptv, m_ftol, m_xtol, m_memory, m_e, m_seed, m_verbosity,
           m_log);
    }

private:
    unsigned int m_gen;
    mutable vector_double m_F;
    mutable vector_double m_CR;
    mutable std::vector<unsigned int> m_variant;
    std::vector<unsigned int> m_allowed_variants;
    unsigned int m_variant_adptv;
    double m_ftol;
    double m_xtol;
    bool m_memory;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::de1220)

#endif
