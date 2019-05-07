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

#ifndef PAGMO_ALGORITHMS_SADE_HPP
#define PAGMO_ALGORITHMS_SADE_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{
/// Self-adaptive Differential Evolution Algorithm
/**
 * \image html adaptation.jpg "Adapt already!".
 *
 * Two different variants of the Differential Evolution algorithm exploiting the idea of self-adaptation.
 *
 * The original Differential Evolution algorithm (pagmo::de) can be significantly improved introducing the
 * idea of parameter self-adaptation. Many different proposals have been made to self-adapt both the CR and the F
 * parameters
 * of the original differential evolution algorithm. In PaGMO we implement two different mechanisms we found effective.
 * The first one, proposed by Brest et al., does not make use of the DE operators to produce new
 * values for F and CR and, strictly speaking, is thus not self-adaptation, rather parameter control.
 * The resulting DE variant is often referred to as jDE. The second variant
 * here implemented is inspired by the ideas introduced by Elsayed et al. and uses a variaton of the selected DE
 * operator to produce new
 * CR anf F parameters for each individual. We refer to this variant as to iDE.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from pagmo::sade is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * .. warning::
 *
 *    The algorithm referred to as SaDE in the literature is not the algorithm implemented in pagmo. We
 *    use the name sade to indicate, generically, self-adaptation in a differential evolution algorithm
 *
 * .. note::
 *
 *    The feasibility correction, that is the correction applied to an allele when some mutation puts it outside
 *    the allowed box-bounds, is here done by creating a random number in the bounds.
 *
 * .. seealso::
 *
 *    (jDE) - Brest, J., Greiner, S., Bošković, B., Mernik, M., & Zumer, V. (2006). Self-adapting control parameters
 *    in differential evolution: a comparative study on numerical benchmark problems. Evolutionary Computation, IEEE
 *    Transactions on, 10(6), 646-657. Chicago
 *
 * .. seealso::
 *
 *    (iDE) - Elsayed, S. M., Sarker, R. A., & Essam, D. L. (2011, June). Differential evolution with multiple
 *    strategies for solving CEC2011 real-world numerical optimization problems. In Evolutionary Computation (CEC), 2011
 *    IEEE Congress on (pp. 1041-1048). IEEE.
 * \endverbatim
 *
 */
class PAGMO_DLL_PUBLIC sade
{
public:
    /// Single entry of the log (gen, fevals, best, F, CR, dx, df)
    typedef std::tuple<unsigned, unsigned long long, double, double, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor.
    /**
     * Constructs self-adaptive differential evolution
     *
     * Two self-adaptation variants are available to control the F and CR parameters:
     *
     * @code{.unparsed}
     * 1 - jDE (Brest et al.)                       2 - iDE (Elsayed at al.)
     * @endcode
     *
     * The following variants are available to produce a mutant vector:
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
     * The first ten are the classical mutation variants introduced in the orginal DE algorithm, the remaining ones are,
     * instead, considered in the work by Elsayed et al.
     *
     * @param gen number of generations.
     * @param variant mutation variant (dafault variant is 2: /rand/1/exp)
     * @param variant_adptv F and CR parameter adaptation scheme to be used (one of 1..2)
     * @param ftol stopping criteria on the x tolerance (default is 1e-6)
     * @param xtol stopping criteria on the f tolerance (default is 1e-6)
     * @param memory when true the adapted parameters CR anf F are not reset between successive calls to the evolve
     method
     * @param seed seed used by the internal random number generator (default is random)
     * @throws std::invalid_argument if \p variant_adptv is not one of 0,1
     * @throws std::invalid_argument if variant is not one of 1, .., 18
     */
    sade(unsigned gen = 1u, unsigned variant = 2u, unsigned variant_adptv = 1u, double ftol = 1e-6, double xtol = 1e-6,
         bool memory = false, unsigned seed = pagmo::random_device::next());

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
     * Gen:        Fevals:          Best:             F:            CR:            dx:            df:
     *  301           4515       0.668472       0.374983       0.502932    0.000276682    0.000388866
     *  302           4530       0.668472       0.374983       0.502932    0.000213271     0.00020986
     *  303           4545       0.668426       0.598243       0.234825    0.000167061    0.000186339
     *  304           4560       0.668426       0.598243       0.234825    0.000217549    0.000144896
     *  305           4575       0.668339       0.807236       0.863048    0.000192539    0.000232005
     *  306           4590       0.668339       0.807236       0.863048    0.000143711    0.000229041
     *  307           4605       0.668307       0.374983       0.820731    0.000163919    0.000245393

     * @endcode
     * Gen, is the generation number, Fevals the number of function evaluation used, Best is the best fitness
     * function currently in the population, F is the F used to create the best so far, CR
     * the CR used to create the best so far, dx is the population flatness evaluated as the distance between
     * the decisions vector of the best and of the worst individual and df is the population flatness evaluated
     * as the distance between the fitness of the best and of the worst individual.
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

    /// Gets the generations
    /**
     * @return the number of generations to evolve for
     */
    unsigned get_gen() const
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
        return "saDE: Self-adaptive Differential Evolution";
    }

    // Extra info
    std::string get_extra_info() const;

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a sade::log_line_type containing: Gen, Fevals, Best, F, CR, dx, df as described
     * in sade::set_verbosity
     * @return an <tt>std::vector</tt> of sade::log_line_type containing the logged values Gen, Fevals, Best, F, CR,
     * dx, df
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    unsigned m_gen;
    mutable vector_double m_F;
    mutable vector_double m_CR;
    unsigned m_variant;
    unsigned m_variant_adptv;
    double m_Ftol;
    double m_xtol;
    bool m_memory;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::sade)

#endif
