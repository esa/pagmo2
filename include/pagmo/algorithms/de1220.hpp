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

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{

/// Static variables used in pagmo::de1220
template <typename T>
struct de1220_statics {
    /// Allowed mutation variants considered by default: {2u ,3u ,7u ,10u ,13u ,14u ,15u ,16u}
    static const std::vector<unsigned> allowed_variants;
};

template <typename T>
const std::vector<unsigned> de1220_statics<T>::allowed_variants = {2u, 3u, 7u, 10u, 13u, 14u, 15u, 16u};

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

class PAGMO_DLL_PUBLIC de1220
{
public:
    /// Single entry of the log (gen, fevals, best, F, CR, Variant, dx, df)
    typedef std::tuple<unsigned, unsigned long long, double, double, double, unsigned, double, double> log_line_type;
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
    de1220(unsigned gen = 1u, std::vector<unsigned> allowed_variants = de1220_statics<void>::allowed_variants,
           unsigned variant_adptv = 1u, double ftol = 1e-6, double xtol = 1e-6, bool memory = false,
           unsigned seed = pagmo::random_device::next());

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
        return "sa-DE1220: Self-adaptive Differential Evolution 1220";
    }

    // Extra info
    std::string get_extra_info() const;

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

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    unsigned m_gen;
    mutable vector_double m_F;
    mutable vector_double m_CR;
    mutable std::vector<unsigned> m_variant;
    std::vector<unsigned> m_allowed_variants;
    unsigned m_variant_adptv;
    double m_ftol;
    double m_xtol;
    bool m_memory;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::de1220)

#endif
