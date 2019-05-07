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

#ifndef PAGMO_ALGORITHMS_MBH_HPP
#define PAGMO_ALGORITHMS_MBH_HPP

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>

namespace pagmo
{

/// Monotonic Basin Hopping (generalized).
/**
 * \image html mbh.png "A schematic diagram illustrating the fitness landscape as seen by basin hopping" width=3cm
 *
 * Monotonic basin hopping, or simply, basin hopping, is an algorithm rooted in the idea of mapping
 * the objective function \f$f(\mathbf x_0)\f$ into the local minima found starting from \f$\mathbf x_0\f$.
 * This simple idea allows a substantial increase of efficiency in solving problems, such as the Lennard-Jones
 * cluster or the MGA-1DSM interplanetary trajectory problem that are conjectured to have a so-called
 * funnel structure.
 *
 * In pagmo we provide an original generalization of this concept resulting in a meta-algorithm that operates
 * on any pagmo::population using any suitable user-defined algorithm (UDA). When a population containing a single
 * individual is used and coupled with a local optimizer, the original method is recovered.
 * The pseudo code of our generalized version is:
 * @code{.unparsed}
 * > Create a pagmo::population
 * > Select a UDA
 * > Store best individual
 * > while i < stop_criteria
 * > > Perturb the population in a selected neighbourhood
 * > > Evolve the population using the algorithm
 * > > if the best individual is improved
 * > > > increment i
 * > > > update best individual
 * > > else
 * > > > i = 0
 * @endcode
 *
 * pagmo::mbh is a user-defined algorithm (UDA) that can be used to construct pagmo::algorithm objects.
 *
 * See: https://arxiv.org/pdf/cond-mat/9803344.pdf for the paper introducing the basin hopping idea for a Lennard-Jones
 * cluster optimization.
 */
class PAGMO_DLL_PUBLIC mbh
{
public:
    /// Single entry of the log (feval, best fitness, n. constraints violated, violation norm, trial).
    typedef std::tuple<unsigned long long, double, vector_double::size_type, double, unsigned> log_line_type;
    /// The log.
    typedef std::vector<log_line_type> log_type;
    // Default constructor.
    mbh();

private:
    // Enabler for the ctor from UDA or algorithm. In this case we allow construction from type algorithm.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<algorithm, T &&>::value, int>;
    void scalar_ctor_impl(double);
    void vector_ctor_impl(const vector_double &);

public:
    /// Constructor (scalar perturbation).
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``T`` can be used to construct a :cpp:class:`pagmo::algorithm`.
     *
     * \endverbatim
     *
     * This constructor will construct a monotonic basin hopping algorithm using a scalar perturbation.
     *
     * @param a a user-defined algorithm (UDA) or a pagmo::algorithm that will be used to construct the inner algorithm.
     * @param stop consecutive runs of the inner algorithm that need to
     * result in no improvement for pagmo::mbh to stop.
     * @param perturb the perturbation to be applied to each component
     * of the decision vector of the best population found when generating a new starting point.
     * These are defined relative to the corresponding bounds.
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws unspecified any exception thrown by the constructor of pagmo::algorithm.
     * @throws std::invalid_argument if \p perturb is not in the (0,1] range.
     */
    template <typename T, ctor_enabler<T> = 0>
    explicit mbh(T &&a, unsigned stop, double perturb, unsigned seed = pagmo::random_device::next())
        : m_algorithm(std::forward<T>(a)), m_stop(stop), m_perturb(1, perturb), m_e(seed), m_seed(seed), m_verbosity(0u)
    {
        scalar_ctor_impl(perturb);
    }
    /// Constructor (vector perturbation).
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    This constructor is enabled only if ``T``, after the removal of cv/reference qualifiers,
     *    is not :cpp:class:`pagmo::algorithm`.
     *
     * \endverbatim
     *
     * This constructor will construct a monotonic basin hopping algorithm using a vector perturbation.
     *
     * @param a a user-defined algorithm (UDA) or a pagmo::algorithm that will be used to construct the inner algorithm.
     * @param stop consecutive runs of the inner algorithm that need to
     * result in no improvement for pagmo::mbh to stop.
     * @param perturb a pagmo::vector_double with the perturbations to be applied to each component
     * of the decision vector of the best population found when generating a new starting point.
     * These are defined relative to the corresponding bounds.
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws unspecified any exception thrown by the constructor of pagmo::algorithm.
     * @throws std::invalid_argument if not all the elements of \p perturb are in the (0,1] range.
     */
    template <typename T, ctor_enabler<T> = 0>
    explicit mbh(T &&a, unsigned stop, vector_double perturb, unsigned seed = pagmo::random_device::next())
        : m_algorithm(std::forward<T>(a)), m_stop(stop), m_perturb(perturb), m_e(seed), m_seed(seed), m_verbosity(0u)
    {
        vector_ctor_impl(perturb);
    }
    // Evolve method.
    population evolve(population) const;
    // Set the seed.
    void set_seed(unsigned);
    /// Get the seed.
    /**
     * @return the seed controlling the algorithm's stochastic behaviour.
     */
    unsigned get_seed() const
    {
        return m_seed;
    }
    /// Set the algorithm verbosity.
    /**
     * This method will sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity,
     * - >0: will print and log one line at the end of each call to the inner algorithm.
     *
     * Example (verbosity 100):
     * @code
     * Fevals:          Best:      Violated:    Viol. Norm:         Trial:
     *     105        110.395              1      0.0259512              0 i
     *     211        110.395              1      0.0259512              1 i
     *     319        110.395              1      0.0259512              2 i
     *     422        110.514              1      0.0181383              0 i
     *     525         111.33              1      0.0149418              0 i
     *     628         111.33              1      0.0149418              1 i
     *     731         111.33              1      0.0149418              2 i
     *     834         111.33              1      0.0149418              3 i
     *     937         111.33              1      0.0149418              4 i
     *    1045         111.33              1      0.0149418              5 i
     * @endcode
     * \p Fevals is the number of fitness evaluations, \p Best is the objective function of the best
     * fitness currently in the population, \p Violated is the number of constraints currently violated
     * by the best solution, <tt>Viol. Norm</tt> is the norm of the violation (discounted already by the constraints
     * tolerance) and \p Trial is the trial number (which will determine the algorithm stop).
     * The small \p i appearing at the end of the line stands for "infeasible" and will disappear only
     * once \p Violated is 0.
     *
     * @param level verbosity level.
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    };
    /// Get the verbosity level.
    /**
     * @return the verbosity level.
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }
    /// Get the perturbation vector.
    /**
     * @return a const reference to the perturbation vector.
     */
    const vector_double &get_perturb() const
    {
        return m_perturb;
    }
    // Set the perturbation vector.
    void set_perturb(const vector_double &);
    // Algorithm's thread safety level.
    thread_safety get_thread_safety() const;
    /// Getter for the inner algorithm.
    /**
     * Returns a const reference to the inner pagmo::algorithm.
     *
     * @return a const reference to the inner pagmo::algorithm.
     */
    const algorithm &get_inner_algorithm() const
    {
        return m_algorithm;
    }
    /// Getter for the inner algorithm.
    /**
     * Returns a reference to the inner pagmo::algorithm.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    The ability to extract a non const reference is provided only in order to allow to call
     *    non-const methods on the internal :cpp:class:`pagmo::algorithm` instance. Assigning a new
     *    :cpp:class:`pagmo::algorithm` via this reference is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a reference to the inner pagmo::algorithm.
     */
    algorithm &get_inner_algorithm()
    {
        return m_algorithm;
    }
    /// Get log.
    /**
     * A log containing relevant quantities monitoring the last call to mbh::evolve(). Each element of the returned
     * <tt>std::vector</tt> is a mbh::log_line_type containing: \p Fevals, \p Best, \p Violated, <tt>Viol.Norm</tt> and
     * \p Trial as described in mbh::set_verbosity().
     *
     * @return an <tt>std::vector</tt> of mbh::log_line_type containing the logged values Fevals, Best,
     * Violated, Viol.Norm and Trial.
     */
    const log_type &get_log() const
    {
        return m_log;
    }
    /// Algorithm name.
    /**
     * @return a string containing the algorithm name.
     */
    std::string get_name() const
    {
        return "MBH: Monotonic Basin Hopping - Generalized";
    }
    // Extra info.
    std::string get_extra_info() const;
    // Object serialization.
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    algorithm m_algorithm;
    unsigned m_stop;
    // The member m_perturb is mutable as to allow to construct mbh also using a perturbation defined as a scalar
    // (in which case upon the first call to evolve it is expanded to the problem dimension)
    // and as a vector (in which case mbh will only operate on problem having the correct dimension)
    // While the use of "mutable" is not encouraged, in this case the alternative would be to have the user
    // construct the mbh algo passing one further parameter (the problem dmension) rather than having this determined
    // upon the first call to evolve.
    mutable std::vector<double> m_perturb;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::mbh)

#endif
