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

#ifndef PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP
#define PAGMO_ALGORITHMS_CSTRS_SELF_ADAPTIVE_HPP

#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{
namespace detail
{

// Constrainted self adaptive udp
/**
 * Implements a udp that wraps a population and results in self adaptive constraints handling.
 *
 * The key idea of this constraint handling technique is to represent the constraint violation by one single
 * infeasibility measure, and to adapt dynamically the penalization of infeasible solutions. As the penalization process
 * depends on a given population, a method to update the penalties to a new population is provided.
 *
 * @see Farmani R., & Wright, J. A. (2003). Self-adaptive fitness formulation for constrained optimization.
 * Evolutionary Computation, IEEE Transactions on, 7(5), 445-455 for the paper introducing the method.
 *
 */
// TODO this UDP has a series of problems, some of which are summarized
// in these reports:
// https://github.com/esa/pagmo2/issues/270
// https://github.com/esa/pagmo2/issues/269
//
// The UDP does not correctly advertises its own thread safety
// level (which depends on the thread safety level of the internal pop,
// but cannot currently be larger than basic, due to the mutable cache).
// Also, the UDP is not serializable, which will be an issue if the
// cstrs internal uda requires serialization.
//
// Proposals to start fixing the UDP:
// - don't store a pointer to the pop, rather a copy (this allows
//   for trivial serialization). Impact to be understood;
// - properly declare the thread safety level;
// - the cache can be an *optional* speed boost: if, in cstrs,
//   we cannot locate a decision vector in the cache (meaning that
//   the UDA operated on a copy of the original input problem), just re-evaluate
//   the dv instead of asserting failure.
struct PAGMO_DLL_PUBLIC penalized_udp {
    // Unused default constructor to please the is_udp type trait
    penalized_udp()
    {
        assert(false);
    }

    // Constructs the udp. At construction all member get initialized calling update().
    penalized_udp(population &);

    // The bounds are unchanged
    std::pair<vector_double, vector_double> get_bounds() const;

    // The fitness computation
    vector_double fitness(const vector_double &) const;

    // Call to this method updates all the members that are used to penalize the objective function
    // As the penalization algorithm depends heavily on the ref population this method takes care of
    // updating the necessary information. It also builds the hash map used to avoid unecessary fitness
    // evaluations. We exclude this method from the test as all of its corner cases are difficult to trigger
    // and test for correctness
    PAGMO_DLL_LOCAL void update();

    // Computes c_max holding the maximum value of the violation of each constraint in the whole ref population
    PAGMO_DLL_LOCAL void compute_c_max();

    // Assuming the various data member contain useful information, this computes the
    // infeasibility measure of a certain fitness
    PAGMO_DLL_LOCAL double compute_infeasibility(const vector_double &) const;

    // According to the population, the first penalty may or may not be applied
    bool m_apply_penalty_1;
    // The parameter gamma that scales the second penalty
    double m_scaling_factor;
    // The normalization of each constraint
    vector_double m_c_max;
    // The fitness of the three reference individuals
    vector_double m_f_hat_down;
    vector_double m_f_hat_up;
    vector_double m_f_hat_round;
    // The infeasibilities of the three reference individuals
    double m_i_hat_down;
    double m_i_hat_up;
    double m_i_hat_round;

    vector_double::size_type m_n_feasible;
    // A NAKED pointer to the reference population, allowing to call the fitness function and later recover
    // the counters outside of the class, and avoiding unecessary copies. Use with care.
    population *m_pop_ptr;
    // The hash map connecting the decision vector to their fitnesses. The use of
    // custom comparison is needed to take care of nans, while the custom hasher is needed as std::hash does not
    // work on std::vectors
    mutable std::unordered_map<vector_double, vector_double, detail::hash_vf<double>, detail::equal_to_vf<double>>
        m_fitness_map;
};

// Only for debug purposes
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const penalized_udp &);

} // namespace detail

/// Self-adaptive constraints handling
/**
 *
 * This meta-algorithm implements a constraint handling technique that allows the use of any user-defined algorithm
 * (UDA) able to deal with single-objective unconstrained problems, on single-objective constrained problems. The
 * technique self-adapts its parameters during
 * each successive call to the inner UDA basing its decisions on the entire underlying population. The resulting
 * approach is an alternative to using the meta-problem pagmo::unconstrain to transform the
 * constrained fitness into an unconstrained fitness.
 *
 * The self-adaptive constraints handling meta-algorithm is largely based on the ideas of Faramani and Wright but it
 * extends their use to any-algorithm, in particular to non generational population based evolutionary approaches where
 * a steady-state reinsertion is used (i.e., as soon as an individual is found fit it is immediately reinserted into the
 * pop and will influence the next offspring genetic material).
 *
 * Each decision vector is assigned an infeasibility measure \f$\iota\f$ which accounts for the normalized violation of
 * all the constraints (discounted by the constraints tolerance as returned by pagmo::problem::get_c_tol()). The
 * normalization factor used \f$c_{j_{max}}\f$ is the maximum violation of the \f$j-th\f$ constraint.
 *
 * As in the original paper, three individuals in the evolving population are then used to penalize the single
 * objective.
 *
 * \f[
 * \begin{array}{rl}
 *   \check X & \mbox{: the best decision vector} \\
 *   \hat X & \mbox{: the worst decision vector} \\
 *   \breve X & \mbox{: the decision vector with the highest objective}
 * \end{array}
 * \f]
 *
 * The best and worst decision vectors are defined accounting for their infeasibilities and for the value of the
 * objective function. Using the above definitions the overall pseudo code can be summarized as follows:
 *
 * @code{.unparsed}
 * > Select a pagmo::population (related to a single-objective constrained problem)
 * > Select a UDA (able to solve single-objective unconstrained problems)
 * > while i < iter
 * > > Compute the normalization factors (will depend on the current population)
 * > > Compute the best, worst, highest (will depend on the current population)
 * > > Evolve the population using the UDA and a penalized objective
 * > > Reinsert the best decision vector from the previous evolution
 * @endcode
 *
 * pagmo::cstrs_self_adaptive is a user-defined algorithm (UDA) that can be used to construct pagmo::algorithm objects.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    Self-adaptive constraints handling implements an internal cache to avoid the re-evaluation of the fitness
 *    for decision vectors already evaluated. This makes the final counter of function evaluations somewhat
 *    unpredictable. The number of function evaluation will be bounded to ``iters`` times the fevals made by one call to
 *    the inner UDA. The internal cache is reset at each iteration, but its size will grow unlimited during each call to
 *    the inner UDA evolve method.
 *
 * .. note::
 *
 *    Several modification were made to the original Faramani and Wright ideas to allow their approach to work on
 *    corner cases and with any UDAs. Most notably, a violation to the :math:`j`-th  constraint is ignored if all
 *    the decision vectors in the population satisfy that particular constraint (i.e. if :math:`c_{j_{max}} = 0`).
 *
 * .. note::
 *
 *    The performances of :cpp:class:`pagmo::cstrs_self_adaptive` are highly dependent on the particular inner UDA
 *    employed and in particular to its parameters (generations / iterations).
 *
 * .. seealso::
 *
 *    Farmani, Raziyeh, and Jonathan A. Wright. "Self-adaptive fitness formulation for constrained optimization." IEEE
 *    Transactions on Evolutionary Computation 7.5 (2003): 445-455.
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC cstrs_self_adaptive
{
public:
    /// Single entry of the log (iter, fevals, best_f, infeas, n. constraints violated, violation norm).
    typedef std::tuple<unsigned, unsigned long long, double, double, vector_double::size_type, double,
                       vector_double::size_type>
        log_line_type;
    /// The log.
    typedef std::vector<log_line_type> log_type;
    /// Default constructor.
    /**
     * @param iters Number of iterations (calls to the inner UDA). After each iteration the penalty is adapted
     * The default constructor will initialize the algorithm with the following parameters:
     * - inner algorithm: pagmo::de{1u};
     * - seed: random.
     */
    cstrs_self_adaptive(unsigned iters = 1u);

private:
    // Enabler for the ctor from UDA.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<algorithm, T &&>::value, int>;

public:
    /// Constructor.
    /**
     *
     * @param iters Number of iterations (calls to the inner algorithm). After each iteration the penalty is adapted
     * @param a a pagmo::algorithm (or UDA) that will be used to construct the inner algorithm.
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws unspecified any exception thrown by the constructor of pagmo::algorithm.
     */
    template <typename T, ctor_enabler<T> = 0>
    explicit cstrs_self_adaptive(unsigned iters, T &&a, unsigned seed = pagmo::random_device::next())
        : m_algorithm(std::forward<T>(a)), m_iters(iters), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
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
     * - >0: will print and log one line each  \p level call to the inner algorithm.
     *
     * Example (verbosity 10):
     * @code{.unparsed}
     *   Iter:        Fevals:          Best: Infeasibility:      Violated:    Viol. Norm:   N. Feasible:
     *       1              0       -69.2141       0.235562              6        117.743              0 i
     *      11            200       -69.2141       0.248216              6        117.743              0 i
     *      21            400       -29.4754      0.0711599              5          44.39              0 i
     *      31            600       -30.0791      0.0878253              4        44.3803              0 i
     *     ...            ...       ........      .........              .        .......              . .
     *     211           4190       -7.68336    0.000341894              1       0.273829              0 i
     *     221           4390       -7.89941     0.00031154              1       0.273829              0 i
     *     231           4590       -8.38299    0.000168309              1       0.147935              0 i
     *     241           4790       -8.38299    0.000181461              1       0.147935              0 i
     *     251           4989       -8.71021    0.000191197              1       0.100357              0 i
     *     261           5189       -8.71021    0.000165734              1       0.100357              0 i
     *     271           5389       -10.7421              0              0              0              3
     *     281           5585       -10.7421              0              0              0              3
     *     291           5784       -11.4868              0              0              0              4
     *
     * @endcode
     * \p Iter is the iteration number, \p Fevals is the number of fitness evaluations, \p Best is the objective
     * function of the best fitness currently in the population, \p Infeasibility is the normailized infeasibility
     * measure, \p Violated is the number of constraints currently violated by the best solution, <tt>Viol. Norm</tt> is
     * the norm of the violation (discounted already by the constraints tolerance) and N. Feasible is the number of
     * feasible individuals in the current iteration. The small \p i appearing at the end of the line stands for
     * "infeasible" and will disappear only once \p Violated is 0.
     *
     * @param level verbosity level.
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    }

    /// Get the verbosity level.
    /**
     * @return the verbosity level.
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }

    /// Get log.
    /**
     * A log containing relevant quantities monitoring the last call to cstrs_self_adaptive::evolve(). Each element of
     * the returned <tt>std::vector</tt> is a cstrs_self_adaptive::log_line_type containing: \p Iter, \p Fevals, \p
     * Best, \p Infeasibility, \p Violated, <tt>Viol. Norm</tt>, <tt>N. Feasible</tt> as described in
     * cstrs_self_adaptive::set_verbosity().
     *
     * @return an <tt>std::vector</tt> of cstrs_self_adaptive::log_line_type containing the logged values Iters,
     * Fevals, Best, Infeasibility, Violated and Viol. Norm and N. Feasible.
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    /// Algorithm name
    /**
     * @return a string containing the algorithm name.
     */
    std::string get_name() const
    {
        return "sa-CNSTR: Self-adaptive constraints handling";
    }

    // Extra info
    std::string get_extra_info() const;

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

    /// Getter for the inner problem.
    /**
     * Returns a reference to the inner pagmo::algorithm.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
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

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Inner algorithm
    algorithm m_algorithm;
    unsigned m_iters;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::cstrs_self_adaptive)

#endif
