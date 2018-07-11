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

#include <algorithm> //std::if_all
#include <iomanip>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/generic.hpp> // pagmo::uniform_real_from_range

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
class mbh
{
    // Enabler for the ctor from UDA or algorithm. In this case we allow construction from type algorithm.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<algorithm, T &&>::value, int>;

public:
    /// Single entry of the log (feval, best fitness, n. constraints violated, violation norm, trial).
    typedef std::tuple<unsigned long long, double, vector_double::size_type, double, unsigned> log_line_type;
    /// The log.
    typedef std::vector<log_line_type> log_type;
    /// Default constructor.
    /**
     * The default constructor will initialize the algorithm with the following parameters:
     * - inner algorithm: pagmo::compass_search;
     * - consecutive runs of the inner algorithm that need to result in no improvement for pagmo::mbh to stop: 5;
     * - scalar perturbation: 1E-2;
     * - seed: random.
     *
     * @throws unspecified any exception thrown by the constructor of pagmo::algorithm.
     */
    mbh() : m_algorithm(compass_search{}), m_stop(5u), m_perturb(1, 1e-2), m_verbosity(0u)
    {
        const auto rnd = pagmo::random_device::next();
        m_seed = rnd;
        m_e.seed(rnd);
    }
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
        if (perturb > 1. || perturb <= 0. || std::isnan(perturb)) {
            pagmo_throw(std::invalid_argument, "The scalar perturbation must be in (0, 1], while a value of "
                                                   + std::to_string(perturb) + " was detected.");
        }
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
        if (!std::all_of(perturb.begin(), perturb.end(),
                         [](double item) { return (item > 0. && item <= 1. && !std::isnan(item)); })) {
            pagmo_throw(std::invalid_argument,
                        "The perturbation must have all components in (0, 1], while that is not the case.");
        }
    }
    /// Evolve method.
    /**
     * This method will evolve the input population up to when \p stop consecutve runs of the internal
     * algorithm do not improve the solution.
     *
     * @param pop population to be evolved.
     *
     * @return evolved population.
     *
     * @throws std::invalid_argument if the problem is multi-objective or stochastic, or if the perturbation vector size
     * does not equal the problem size.
     */
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                              // allowed
        auto dim = prob.get_nx();             // This getter does not return a const reference but a copy
        auto nec = prob.get_nec();            // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto NP = pop.size();

        auto fevals0 = prob.get_fevals(); // discount for the already made fevals
        unsigned count = 1u;              // regulates the screen output

        // PREAMBLE-------------------------------------------------------------------------------------------------
        if (prob.get_nobj() != 1u) {
            pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                                   + get_name() + " cannot deal with them");
        }
        if (prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument, "The input problem " + prob.get_name() + " appears to be stochastic, "
                                                   + get_name() + " cannot deal with it");
        }
        // Get out if there is nothing to do.
        if (m_stop == 0u) {
            return pop;
        }
        // Check if the perturbation vector has size 1, in which case the whole perturbation vector is filled with
        // the same value equal to its first entry
        if (m_perturb.size() == 1u) {
            for (decltype(dim) i = 1u; i < dim; ++i) {
                m_perturb.push_back(m_perturb[0]);
            }
        }
        // Check that the perturbation vector size equals the size of the problem
        if (m_perturb.size() != dim) {
            pagmo_throw(std::invalid_argument, "The perturbation vector size is: " + std::to_string(m_perturb.size())
                                                   + ", while the problem dimension is: " + std::to_string(dim)
                                                   + ". They need to be equal for MBH to work.");
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();
        // mbh main loop
        unsigned i = 0u;
        while (i < m_stop) {
            // 1 - We make a copy of the current population
            population pop_old(pop);
            // 2 - We perturb the current population (NP funevals are made here)
            for (decltype(NP) j = 0u; j < NP; ++j) {
                vector_double tmp_x(dim);
                for (decltype(dim) k = 0u; k < dim; ++k) {
                    tmp_x[k] = uniform_real_from_range(
                        std::max(pop.get_x()[j][k] - m_perturb[k] * (ub[k] - lb[k]), lb[k]),
                        std::min(pop.get_x()[j][k] + m_perturb[k] * (ub[k] - lb[k]), ub[k]), m_e);
                }
                pop.set_x(j, tmp_x); // fitness is evaluated here
            }
            // 3 - We evolve the current population with the selected algorithm
            pop = m_algorithm.evolve(pop);
            i++;
            // 4 - We reset the counter if we have improved, otherwise we reset the population
            if (compare_fc(pop.get_f()[pop.best_idx()], pop_old.get_f()[pop_old.best_idx()], nec, prob.get_c_tol())) {
                i = 0u;
            } else {
                for (decltype(NP) j = 0u; j < NP; ++j) {
                    pop.set_xf(j, pop_old.get_x()[j], pop_old.get_f()[j]);
                }
            }
            // 5 - We log to screen
            if (m_verbosity > 0u) {
                // Prints a log line after each call to the inner algorithm
                // 1 - Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "Violated:", std::setw(15), "Viol. Norm:", std::setw(15), "Trial:", '\n');
                }
                // 2 - Print
                auto cur_best_f = pop.get_f()[pop.best_idx()];
                auto c1eq = detail::test_eq_constraints(cur_best_f.data() + 1, cur_best_f.data() + 1 + nec,
                                                        prob.get_c_tol().data());
                auto c1ineq = detail::test_ineq_constraints(
                    cur_best_f.data() + 1 + nec, cur_best_f.data() + cur_best_f.size(), prob.get_c_tol().data() + nec);
                auto n = prob.get_nc() - c1eq.first - c1ineq.first;
                auto l = c1eq.second + c1ineq.second;
                print(std::setw(7), prob.get_fevals() - fevals0, std::setw(15), cur_best_f[0], std::setw(15), n,
                      std::setw(15), l, std::setw(15), i);
                if (!prob.feasibility_f(pop.get_f()[pop.best_idx()])) {
                    std::cout << " i";
                }
                ++count;
                std::cout << std::endl; // we flush here as we want the user to read in real time ...
                // Logs
                m_log.emplace_back(prob.get_fevals() - fevals0, cur_best_f[0], n, l, i);
            }
        }
        // We extract chromosomes and fitnesses
        return pop;
    }
    /// Set the seed.
    /**
     * @param seed the seed controlling the algorithm's stochastic behaviour.
     */
    void set_seed(unsigned seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    }
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
    /// Set the perturbation vector.
    /**
     * @param perturb the perturbation vector.
     *
     * @throws std::invalid_argument if not all the components of the perturbation vector are in the (0,1] range.
     */
    void set_perturb(const vector_double &perturb)
    {
        if (!std::all_of(perturb.begin(), perturb.end(),
                         [](double item) { return (item > 0. && item <= 1. && !std::isnan(item)); })) {
            pagmo_throw(std::invalid_argument,
                        "The perturbation must have all components in (0, 1], while that is not the case.");
        }
        m_perturb = perturb;
    }
    /// Algorithm's thread safety level.
    /**
     * The thread safety of a meta-algorithm is defined by the thread safety of the interal pagmo::algorithm.
     *
     * @return the thread safety level of the interal pagmo::algorithm.
     */
    thread_safety get_thread_safety() const
    {
        return m_algorithm.get_thread_safety();
    }
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
    /// Extra informations.
    /**
     * @return a string containing extra informations on the algorithm.
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        stream(ss, "\tStop: ", m_stop);
        stream(ss, "\n\tPerturbation vector: ", m_perturb);
        stream(ss, "\n\tSeed: ", m_seed);
        stream(ss, "\n\tVerbosity: ", m_verbosity);
        stream(ss, "\n\n\tInner algorithm: ", m_algorithm.get_name());
        stream(ss, "\n\tInner algorithm extra info: ");
        stream(ss, "\n", m_algorithm.get_extra_info());
        return ss.str();
    }
    /// Object serialization.
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the inner algorithm and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_algorithm, m_stop, m_perturb, m_e, m_seed, m_verbosity, m_log);
    }

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

PAGMO_REGISTER_ALGORITHM(pagmo::mbh)

#endif
