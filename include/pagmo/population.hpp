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

#ifndef PAGMO_POPULATION_HPP
#define PAGMO_POPULATION_HPP

#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/bfe.hpp>
#include <pagmo/detail/island_fwd.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{
/// Population class.
/**
 * \image html pop_no_text.png
 *
 * This class represents a population of individuals, i.e., potential
 * candidate solutions to a given problem. In pagmo an
 * individual is determined by:
 * - a unique ID used to track it across generations and migrations,
 * - a chromosome (a decision vector),
 * - the fitness of the chromosome as evaluated by a pagmo::problem,
 * and thus including objectives, equality constraints and inequality
 * constraints if present.
 *
 * A special mechanism is implemented to track the best individual that has ever
 * been part of the population. Such an individual is called *champion* and its
 * decision vector and fitness vector are automatically kept updated. The *champion* is
 * not necessarily an individual currently in the population. The *champion* is
 * only defined and accessible via the population interface if the pagmo::problem
 * currently contained in the pagmo::population is single objective.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from :cpp:class:`pagmo::population` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC population
{
    // Make friends with island for direct
    // access to the population's members during
    // evolution.
    friend class PAGMO_DLL_PUBLIC island;

public:
    /// The size type of the population.
    typedef pop_size_t size_type;
    // Default constructor
    population();

private:
    void prob_ctor_impl(size_type);
    // Enable the generic ctor only if T is not a population (after removing
    // const/reference qualifiers).
    template <typename T>
    using generic_ctor_enabler
        = enable_if_t<detail::conjunction<detail::negation<std::is_same<population, uncvref_t<T>>>,
                                          std::is_constructible<problem, T &&>>::value,
                      int>;

public:
    /// Constructor from a problem.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if, after the removal of cv/reference qualifiers,
     *    ``T`` is not :cpp:class:`pagmo::population`, and if :cpp:class:`pagmo::problem` is constructible from ``T``.
     *
     * \endverbatim
     *
     * Constructs a population with \p pop_size individuals associated
     * to the problem \p x and setting the population random seed
     * to \p seed. The input problem \p x can be either a pagmo::problem or a user-defined problem
     * (UDP).
     *
     * @param x the problem the population refers to.
     * @param pop_size population size (i.e. number of individuals therein).
     * @param seed seed of the random number generator used, for example, to
     * create new random individuals within the bounds.
     *
     * @throws unspecified any exception thrown by random_decision_vector(), push_back(), or by the
     * invoked constructor of pagmo::problem.
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit population(T &&x, size_type pop_size = 0u, unsigned seed = pagmo::random_device::next())
        : m_prob(std::forward<T>(x)), m_e(seed), m_seed(seed)
    {
        prob_ctor_impl(pop_size);
    }

private:
    // Implementation of the ctor from bfe. Distinguish the two cases
    // in which bfe or a udbfe were provided.
    void constructor_from_bfe_impl(const bfe &, size_type, const std::true_type &);
    template <typename U>
    void constructor_from_bfe_impl(U &&b, size_type pop_size, const std::false_type &)
    {
        constructor_from_bfe_impl(bfe(std::forward<U>(b)), pop_size, std::true_type{});
    }

public:
    /// Constructor from a problem and a batch fitness evaluator.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if :cpp:class:`pagmo::problem` is constructible from ``T``,
     *    and :cpp:class:`pagmo::bfe` is constructible from ``U``.
     *
     * Constructs a population with *pop_size* individuals associated
     * to the problem *x* and setting the population random seed
     * to *seed*. The input problem *x* can be either a :cpp:class:`pagmo::problem` or a user-defined problem
     * (UDP). The fitnesses of the individuals will be evaluated with the input
     * :cpp:class:`pagmo::bfe` or UDBFE *b*.
     *
     * \endverbatim
     *
     * @param x the problem the population refers to.
     * @param b the (user-defined) batch fitness evaluator that will be used to evaluate the fitnesses of the
     * individuals.
     * @param pop_size population size (i.e. number of individuals therein).
     * @param seed seed of the random number generator used, for example, to
     * create new random individuals within the bounds.
     *
     * @throws unspecified any exception thrown by batch_random_decision_vector(), the public API of the (user-defined)
     * batch fitness evaluator, push_back(), or by the invoked constructor of pagmo::problem.
     */
    template <
        typename T, typename U,
        enable_if_t<detail::conjunction<std::is_constructible<problem, T &&>, std::is_constructible<bfe, U &&>>::value,
                    int> = 0>
    explicit population(T &&x, U &&b, size_type pop_size = 0u, unsigned seed = pagmo::random_device::next())
        : m_prob(std::forward<T>(x)), m_e(seed), m_seed(seed)
    {
        constructor_from_bfe_impl(std::forward<U>(b), pop_size, std::is_same<uncvref_t<U>, bfe>{});
    }

    // Copy constructor.
    population(const population &);
    // Move constructor.
    population(population &&) noexcept;
    // Copy assignment operator.
    population &operator=(const population &);
    // Move assignment operator.
    population &operator=(population &&) noexcept;
    // Destructor.
    ~population();

private:
    // Internal implementation of push_back().
    template <typename T, typename U>
    void push_back_impl(T &&, U &&);
    // Short routine to update the champion. Does nothing if the problem is MO
    PAGMO_DLL_LOCAL void update_champion(vector_double, vector_double);

public:
    // Adds one decision vector (chromosome) to the population.
    void push_back(const vector_double &);
    // Adds one decision vector (chromosome) to the population (move overload).
    void push_back(vector_double &&);
    // Adds one decision vector/fitness vector to the population.
    void push_back(const vector_double &, const vector_double &);
    // Adds one decision vector/fitness vector to the population (move overload).
    void push_back(vector_double &&, vector_double &&);

    // Creates a random decision vector
    vector_double random_decision_vector() const;

    // Index of the best individual
    size_type best_idx() const;
    // Index of the best individual (accounting for a vector tolerance)
    size_type best_idx(const vector_double &) const;
    // Index of the best individual (accounting for a scalar tolerance)
    size_type best_idx(double) const;

    // Index of the worst individual
    size_type worst_idx() const;
    // Index of the worst individual (accounting for a vector tolerance)
    size_type worst_idx(const vector_double &) const;
    // Index of the worst individual (accounting for a scalar tolerance)
    size_type worst_idx(double) const;

    // Champion decision vector
    vector_double champion_x() const;
    // Champion fitness
    vector_double champion_f() const;

    /// Number of individuals in the population
    /**
     * @return the number of individuals in the population
     */
    size_type size() const
    {
        assert(m_f.size() == m_ID.size());
        assert(m_x.size() == m_ID.size());
        return m_ID.size();
    }

    // Sets the \f$i\f$-th individual decision vector, and fitness
    void set_xf(size_type, const vector_double &, const vector_double &);
    // Sets the \f$i\f$-th individual's chromosome
    void set_x(size_type, const vector_double &);

    /// Const getter for the pagmo::problem.
    /**
     * @return a const reference to the internal pagmo::problem.
     */
    const problem &get_problem() const
    {
        return m_prob;
    }

    /// Getter for the pagmo::problem.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    The ability to extract a mutable reference to the problem is provided solely in order to
     *    allow calling non-const methods on the problem. Assigning the population's problem via a reference
     *    returned by this method is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a reference to the internal pagmo::problem.
     */
    problem &get_problem()
    {
        return m_prob;
    }

    /// Const getter for the fitness vectors.
    /**
     * @return a const reference to the vector of fitness vectors.
     */
    const std::vector<vector_double> &get_f() const
    {
        return m_f;
    }

    /// Const getter for the decision vectors.
    /**
     * @return a const reference to the vector of decision vectors.
     */
    const std::vector<vector_double> &get_x() const
    {
        return m_x;
    }

    /// Const getter for the individual IDs.
    /**
     * @return a const reference to the vector of individual IDs.
     */
    const std::vector<unsigned long long> &get_ID() const
    {
        return m_ID;
    }

    /// Getter for the seed of the population random engine.
    /**
     * @return the seed of the population's random engine.
     */
    unsigned get_seed() const
    {
        return m_seed;
    }

    /// Save to archive.
    /**
     * This method will save \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the internal pagmo::problem and of primitive
     * types.
     */
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_prob, m_ID, m_x, m_f, m_champion_x, m_champion_f, m_e, m_seed);
    }
    /// Load from archive.
    /**
     * This method will load a pagmo::population from \p ar into \p this.
     *
     * @param ar source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of the internal pagmo::problem and of
     * primitive
     * types.
     */
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        population tmp;
        try {
            detail::from_archive(ar, tmp.m_prob, tmp.m_ID, tmp.m_x, tmp.m_f, tmp.m_champion_x, tmp.m_champion_f,
                                 tmp.m_e, tmp.m_seed);
            // LCOV_EXCL_START
        } catch (...) {
            // NOTE: if anything goes wrong during deserialization, erase
            // all individuals before re-throwing (otherwise, we would hit
            // assertion errors in debug mode in tmp's dtor if the vectors containing
            // the individuals have inconsistent lengths).
            tmp.clear();
            throw;
        }
        // LCOV_EXCL_STOP
        *this = std::move(tmp);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
private:
    void clear();

private:
    // Problem.
    problem m_prob;
    // ID of the various decision vectors
    std::vector<unsigned long long> m_ID;
    // Decision vectors.
    std::vector<vector_double> m_x;
    // Fitness vectors.
    std::vector<vector_double> m_f;
    // The Champion chromosome
    vector_double m_champion_x;
    // The Champion fitness
    vector_double m_champion_f;
    // Random engine.
    mutable detail::random_engine_type m_e;
    // Seed.
    unsigned m_seed;
};

// Streaming operator for the class pagmo::population.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const population &);

} // namespace pagmo

// Disable tracking for the serialisation of population.
BOOST_CLASS_TRACKING(pagmo::population, boost::serialization::track_never)

#endif
