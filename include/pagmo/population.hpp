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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/generic.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

namespace detail
{

// Fwd-declaration of the struct holding the implementation of the
// population's parallel initialisation functionality.
template <typename>
struct pop_parinit;

} // namespace detail

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
class population
{
    // Enable the generic ctor only if T is not a population (after removing
    // const/reference qualifiers).
    template <typename T>
    using generic_ctor_enabler
        = enable_if_t<!std::is_same<population, uncvref_t<T>>::value && std::is_constructible<problem, T &&>::value,
                      int>;

public:
    /// The size type of the population.
    typedef std::vector<vector_double>::size_type size_type;
    /// Default constructor
    /**
     * Constructs an empty population with a pagmo::null_problem.
     * The random seed is initialised to zero.
     *
     * @throws unspecified any exception thrown by the constructor from problem.
     */
    population() : population(null_problem{}, 0u, 0u) {}

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
    explicit population(T &&x, size_type pop_size = 0u, unsigned seed = pagmo::random_device::next(),
                        const std::string &init_mode = "serial")
        : m_prob(std::forward<T>(x)), m_e(seed), m_seed(seed)
    {
        if (init_mode == "serial") {
            for (size_type i = 0u; i < pop_size; ++i) {
                push_back(random_decision_vector());
            }
        } else {
            // NOTE: put T as type in here, which allows us to delay the instantiation
            // of pop_parinit until after its implementation has been defined.
            auto ret = detail::pop_parinit<T>::s_func(init_mode, m_prob, seed, pop_size);
            // Double check the return value.
            assert(ret.first.size() == pop_size);
            assert(ret.second.size() == pop_size);
            // Push back the generated dvs and fitnesses.
            for (size_type i = 0u; i < pop_size; ++i) {
                push_back(ret.first[i], ret.second[i]);
            }
        }
    }

    /// Defaulted copy constructor.
    population(const population &) = default;

    /// Defaulted move constructor.
    /**
     * @param pop construction argument.
     */
    population(population &&pop) noexcept
        : m_prob(std::move(pop.m_prob)), m_ID(std::move(pop.m_ID)), m_x(std::move(pop.m_x)), m_f(std::move(pop.m_f)),
          m_champion_x(std::move(pop.m_champion_x)), m_champion_f(std::move(pop.m_champion_f)), m_e(std::move(pop.m_e)),
          m_seed(std::move(pop.m_seed))
    {
    }

    /// Copy assignment operator.
    /**
     * Copy assignment is implemented via copy+move.
     *
     * @param other assignment argument.
     *
     * @return a reference to \p this.
     *
     * @throws unspecified any exception thrown by the copy constructor.
     */
    population &operator=(const population &other)
    {
        if (this != &other) {
            *this = population(other);
        }
        return *this;
    }

    /// Defaulted move assignment operator.
    /**
     * @param pop assignment argument.
     *
     * @return a reference to \p this.
     */
    population &operator=(population &&pop) noexcept
    {
        if (this != &pop) {
            m_prob = std::move(pop.m_prob);
            m_ID = std::move(pop.m_ID);
            m_x = std::move(pop.m_x);
            m_f = std::move(pop.m_f);
            m_champion_x = std::move(pop.m_champion_x);
            m_champion_f = std::move(pop.m_champion_f);
            m_e = std::move(pop.m_e);
            m_seed = std::move(pop.m_seed);
        }
        return *this;
    }

    /// Destructor.
    /**
     * The destructor will run sanity checks in debug mode.
     */
    ~population()
    {
        assert(m_ID.size() == m_x.size());
        assert(m_ID.size() == m_f.size());
    }

    /// Adds one decision vector (chromosome) to the population.
    /**
     * Appends a new chromosome \p x to the population, evaluating
     * its fitness and creating a new unique identifier for the newly
     * born individual.
     *
     * In case of exceptions, the population will not be altered.
     *
     * @param x decision vector to be added to the population.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers or by problem::fitness().
     */
    void push_back(const vector_double &x)
    {
        // This line will throw if dv dimensions are wrong, or fitness dimensions are worng
        auto f = m_prob.fitness(x);
        push_back(x, f);
    }

    /// Adds one decision vector/fitness vector to the population
    /**
     * Appends a new chromosome \p x to the population, and sets
     * its fitness to \p f creating a new unique identifier for the newly
     * born individual.
     *
     * In case of exceptions, the population will not be altered.
     *
     * @param x decision vector to be added to the population.
     * @param f fitness vector corresponding to the decision vector
     *
     * @throws std::invalid_argument if the size of \p x differs from the problem's dimension,
     * or if the size of \p f differs from the fitness dimension.
     * @throws unspecified any exception thrown by memory errors in standard containers.
     */
    void push_back(const vector_double &x, const vector_double &f)
    {
        // Checks on the input vectors.
        if (x.size() != m_prob.get_nx()) {
            pagmo_throw(std::invalid_argument,
                        "Trying to add a decision vector of dimension: " + std::to_string(x.size())
                            + ", while the problem's dimension is: " + std::to_string(m_prob.get_nx()));
        }
        if (f.size() != m_prob.get_nf()) {
            pagmo_throw(std::invalid_argument,
                        "Trying to add a fitness of dimension: " + std::to_string(f.size())
                            + ", while the problem's fitness has dimension: " + std::to_string(m_prob.get_nf()));
        }

        // Prepare quantities to be appended to the internal vectors.
        const auto new_id = std::uniform_int_distribution<unsigned long long>()(m_e);
        auto x_copy(x);
        auto f_copy(f);
        // Reserve space in the vectors.
        // NOTE: in face of overflow here, reserve(0) will be called, which is fine.
        // The first push_back below will then fail, with no modifications to the class taking place.
        m_ID.reserve(m_ID.size() + 1u);
        m_x.reserve(m_x.size() + 1u);
        m_f.reserve(m_f.size() + 1u);

        // update champion either throws before modfying anything, or completes successfully. The rest is noexcept.
        update_champion(x, f);
        m_ID.push_back(new_id);
        m_x.push_back(std::move(x_copy));
        m_f.push_back(std::move(f_copy));
    }

    /// Creates a random decision vector
    /**
     * Creates a random decision vector within the problem's bounds.
     * It calls internally pagmo::random_decision_vector().
     *
     * @returns a random decision vector
     *
     * @throws unspecified all exceptions thrown by pagmo::random_decision_vector()
     */
    vector_double random_decision_vector() const
    {
        return pagmo::random_decision_vector(m_prob.get_bounds(), m_e, get_problem().get_nix());
    }

    /// Index of the best individual
    /**
     * If the problem is single-objective and unconstrained, the best
     * is simply the individual with the smallest fitness. If the problem
     * is, instead, single objective, but with constraints, the best
     * will be defined using the criteria specified in pagmo::sort_population_con().
     * If the problem is multi-objective one single best is not defined. In
     * this case the user can still obtain a strict ordering of the population
     * individuals by calling the pagmo::sort_population_mo() function.
     *
     * The pagmo::population::get_c_tol() tolerances are accounted by default.
     * If different tolerances are required the other overloads can be used.
     *
     * @returns the index of the best individual.
     *
     * @throws std::invalid_argument if the problem is multiobjective and thus
     * a best individual is not well defined, or if the population is empty.
     * @throws unspecified any exception thrown by pagmo::sort_population_con().
     */
    size_type best_idx() const
    {
        return best_idx(get_problem().get_c_tol());
    }

    /// Index of the best individual (accounting for a vector tolerance)
    /**
     * @param tol vector of tolerances to be applied to each constraints.
     *
     * @returns the index of the best individual.
     *
     * @throws std::invalid_argument if the problem is multiobjective and thus
     * a best individual is not well defined, or if the population is empty.
     * @throws unspecified any exception thrown by pagmo::sort_population_con().
     */
    size_type best_idx(const vector_double &tol) const
    {
        if (!size()) {
            pagmo_throw(std::invalid_argument, "Cannot determine the best individual of an empty population");
        }
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        "The best individual can only be extracted in single objective problems");
        }
        if (m_prob.get_nc() > 0u) { // TODO: should we also code a min_element_population_con?
            return sort_population_con(m_f, m_prob.get_nec(), tol)[0];
        }
        // Sort for single objective, unconstrained optimization
        std::vector<size_type> indexes(size());
        std::iota(indexes.begin(), indexes.end(), size_type(0u));
        return *std::min_element(indexes.begin(), indexes.end(),
                                 [this](size_type idx1, size_type idx2) { return m_f[idx1] < m_f[idx2]; });
    }

    /// Index of the best individual (accounting for a scalar tolerance)
    /**
     * @param tol scalar tolerance to be considered for each constraint.
     *
     * @return index of the best individual.
     */
    size_type best_idx(double tol) const
    {
        vector_double tol_vector(m_prob.get_nf() - 1u, tol);
        return best_idx(tol_vector);
    }

    /// Index of the worst individual
    /**
     * If the problem is single-objective and unconstrained, the worst
     * is simply the individual with the largest fitness. If the problem
     * is, instead, single objective, but with constraints, the worst individual
     * will be defined using the criteria specified in pagmo::sort_population_con().
     * If the problem is multi-objective one single worst is not defined. In
     * this case the user can still obtain a strict ordering of the population
     * individuals by calling the pagmo::sort_population_mo() function.
     *
     * The pagmo::population::get_c_tol() tolerances are accounted by default.
     * If different tolerances are required the other overloads can be used.
     *
     * @returns the index of the worst individual.
     *
     * @throws std::invalid_argument if the problem is multiobjective and thus
     * a worst individual is not well defined, or if the population is empty.
     * @throws unspecified any exception thrown by pagmo::sort_population_con().
     */
    size_type worst_idx() const
    {
        return worst_idx(get_problem().get_c_tol());
    }

    /// Index of the worst individual (accounting for a vector tolerance)
    /**
     * If the problem is single-objective and unconstrained, the worst
     * is simply the individual with the largest fitness. If the problem
     * is, instead, single objective, but with constraints, the worst individual
     * will be defined using the criteria specified in pagmo::sort_population_con().
     * If the problem is multi-objective one single worst is not defined. In
     * this case the user can still obtain a strict ordering of the population
     * individuals by calling the pagmo::sort_population_mo() function.
     *
     * @param tol vector of tolerances to be applied to each constraints.
     *
     * @returns the index of the worst individual.
     *
     * @throws std::invalid_argument if the problem is multiobjective and thus
     * a worst individual is not well defined, or if the population is empty.
     * @throws unspecified any exception thrown by pagmo::sort_population_con().
     */
    size_type worst_idx(const vector_double &tol) const
    {
        if (!size()) {
            pagmo_throw(std::invalid_argument, "Cannot determine the worst element of an empty population");
        }
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        "The worst element of a population can only be extracted in single objective problems");
        }
        if (m_prob.get_nc() > 0u) { // TODO: should we also code a min_element_population_con?
            return sort_population_con(m_f, m_prob.get_nec(), tol).back();
        }
        // Sort for single objective, unconstrained optimization
        std::vector<size_type> indexes(size());
        std::iota(indexes.begin(), indexes.end(), size_type(0u));
        return *std::max_element(indexes.begin(), indexes.end(),
                                 [this](size_type idx1, size_type idx2) { return m_f[idx1] < m_f[idx2]; });
    }

    /// Index of the worst individual (accounting for a scalar tolerance)
    /**
     * @param tol scalar tolerance to be considered for each constraint.
     *
     * @return index of the worst individual.
     */
    size_type worst_idx(double tol) const
    {
        vector_double tol_vector(m_prob.get_nf() - 1u, tol);
        return worst_idx(tol_vector);
    }

    /// Champion decision vector
    /**
     * @return the champion decision vector.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    If the problem is stochastic the champion is the individual that had the lowest fitness for
     *    some lucky seed, not on average across seeds. Re-evaluating its decision vector may then result in a different
     *    fitness.
     *
     * \endverbatim
     *
     * @throw std::invalid_argument if the current problem is not single objective.
     */
    vector_double champion_x() const
    {
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        "The Champion of a population can only be extracted in single objective problems");
        }
        if (m_prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The Champion of a population can only be extracted for non stochastic problems");
        }
        return m_champion_x;
    }

    /// Champion fitness
    /**
     * @return the champion fitness.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    If the problem is stochastic the champion is the individual that had the lowest fitness for
     *    some lucky seed, not on average across seeds. Re-evaluating its decision vector may then result in a different
     *    fitness.
     *
     * \endverbatim
     *
     * @throw std::invalid_argument if the current problem is not single objective.
     */
    vector_double champion_f() const
    {
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        "The Champion of a population can only be extracted in single objective problems");
        }
        if (m_prob.is_stochastic()) {
            pagmo_throw(std::invalid_argument,
                        "The Champion of a population can only be extracted for non stochastic problems");
        }
        return m_champion_f;
    }

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

    /// Sets the \f$i\f$-th individual decision vector, and fitness
    /**
     * Sets simultaneously the \f$i\f$-th individual decision vector
     * and fitness thus avoiding to trigger a fitness function evaluation.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    Pagmo will only control the input fitness ``f`` dimension, so the user can associate decision vector, fitness
     *    vectors pairs that are not consistent with the fitness function.
     *
     * \endverbatim
     *
     * @param i individual's index in the population.
     * @param x a decision vector (chromosome).
     * @param f a fitness vector.
     *
     * @throws std::invalid_argument if either:
     * - \p i is invalid (i.e. larger or equal to the population size),
     * - \p x has not the correct dimension,
     * - \p f has not the correct dimension.
     */
    void set_xf(size_type i, const vector_double &x, const vector_double &f)
    {
        if (i >= size()) {
            pagmo_throw(std::invalid_argument, "Trying to access individual at position: " + std::to_string(i)
                                                   + ", while population has size: " + std::to_string(size()));
        }
        if (f.size() != m_prob.get_nf()) {
            pagmo_throw(std::invalid_argument,
                        "Trying to set a fitness of dimension: " + std::to_string(f.size())
                            + ", while the problem's fitness has dimension: " + std::to_string(m_prob.get_nf()));
        }
        if (x.size() != m_prob.get_nx()) {
            pagmo_throw(std::invalid_argument,
                        "Trying to set a decision vector of dimension: " + std::to_string(x.size())
                            + ", while the problem's dimension is: " + std::to_string(m_prob.get_nx()));
        }

        // Reserve space for the incoming vectors. If any of this throws,
        // the data in m_x[i]/m_f[i] will not be modified.
        m_x[i].reserve(x.size());
        m_f[i].reserve(f.size());

        update_champion(x, f);
        // Use resize + std::copy: since we reserved enough space above, none of this
        // can throw.
        m_x[i].resize(x.size());
        m_f[i].resize(f.size());
        std::copy(x.begin(), x.end(), m_x[i].begin());
        std::copy(f.begin(), f.end(), m_f[i].begin());
    }

    /// Sets the \f$i\f$-th individual's chromosome
    /**
     * Sets the chromosome of the \f$i\f$-th individual to the
     * value \p x and changes its fitness accordingly. The
     * individual's ID remains the same.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    A call to this method triggers one fitness function evaluation.
     *
     * \endverbatim
     *
     * @param i individual's index in the population
     * @param x decision vector
     *
     * @throws unspecified any exception thrown by population::set_xf().
     */
    void set_x(size_type i, const vector_double &x)
    {
        set_xf(i, x, m_prob.fitness(x));
    }

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

    /// Streaming operator for the class pagmo::population.
    /**
     * @param os target stream.
     * @param p the population to be directed to stream.
     *
     * @return a reference to \p os.
     */
    friend std::ostream &operator<<(std::ostream &os, const population &p)
    {
        stream(os, p.m_prob, '\n');
        stream(os, "Population size: ", p.size(), "\n\n");
        stream(os, "List of individuals: ", '\n');
        for (size_type i = 0u; i < p.size(); ++i) {
            stream(os, "#", i, ":\n");
            stream(os, "\tID:\t\t\t", p.m_ID[i], '\n');
            stream(os, "\tDecision vector:\t", p.m_x[i], '\n');
            stream(os, "\tFitness vector:\t\t", p.m_f[i], '\n');
        }
        if (p.get_problem().get_nobj() == 1u && !p.get_problem().is_stochastic()) {
            stream(os, "\nChampion decision vector: ", p.champion_x(), '\n');
            stream(os, "Champion fitness: ", p.champion_f(), '\n');
        }
        return os;
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
    void save(Archive &ar) const
    {
        ar(m_prob, m_ID, m_x, m_f, m_champion_x, m_champion_f, m_e, m_seed);
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
    void load(Archive &ar)
    {
        population tmp;
        ar(tmp.m_prob, tmp.m_ID, tmp.m_x, tmp.m_f, tmp.m_champion_x, tmp.m_champion_f, tmp.m_e, tmp.m_seed);
        *this = std::move(tmp);
    }

private:
    // Short routine to update the champion. Does nothing if the problem is MO
    void update_champion(vector_double x, vector_double f)
    {
        assert(f.size() > 0u);
        // If the problem has multiple objectives do nothing
        if (m_prob.get_nobj() == 1u) {
            // If the champion does not exist create it, otherwise update it if worse than the new solution
            if (m_champion_x.size() == 0u) {
                m_champion_x = std::move(x);
                m_champion_f = std::move(f);
            } else if (m_prob.get_nc() == 0u) { // unconstrained
                if (f[0] < m_champion_f[0]) {
                    m_champion_x = std::move(x);
                    m_champion_f = std::move(f);
                }
            } else { // constrained
                if (compare_fc(f, m_champion_f, m_prob.get_nec(), m_prob.get_c_tol())) {
                    m_champion_x = std::move(x);
                    m_champion_f = std::move(f);
                }
            }
        }
    }
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

namespace detail
{

// Machinery for the default implementation of parallel initialisation.
template <typename>
struct pop_parinit {
    using func_t = std::function<std::pair<std::vector<vector_double>, std::vector<vector_double>>(
        const std::string &, problem &, unsigned, population::size_type)>;
    static func_t s_func;
};

// Parallel initialisation via multiple threads.
inline void pop_parinit_thread(std::pair<std::vector<vector_double>, std::vector<vector_double>> &ret, problem &prob,
                               unsigned seed, population::size_type pop_size)
{
    // A few typing shortcuts.
    using rng_type = detail::random_engine_type;
    using rng_seed_type = typename rng_type::result_type;
    using range_type = tbb::blocked_range<decltype(pop_size)>;

    // Check the input retval.
    assert(ret.first.size() == pop_size);
    assert(ret.second.size() == pop_size);

    // Cache the problem bounds and the number of integer variables.
    const auto prob_bounds = prob.get_bounds();
    const auto nix = prob.get_nix();

    // Body of the parallel functors below.
    auto par_body = [&ret, seed, &prob_bounds, nix](const problem &p, const range_type &range) {
        // Init a random engine with a local seed. We will mix
        // the original seed with the limits of the current iteration.
        auto local_seed = static_cast<std::size_t>(seed);
        boost::hash_combine(local_seed, static_cast<std::size_t>(range.begin()));
        boost::hash_combine(local_seed, static_cast<std::size_t>(range.end()));
        rng_type eng(static_cast<rng_seed_type>(local_seed));

        for (auto i = range.begin(); i != range.end(); ++i) {
            // Create a random dv, compute its fitness and move both into ret.
            auto random_dv = pagmo::random_decision_vector(prob_bounds, eng, nix);
            auto fitness = p.fitness(random_dv);
            ret.first[i] = std::move(random_dv);
            ret.second[i] = std::move(fitness);
        }
    };

    if (prob.get_thread_safety() >= thread_safety::constant) {
        // We can concurrently call the objfun on the input prob, hence we can
        // capture it by reference and do all the fitness calls on the same object.
        tbb::parallel_for(range_type(0u, pop_size),
                          [&prob, &par_body](const range_type &range) { par_body(prob, range); });
    } else if (prob.get_thread_safety() == thread_safety::basic) {
        // We cannot concurrently call the objfun on the input prob. We will need
        // to make a copy of prob for each parallel iteration.
        tbb::parallel_for(range_type(0u, pop_size),
                          [prob, &par_body](const range_type &range) { par_body(prob, range); });

        // Update manually the fevals.
        prob.set_fevals(prob.get_fevals() + pop_size);
    } else {
        pagmo_throw(
            std::invalid_argument,
            "The 'par_thread' parallel population initialisation mode was selected, but the problem '" + prob.get_name()
                + "' does not provide the required level of thread safety. Please choose another initialisation mode");
    }
}

inline std::pair<std::vector<vector_double>, std::vector<vector_double>>
default_pop_parinit(const std::string &init_mode, problem &prob, unsigned seed, population::size_type pop_size)
{
    // Prepare the return values.
    std::pair<std::vector<vector_double>, std::vector<vector_double>> retval;
    retval.first.resize(boost::numeric_cast<vector_double::size_type>(pop_size));
    retval.second.resize(boost::numeric_cast<vector_double::size_type>(pop_size));

    if (init_mode == "par_thread" || init_mode == "par_auto") {
        pop_parinit_thread(retval, prob, seed, pop_size);
    } else {
        pagmo_throw(std::invalid_argument, "The '" + init_mode
                                               + "' parallel population initialisation mode is not valid. The valid "
                                                 "modes are 'par_thread' and 'par_auto'");
    }

    return retval;
}

template <typename T>
typename pop_parinit<T>::func_t pop_parinit<T>::s_func = default_pop_parinit;

} // namespace detail

} // namespace pagmo

#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic pop
#endif

#endif
