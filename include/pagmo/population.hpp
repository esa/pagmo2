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

#ifndef PAGMO_POPULATION_HPP
#define PAGMO_POPULATION_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "problem.hpp"
#include "rng.hpp"
#include "serialization.hpp"
#include "type_traits.hpp"
#include "types.hpp"
#include "utils/constrained.hpp"
#include "utils/generic.hpp"

namespace pagmo
{
/// Population class.
/**
 * \image html population.jpg
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
 * currently contained in the pagmo::population is single objective (i.e., population::get_problem().get_nobj()
 * returns exactly 1).
 */
class population
{
    // Enable the generic ctor only if T is not a population (after removing
    // const/reference qualifiers).
    template <typename T>
    using generic_ctor_enabler = enable_if_t<!std::is_same<population, uncvref_t<T>>::value, int>;

public:
    /// The size type of the population.
    /**
     * A shortcut to <tt>std::vector<vector_double>::size_type</tt>.
     */
    typedef std::vector<vector_double>::size_type size_type;
    /// Default constructor
    /**
     * Constructs an empty population with a pagmo::null_problem.
     * The random seed is initialised to zero.
     */
    population() : population(null_problem{}, 0u, 0u)
    {
    }

    /// Constructor from a problem.
    /**
     * **NOTE**: this constructor is enabled only if, after the removal of cv/reference qualifiers,
     * \p T is not pagmo::population.
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
        for (size_type i = 0u; i < pop_size; ++i) {
            push_back(random_decision_vector());
        }
    }

    /// Defaulted copy constructor.
    population(const population &) = default;

    /// Defaulted move constructor.
    population(population &&) = default;

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
     * @return a reference to \p this.
     */
    population &operator=(population &&) = default;

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
     * @throws unspecified any exception thrown by memory errors in standard containers.
     */
    void push_back(const vector_double &x, const vector_double &f)
    {
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
        return pagmo::random_decision_vector(m_prob.get_bounds(), m_e);
    }

    /// Index of best individual (accounting for a vector tolerance)
    /**
     * If the problem is single-objective and unconstrained, the best
     * is simply the individual with the smallest fitness. If the problem
     * is, instead, single objective, but with constraints, the best
     * will be defined using the criteria specified in pagmo::sort_population_con().
     * If the problem is multi-objective one single best is not well defined. In
     * this case the user can still obtain a strict ordering of the population
     * individuals by calling the pagmo::sort_population_mo() function.
     *
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

    /// Index of best individual (accounting for a scalar tolerance)
    /**
     * @param tol scalar tolerance to be considered for each constraint.
     *
     * @return index of the best individual.
     */
    size_type best_idx(double tol = 0.) const
    {
        vector_double tol_vector(m_prob.get_nf() - 1u, tol);
        return best_idx(tol_vector);
    }

    /// Index of worst individual (accounting for a vector tolerance)
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

    /// Index of worst individual (accounting for a scalar tolerance)
    /**
     * @param tol scalar tolerance to be considered for each constraint.
     *
     * @return index of the worst individual.
     */
    size_type worst_idx(double tol = 0.) const
    {
        vector_double tol_vector(m_prob.get_nf() - 1u, tol);
        return worst_idx(tol_vector);
    }

    /// Champion decision vector
    /**
     * @return the champion decision vector.
     *
     * @throw std::invalid_argument if the current problem is not single objective.
     */
    vector_double champion_x() const
    {
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        "The Champion of a population can only be extracted in single objective problems");
        }
        return m_champion_x;
    }

    /// Champion fitness
    /**
     * @return the champion fitness.
     *
     * @throw std::invalid_argument if the current problem is not single objective.
     */
    vector_double champion_f() const
    {
        if (m_prob.get_nobj() > 1u) {
            pagmo_throw(std::invalid_argument,
                        "The Champion of a population can only be extracted in single objective problems");
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
     * **NOTE**: The user must make sure that the input fitness \p f makes sense
     * as pagmo will only check its dimension.
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
            pagmo_throw(std::invalid_argument, "Trying to set a fitness of dimension: " + std::to_string(f.size())
                                                   + ", while the problem's fitness has dimension: "
                                                   + std::to_string(m_prob.get_nf()));
        }
        if (x.size() != m_prob.get_nx()) {
            pagmo_throw(std::invalid_argument, "Trying to set a decision vector of dimension: "
                                                   + std::to_string(x.size()) + ", while the problem's dimension is: "
                                                   + std::to_string(m_prob.get_nx()));
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
     * **NOTE** a call to this method triggers one fitness function evaluation.
     *
     * @param i individual's index in the population
     * @param x decision vector
     *
     * @throws unspecified any exception thrown by set_xf
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
        if (p.get_problem().get_nobj() == 1u) {
            stream(os, "\nChampion decision vector: ", p.champion_x(), '\n');
            stream(os, "Champion fitness: ", p.champion_f(), '\n');
        }
        return os;
    }
    /// Serialization.
    /**
     * @param ar the target archive.
     *
     * @throws unspecified any exception thrown by the serialization of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_prob, m_ID, m_x, m_f, m_champion_x, m_champion_f, m_e, m_seed);
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
            } else if (f[0] < m_champion_f[0]) {
                m_champion_x = std::move(x);
                m_champion_f = std::move(f);
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

} // namespace pagmo

#endif
