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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <pagmo/bfe.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/generic.hpp>

namespace pagmo
{

/// Default constructor
/**
 * Constructs an empty population with a default-constructed problem.
 * The random seed is initialised to zero.
 *
 * @throws unspecified any exception thrown by the constructor from problem.
 */
population::population() : population(problem{}, 0u, 0u) {}

void population::prob_ctor_impl(size_type pop_size)
{
    // NOTE: generate the random decision vectors in temporary storage,
    // and only at the end move them into the population. This ensures
    // that, for a given rng seed, the generated dvs are identical
    // to those generated in the constructor from bfe.
    std::vector<std::pair<vector_double, vector_double>> tmp(pop_size);
    for (size_type i = 0u; i < pop_size; ++i) {
        tmp[i].first = random_decision_vector();
        tmp[i].second = m_prob.fitness(tmp[i].first);
    }
    // Move the generated dvs/fvs into the population. push_back()
    // will take care of generating the IDs, updating the champion, etc.
    for (size_type i = 0u; i < pop_size; ++i) {
        push_back(std::move(tmp[i].first), std::move(tmp[i].second));
    }
}

// Implementation of the ctor from bfe. Distinguish the two cases
// in which bfe or a udbfe were provided.
void population::constructor_from_bfe_impl(const bfe &b, size_type pop_size, const std::true_type &)
{
    // Create a batch of random decision vectors.
    const auto dvs = batch_random_decision_vector(m_prob, pop_size, m_e);

    // Evaluate them in batch mode.
    const auto fvs = b(m_prob, dvs);

    // Sanity checks.
    assert(pop_size == 0u || dvs.size() % pop_size == 0u);
    assert(pop_size == 0u || fvs.size() % pop_size == 0u);

    // Add the dvs/fvs to the population.
    const auto nx = m_prob.get_nx();
    const auto nf = m_prob.get_nf();
    assert(dvs.size() % nx == 0u);
    assert(fvs.size() % nf == 0u);
    for (size_type i = 0; i < pop_size; ++i) {
        push_back(vector_double(dvs.data() + i * nx, dvs.data() + (i + 1u) * nx),
                  vector_double(fvs.data() + i * nf, fvs.data() + (i + 1u) * nf));
    }
}

/// Defaulted copy constructor.
population::population(const population &) = default;

/// Move constructor.
/**
 * @param pop construction argument.
 */
population::population(population &&pop) noexcept
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
population &population::operator=(const population &other)
{
    if (this != &other) {
        *this = population(other);
    }
    return *this;
}

/// Move assignment operator.
/**
 * @param pop assignment argument.
 *
 * @return a reference to \p this.
 */
population &population::operator=(population &&pop) noexcept
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
population::~population()
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
 * @throws std::overflow_error if the addition of ``x`` to the population would overflow the population size limit.
 * @throws std::invalid_argument if the size of ``x`` differs from the problem's dimension.
 * @throws unspecified any exception thrown by memory errors in standard containers or by problem::fitness().
 */
void population::push_back(const vector_double &x)
{
    push_back_impl(x, m_prob.fitness(x));
}

/// Adds one decision vector (chromosome) to the population (move overload).
/**
 * This overload behaves like the previous one, the only difference being that ``x``
 * will be moved (rather than copied) into the population.
 *
 * @param x decision vector to be added to the population.
 *
 * @throws unspecified any exception thrown by the previous ``push_back()`` overload.
 */
void population::push_back(vector_double &&x)
{
    push_back_impl(std::move(x), m_prob.fitness(x));
}

/// Adds one decision vector/fitness vector to the population.
/**
 * Appends a new chromosome \p x to the population, and sets
 * its fitness to \p f creating a new unique identifier for the newly
 * born individual.
 *
 * In case of exceptions, the population will not be altered.
 *
 * @param x decision vector to be added to the population.
 * @param f fitness vector corresponding to the decision vector.
 *
 * @throws std::overflow_error if the addition of ``x`` to the population would overflow the population size limit.
 * @throws std::invalid_argument if the size of ``x`` differs from the problem's dimension,
 * or if the size of ``f`` differs from the fitness dimension.
 * @throws unspecified any exception thrown by memory errors in standard containers.
 */
void population::push_back(const vector_double &x, const vector_double &f)
{
    push_back_impl(x, f);
}

/// Adds one decision vector/fitness vector to the population (move overload).
/**
 * This overload behaves like the previous one, the only difference being that ``x``
 * and ``f`` will be moved (rather than copied) into the population.
 *
 * @param x decision vector to be added to the population.
 * @param f fitness vector corresponding to the decision vector.
 *
 * @throws unspecified any exception thrown by the previous ``push_back()`` overload.
 */
void population::push_back(vector_double &&x, vector_double &&f)
{
    push_back_impl(std::move(x), std::move(f));
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
vector_double population::random_decision_vector() const
{
    return pagmo::random_decision_vector(m_prob, m_e);
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
 * @throws std::overflow_error if the size of the population exceeds an implementation-defined limit.
 * @throws std::invalid_argument if the problem is multiobjective and thus
 * a best individual is not well defined, or if the population is empty.
 * @throws unspecified any exception thrown by pagmo::sort_population_con().
 */
population::size_type population::best_idx() const
{
    return best_idx(get_problem().get_c_tol());
}

/// Index of the best individual (accounting for a vector tolerance)
/**
 * @param tol vector of tolerances to be applied to each constraints.
 *
 * @returns the index of the best individual.
 *
 * @throws std::overflow_error if the size of the population exceeds an implementation-defined limit.
 * @throws std::invalid_argument if the problem is multiobjective and thus
 * a best individual is not well defined, or if the population is empty.
 * @throws unspecified any exception thrown by pagmo::sort_population_con().
 */
population::size_type population::best_idx(const vector_double &tol) const
{
    if (!size()) {
        pagmo_throw(std::invalid_argument, "Cannot determine the best individual of an empty population");
    }
    if (m_prob.get_nobj() > 1u) {
        pagmo_throw(std::invalid_argument, "The best individual can only be extracted in single objective problems");
    }
    if (m_prob.get_nc() > 0u) { // TODO: should we also code a min_element_population_con?
        return sort_population_con(m_f, m_prob.get_nec(), tol)[0];
    }
    // Overflow check on the iterator diff type.
    using it_diff_t = std::iterator_traits<decltype(m_f.begin())>::difference_type;
    using it_udiff_t = std::make_unsigned<it_diff_t>::type;
    // Check that we can represent any index in the population via the iterator difference type.
    // NOTE: size - 1 is fine, as we know that here size cannot be zero.
    // LCOV_EXCL_START
    if (m_f.size() - 1u > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
        pagmo_throw(std::overflow_error, "The size of the population, " + std::to_string(m_f.size())
                                             + ", is too large, and it results in an overflow condition when "
                                               "trying to determine the index of the best individual");
    }
    // LCOV_EXCL_STOP
    return static_cast<size_type>(std::min_element(m_f.begin(), m_f.end()) - m_f.begin());
}

/// Index of the best individual (accounting for a scalar tolerance)
/**
 * @param tol scalar tolerance to be considered for each constraint.
 *
 * @return index of the best individual.
 *
 * @throws unspecified any exception thrown by the previous ``best_idx()`` overload.
 */
population::size_type population::best_idx(double tol) const
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
 * @throws std::overflow_error if the size of the population exceeds an implementation-defined limit.
 * @throws std::invalid_argument if the problem is multiobjective and thus
 * a worst individual is not well defined, or if the population is empty.
 * @throws unspecified any exception thrown by pagmo::sort_population_con().
 */
population::size_type population::worst_idx() const
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
 * @throws std::overflow_error if the size of the population exceeds an implementation-defined limit.
 * @throws std::invalid_argument if the problem is multiobjective and thus
 * a worst individual is not well defined, or if the population is empty.
 * @throws unspecified any exception thrown by pagmo::sort_population_con().
 */
population::size_type population::worst_idx(const vector_double &tol) const
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
    // Overflow check on the iterator diff type.
    using it_diff_t = std::iterator_traits<decltype(m_f.begin())>::difference_type;
    using it_udiff_t = std::make_unsigned<it_diff_t>::type;
    // Check that we can represent any index in the population via the iterator difference type.
    // NOTE: size - 1 is fine, as we know that here size cannot be zero.
    // LCOV_EXCL_START
    if (m_f.size() - 1u > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
        pagmo_throw(std::overflow_error, "The size of the population, " + std::to_string(m_f.size())
                                             + ", is too large, and it results in an overflow condition when "
                                               "trying to determine the index of the worst individual");
    }
    // LCOV_EXCL_STOP
    return static_cast<size_type>(std::max_element(m_f.begin(), m_f.end()) - m_f.begin());
}

/// Index of the worst individual (accounting for a scalar tolerance)
/**
 * @param tol scalar tolerance to be considered for each constraint.
 *
 * @return index of the worst individual.
 *
 * @throws unspecified any exception thrown by the previous ``worst_idx()`` overload.
 */
population::size_type population::worst_idx(double tol) const
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
vector_double population::champion_x() const
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
vector_double population::champion_f() const
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
void population::set_xf(size_type i, const vector_double &x, const vector_double &f)
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
void population::set_x(size_type i, const vector_double &x)
{
    set_xf(i, x, m_prob.fitness(x));
}

/// Streaming operator for pagmo::population.
/**
 * @param os target stream.
 * @param p the population to be directed to stream.
 *
 * @return a reference to \p os.
 */
std::ostream &operator<<(std::ostream &os, const population &p)
{
    stream(os, p.get_problem(), '\n');
    stream(os, "Population size: ", p.size(), "\n\n");
    stream(os, "List of individuals: ", '\n');
    for (population::size_type i = 0u; i < p.size(); ++i) {
        stream(os, "#", i, ":\n");
        stream(os, "\tID:\t\t\t", p.get_ID()[i], '\n');
        stream(os, "\tDecision vector:\t", p.get_x()[i], '\n');
        stream(os, "\tFitness vector:\t\t", p.get_f()[i], '\n');
    }
    if (p.get_problem().get_nobj() == 1u && !p.get_problem().is_stochastic()) {
        stream(os, "\nChampion decision vector: ", p.champion_x(), '\n');
        stream(os, "Champion fitness: ", p.champion_f(), '\n');
    }
    return os;
}

template <typename T, typename U>
void population::push_back_impl(T &&x, U &&f)
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
    // LCOV_EXCL_START
    if (m_ID.size() == std::numeric_limits<decltype(m_ID.size())>::max()
        || m_x.size() == std::numeric_limits<decltype(m_x.size())>::max()) {
        pagmo_throw(std::overflow_error, "Cannot add a new individual to this population: the maximum number of "
                                         "individuals per population has been reached");
    }
    // LCOV_EXCL_STOP

    // Prepare quantities to be appended to the internal vectors.
    const auto new_id = std::uniform_int_distribution<unsigned long long>()(m_e);
    auto x_copy(std::forward<T>(x));
    auto f_copy(std::forward<U>(f));
    // Reserve space in the vectors.
    m_ID.reserve(m_ID.size() + 1u);
    m_x.reserve(m_x.size() + 1u);
    m_f.reserve(m_f.size() + 1u);

    // update_champion() either throws before modfying anything, or it completes successfully. The rest is noexcept.
    update_champion(x_copy, f_copy);
    m_ID.push_back(new_id);
    m_x.push_back(std::move(x_copy));
    m_f.push_back(std::move(f_copy));
}

// Short routine to update the champion. Does nothing if the problem is MO
void population::update_champion(vector_double x, vector_double f)
{
    assert(f.size() > 0u);
    // If the problem has multiple objectives do nothing
    if (m_prob.get_nobj() != 1u) {
        return;
    }
    // If the champion does not exist create it, otherwise update it if worse than the new solution
    if (m_champion_x.size() == 0u) {
        m_champion_x = std::move(x);
        m_champion_f = std::move(f);
    } else if (m_prob.get_nc() == 0u) { // unconstrained
        // NOTE: make sure to use the custom comparison less_than_f(),
        // so that we handle NaN correctly (i.e., a fitness of NaN is
        // considered worse than any other value).
        if (detail::less_than_f(f[0], m_champion_f[0])) {
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

// Small helper to erase all individuals from a population.
void population::clear()
{
    m_ID.clear();
    m_x.clear();
    m_f.clear();
}

} // namespace pagmo
