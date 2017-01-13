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

#include "detail/population_fwd.hpp"
#include "problem.hpp"
#include "problems/null_problem.hpp"
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
 * candidate solutions to a given problem. In PaGMO an
 * individual is determined
 * - by a unique ID used to track him across generations and migrations
 * - by a chromosome (a decision vector)
 * - by the fitness of the chromosome as evaluated by a pagmo::problem.
 * and thus including objectives, equality constraints and inequality
 * constraints if present.
 *
 */
class population
{
    // Enable the generic ctor only if T is not a population (after removing
    // const/reference qualifiers).
    template <typename T>
    using generic_ctor_enabler = enable_if_t<!std::is_same<population, uncvref_t<T>>::value, int>;

public:
#if defined(DOXYGEN_INVOKED)
    /// A shortcut to <tt>std::vector<vector_double>::size_type</tt>.
    typedef std::vector<vector_double>::size_type size_type;
#else
    using size_type = std::vector<vector_double>::size_type;
#endif

    /// Default constructor
    /**
     * Constructs an empty population with a pagmo::null_problem.
     * The random seed is initialised to zero.
     */
    population() : m_prob(null_problem{}), m_e(0u), m_seed(0u)
    {
    }

    /// Constructor from a problem of type \p T
    /**
     * Constructs a population with \p pop_size individuals associated
     * to the problem \p x and setting the population random seed
     * to \p seed. In order for the construction to be succesfull, \p x
     * must be such that a pagmo::problem can be constructed from it.
     *
     * @param[in] x the user problem the population refers to
     * @param[in] pop_size population size (i.e. number of individuals therein)
     * @param[in] seed seed of the random number generator used, for example, to
     * create new random individuals within the bounds
     *
     * @throws unspecified any exception thrown by decision_vector() or by push_back()
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit population(T &&x, size_type pop_size = 0u, unsigned int seed = pagmo::random_device::next())
        : m_prob(std::forward<T>(x)), m_e(seed), m_seed(seed)
    {
        for (size_type i = 0u; i < pop_size; ++i) {
            push_back(decision_vector());
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
     * @param[in] other assignment argument.
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
    population &operator=(population &&) = default;

    /// Trivial destructor.
    /**
     * The destructor will run sanity checks in debug mode.
     */
    ~population()
    {
        assert(m_ID.size() == m_x.size());
        assert(m_ID.size() == m_f.size());
    }

    /// Adds one decision vector (chromosome) to the population
    /**
     * Appends a new chromosome \p x to the population, evaluating
     * its fitness and creating a new unique identifier for the newly
     * born individual.
     *
     * In case of exceptions, the population will not be altered.
     *
     * @param[in] x decision vector to be added to the population.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers or by problem::fitness().
     * Wrong dimensions for the input decision vector or the output fitness will trigger an exception.
     */
    void push_back(const vector_double &x)
    {
        // Prepare quantities to be appended to the internal vectors.
        const auto new_id = std::uniform_int_distribution<unsigned long long>()(m_e);
        auto x_copy(x);
        // This line will throw if dv dimensions are wrong, or fitness dimensions are worng
        auto f = m_prob.fitness(x);
        // Reserve space in the vectors.
        // NOTE: in face of overflow here, reserve(0) will be called, which is fine.
        // The first push_back below will then fail, with no modifications to the class taking place.
        m_ID.reserve(m_ID.size() + 1u);
        m_x.reserve(m_x.size() + 1u);
        m_f.reserve(m_f.size() + 1u);
        // The rest is noexcept.
        m_ID.push_back(new_id);
        m_x.push_back(std::move(x_copy));
        m_f.push_back(std::move(f));
    }

    /// Creates a random decision vector
    /**
     * Creates a random decision vector within the problem's bounds.
     * It calls internally pagmo::decision_vector().
     *
     * @returns a random decision vector
     *
     * @throws unspecified all exceptions thrown by pagmo::decision_vector()
     */
    vector_double decision_vector() const
    {
        return pagmo::decision_vector(m_prob.get_bounds(), m_e);
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
     * @param[in] tol vector of tolerances to be applied to each constraints
     *
     * @returns the index of the best individual
     *
     * @throws std::invalid_argument if the problem is multiobjective and thus
     * a best individual is not well defined
     */
    vector_double::size_type best_idx(const vector_double &tol) const
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
        std::vector<vector_double::size_type> indexes(size());
        std::iota(indexes.begin(), indexes.end(), vector_double::size_type(0u));
        return *std::min_element(
            indexes.begin(), indexes.end(),
            [this](vector_double::size_type idx1, vector_double::size_type idx2) { return m_f[idx1] < m_f[idx2]; });
    }

    /// Index of best individual (accounting for a scalar tolerance)
    /**
     * @param[in] tol scalar tolerance to be considered for each constraint
     *
     * @return index of the best individual
     */
    vector_double::size_type best_idx(double tol = 0.) const
    {
        vector_double tol_vector(m_prob.get_nf() - 1u, tol);
        return best_idx(tol_vector);
    }

    /// Index of worst individual (accounting for a vector tolerance)
    /**
     * If the problem is single-objective and unconstrained, the worst
     * is simply the individual with the largest fitness. If the problem
     * is, instead, single objective, but with constraints, the best individual
     * will be defined using the criteria specified in pagmo::sort_population_con().
     * If the problem is multi-objective one single worst is not defined. In
     * this case the user can still obtain a strict ordering of the population
     * individuals by calling the pagmo::sort_population_mo() function.
     *
     * @param[in] tol vector of tolerances to be applied to each constraints
     *
     * @returns the index of the best individual
     *
     * @throws std::invalid_argument if the problem is multiobjective and thus
     * a best individual is not well defined
     */
    vector_double::size_type worst_idx(const vector_double &tol) const
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
        std::vector<vector_double::size_type> indexes(size());
        std::iota(indexes.begin(), indexes.end(), vector_double::size_type(0u));
        return *std::max_element(
            indexes.begin(), indexes.end(),
            [this](vector_double::size_type idx1, vector_double::size_type idx2) { return m_f[idx1] < m_f[idx2]; });
    }

    /// Index of worst individual (accounting for a scalar tolerance)
    /**
     * @param[in] tol scalar tolerance to be considered for each constraint
     *
     * @return index of the best individual
     */
    vector_double::size_type worst_idx(double tol = 0.) const
    {
        vector_double tol_vector(m_prob.get_nf() - 1u, tol);
        return worst_idx(tol_vector);
    }

    /// Number of individuals in the population
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
     * @note: The user must make sure that the input fitness \p f makes sense
     * as pagmo will only check its dimension.
     *
     * @param[in] i individual's index in the population
     * @param[in] x a decision vector (chromosome)
     * @param[in] f a fitness vector
     *
     * @throws std::invalid_argument if \p i is invalid (i.e. larger or equal to the population size)
     * @throws std::invalid_argument if \p x has not the correct dimension
     * @throws std::invalid_argument if \p f has not the correct dimension
     */
    void set_xf(size_type i, const vector_double &x, const vector_double &f)
    {
        if (i >= size()) {
            pagmo_throw(std::invalid_argument, "Trying to access individual at position: " + std::to_string(i)
                                                   + ", while population has size: " + std::to_string(size()));
        }
        if (f.size() != m_prob.get_nf()) {
            pagmo_throw(std::invalid_argument, "Trying to set a fitness of dimension: " + std::to_string(f.size())
                                                   + ", while problem get_nf returns: "
                                                   + std::to_string(m_prob.get_nf()));
        }
        if (x.size() != m_prob.get_nx()) {
            pagmo_throw(std::invalid_argument, "Trying to set a decision vector of dimension: "
                                                   + std::to_string(x.size()) + ", while problem get_nx returns: "
                                                   + std::to_string(m_prob.get_nx()));
        }
        assert(m_x[i].size() == x.size());
        assert(m_f[i].size() == f.size());
        // Use std::copy in order to make sure we are not allocating and
        // potentially throw.
        std::copy(x.begin(), x.end(), m_x[i].begin());
        std::copy(f.begin(), f.end(), m_f[i].begin());
    }

    /// Sets the \f$i\f$-th individual's chromosome
    /**
     * Sets the chromosome of the \f$i\f$-th individual to the
     * value \p x and changes its fitness accordingly. The
     * individual's ID remains the same.
     *
     * @note a call to this method triggers one fitness function evaluation
     *
     * @param[in] i individual's index in the population
     * @param[in] x decision vector
     *
     * @throws unspecified any exception thrown by set_xf
     */
    void set_x(size_type i, const vector_double &x)
    {
        set_xf(i, x, m_prob.fitness(x));
    }

    /// Setter for the problem seed
    void set_problem_seed(unsigned int seed)
    {
        m_prob.set_seed(seed);
    }

    /// Getter for the pagmo::problem
    const problem &get_problem() const
    {
        return m_prob;
    }

    /// Getter for the fitness vectors
    const std::vector<vector_double> &get_f() const
    {
        return m_f;
    }

    /// Getter for the decision vectors
    const std::vector<vector_double> &get_x() const
    {
        return m_x;
    }

    /// Getter for the individual IDs
    const std::vector<unsigned long long> &get_ID() const
    {
        return m_ID;
    }

    /// Getter for the seed of the population random engine
    unsigned int get_seed() const
    {
        return m_seed;
    }

    /// Streaming operator for the class pagmo::population
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
        return os;
    }
    /// Serialization.
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_prob, m_ID, m_x, m_f, m_e, m_seed);
    }

private:
    // Problem.
    problem m_prob;
    // ID of the various decision vectors
    std::vector<unsigned long long> m_ID;
    // Decision vectors.
    std::vector<vector_double> m_x;
    // Fitness vectors.
    std::vector<vector_double> m_f;
    // Random engine.
    mutable detail::random_engine_type m_e;
    // Seed.
    unsigned int m_seed;
};

} // namespace pagmo

#endif
