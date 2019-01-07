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

#ifndef PAGMO_PROBLEM_DECOMPOSE_HPP
#define PAGMO_PROBLEM_DECOMPOSE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp> // pagmo::decompose_objectives

namespace pagmo
{

/// The decompose meta-problem.
/**
 * \image html decompose.png "Decomposition." width=3cm
 *
 * This meta-problem *decomposes* a multi-objective input user-defined problem,
 * resulting in a single-objective user-defined problem with a fitness function combining the
 * original fitness functions. In particular, three different *decomposition methods* are here
 * made available:
 *
 * - weighted decomposition,
 * - Tchebycheff decomposition,
 * - boundary interception method (with penalty constraint).
 *
 * In the case of \f$n\f$ objectives, we indicate with: \f$ \mathbf f(\mathbf x) = [f_1(\mathbf x), \ldots, f_n(\mathbf
 * x)] \f$ the vector containing the original multiple objectives, with: \f$ \boldsymbol \lambda = (\lambda_1, \ldots,
 * \lambda_n) \f$ an \f$n\f$-dimensional weight vector and with: \f$ \mathbf z^* = (z^*_1, \ldots, z^*_n) \f$
 * an \f$n\f$-dimensional reference point. We also ussume \f$\lambda_i > 0, \forall i=1..n\f$ and \f$\sum_i \lambda_i =
 * 1\f$.
 *
 * The decomposed problem is thus a single objective optimization problem having the following single objective,
 * according to the decomposition method chosen:
 *
 * - weighted decomposition: \f$ f_d(\mathbf x) = \boldsymbol \lambda \cdot \mathbf f \f$,
 * - Tchebycheff decomposition: \f$ f_d(\mathbf x) = \max_{1 \leq i \leq m} \lambda_i \vert f_i(\mathbf x) - z^*_i \vert
 * \f$,
 * - boundary interception method (with penalty constraint): \f$ f_d(\mathbf x) = d_1 + \theta d_2\f$,
 *
 * where \f$d_1 = (\mathbf f - \mathbf z^*) \cdot \hat {\mathbf i}_{\lambda}\f$,
 * \f$d_2 = \vert (\mathbf f - \mathbf z^*) - d_1 \hat {\mathbf i}_{\lambda})\vert\f$ and
 * \f$ \hat {\mathbf i}_{\lambda} = \frac{\boldsymbol \lambda}{\vert \boldsymbol \lambda \vert}\f$.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The reference point :math:`z^*` is often taken as the ideal point and as such
 *    it may be allowed to change during the course of the optimization / evolution. The argument adapt_ideal activates
 *    this behaviour so that whenever a new ideal point is found :math:`z^*` is adapted accordingly.
 *
 * .. note::
 *
 *    The use of :cpp:class:`pagmo::decompose` discards gradients and hessians so that if the original user defined
 *    problem implements them, they will not be available in the decomposed problem. The reason for this behaviour is
 *    that the Tchebycheff decomposition is not differentiable. Also, the use of this class was originally intended for
 *    derivative-free optimization.
 *
 * .. seealso::
 *
 *    "Q. Zhang -- MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"
 *    https://en.wikipedia.org/wiki/Multi-objective_optimization#Scalarizing
 *
 * \endverbatim
 */
class decompose
{
    // Enabler for the ctor from UDP or problem. In this case we allow construction from type problem.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<problem, T &&>::value, int>;

public:
    // Default constructor.
    /**
     * The default constructor will initialize \p this with a 2-objectives pagmo::null_problem,
     * weight vector <tt>[0.5, 0.5]</tt> and reference point <tt>[0., 0.]</tt>.
     *
     * @throws unspecified any exception thrown by the other constructor.
     */
    decompose() : decompose(null_problem{2u}, {0.5, 0.5}, {0., 0.}) {}
    /// Constructor from problem.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``T`` can be used to construct a :cpp:class:`pagmo::problem`.
     *
     * \endverbatim
     *
     * Wraps a user-defined problem (UDP) or a pagmo::problem so that its fitness will be decomposed using one of three
     * decomposition methods. pagmo::decompose objects are user-defined problems that can be used
     * to define a pagmo::problem.
     *
     * @param p the input UDP or pagmo::problem.
     * @param weight the vector of weights \f$\boldsymbol \lambda\f$.
     * @param z the reference point \f$\mathbf z^*\f$.
     * @param method an \p std::string containing the decomposition method chosen.
     * @param adapt_ideal when \p true, the reference point is adapted at each fitness evaluation to be the ideal point.
     *
     * @throws std::invalid_argument if either:
     * - \p p is single objective or constrained,
     * - \p method is not one of <tt>["weighted", "tchebycheff", "bi"]</tt>,
     * - \p weight is not of size \f$n\f$,
     * - \p z is not of size \f$n\f$,
     * - \p weight is not such that \f$\lambda_i > 0, \forall i=1..n\f$,
     * - \p weight is not such that \f$\sum_i \lambda_i = 1\f$.
     */
    template <typename T, ctor_enabler<T> = 0>
    explicit decompose(T &&p, const vector_double &weight, const vector_double &z,
                       const std::string &method = "weighted", bool adapt_ideal = false)
        : m_problem(std::forward<T>(p)), m_weight(weight), m_z(z), m_method(method), m_adapt_ideal(adapt_ideal)
    {
        const auto original_fitness_dimension = m_problem.get_nobj();
        // 0 - we check that the problem is multiobjective and unconstrained
        if (original_fitness_dimension < 2u) {
            pagmo_throw(std::invalid_argument, "Decomposition can only be applied to multi-objective problems");
        }
        if (m_problem.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument, "Decomposition can only be applied to unconstrained problems, it seems "
                                               "you are trying to decompose a problem with "
                                                   + std::to_string(m_problem.get_nc()) + " constraints");
        }
        // 1 - we check that the decomposition method is one of "weighted", "tchebycheff" or "bi"
        if (method != "weighted" && method != "tchebycheff" && method != "bi") {
            pagmo_throw(std::invalid_argument,
                        "Decomposition method requested is: " + method
                            + " while only one of ['weighted', 'tchebycheff', 'bi'] are allowed");
        }
        // 2 - we check the sizes of the input weight vector and of the reference point and forbids inf and nan
        if (weight.size() != original_fitness_dimension) {
            pagmo_throw(
                std::invalid_argument,
                "Weight vector size must be equal to the number of objectives. The size of the weight vector is "
                    + std::to_string(weight.size()) + " while the problem has "
                    + std::to_string(original_fitness_dimension) + " objectives");
        }
        for (auto item : weight) {
            if (!std::isfinite(item)) {
                pagmo_throw(std::invalid_argument, "Weight contains non finite numbers");
            }
        }
        if (z.size() != original_fitness_dimension) {
            pagmo_throw(
                std::invalid_argument,
                "Reference point size must be equal to the number of objectives. The size of the reference point is "
                    + std::to_string(z.size()) + " while the problem has " + std::to_string(original_fitness_dimension)
                    + " objectives");
        }
        for (auto item : z) {
            if (!std::isfinite(item)) {
                pagmo_throw(std::invalid_argument, "Reference point contains non finite numbers");
            }
        }

        // 3 - we check that the weight vector is normalized.
        auto sum = std::accumulate(weight.begin(), weight.end(), 0.);
        if (std::abs(sum - 1.0) > 1E-8) {
            pagmo_throw(std::invalid_argument, "The weight vector must sum to 1 with a tolerance of 1E-8. The sum of "
                                               "the weight vector components was detected to be: "
                                                   + std::to_string(sum));
        }
        // 4 - we check the weight vector only contains positive numbers
        for (decltype(m_weight.size()) i = 0u; i < m_weight.size(); ++i) {
            if (m_weight[i] < 0.) {
                pagmo_throw(std::invalid_argument, "The weight vector may contain only non negative values. A value of "
                                                       + std::to_string(m_weight[i]) + " was detected at index "
                                                       + std::to_string(i));
            }
        }
    }
    /// Fitness computation.
    /**
     * The fitness values returned by the inner problem will be combined using the decomposition method selected during
     * construction.
     *
     * @param x the decision vector.
     *
     * @return the decomposed fitness of \p x.
     *
     * @throws unspecified any exception thrown by decompose::original_fitness(), or by the fitness decomposition.
     */
    vector_double fitness(const vector_double &x) const
    {
        // we compute the fitness of the original multiobjective problem
        auto f = original_fitness(x);
        // if necessary we update the reference point
        if (m_adapt_ideal) {
            for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
                if (f[i] < m_z[i]) {
                    m_z[i] = f[i]; // its mutable so its ok
                }
            }
        }
        // we return the decomposed fitness
        return decompose_objectives(f, m_weight, m_z, m_method);
    }
    /// Fitness of the original problem.
    /**
     * Returns the fitness of the original multi-objective problem used to construct the decomposed problem.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This is *not* the fitness of the decomposed problem. Such a fitness is instead returned by calling
     *    :cpp:func:`decompose::fitness()`.
     *
     * \endverbatim
     *
     * @param x input decision vector.
     *
     * @returns the fitness of the original multi-objective problem.
     *
     * @throws unspecified any exception thrown by the original fitness computation.
     */
    vector_double original_fitness(const vector_double &x) const
    {
        // We call the fitness of the original multiobjective problem
        return m_problem.fitness(x);
    }

    /// Number of objectives.
    /**
     * @return one.
     */
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }

    /// Integer dimension
    /**
     * @return the integer dimension of the inner problem.
     */
    vector_double::size_type get_nix() const
    {
        return m_problem.get_nix();
    }

    /// Box-bounds.
    /**
     * Forwards the bounds computations to the inner pagmo::problem.
     *
     * @return the lower and upper bounds for each of the decision vector components.
     *
     * @throws unspecified any exception thrown by problem::get_bounds().
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return m_problem.get_bounds();
    }

    /// Gets the current reference point.
    /**
     * The reference point to be used for the decomposition. This is only
     * used for Tchebycheff and boundary interception decomposition methods.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The reference point is adapted (and thus may change) at each call of the fitness.
     *
     * \endverbatim
     *
     * @return the reference point.
     */
    vector_double get_z() const
    {
        return m_z;
    }

    /// Problem name.
    /**
     * This method will append <tt>[decomposed]</tt> to the name of the inner problem.
     *
     * @return a string containing the problem name.
     */
    std::string get_name() const
    {
        return m_problem.get_name() + " [decomposed]";
    }

    /// Extra information.
    /**
     * This method will add info about the decomposition method to the extra info provided
     * by the inner problem.
     *
     * @return a string containing extra information on the problem.
     */
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        stream(oss, "\n\tDecomposition method: ", m_method, "\n\tDecomposition weight: ", m_weight,
               "\n\tDecomposition reference: ", m_z, "\n\tIdeal point adaptation: ", m_adapt_ideal, "\n");
        return m_problem.get_extra_info() + oss.str();
    }

    /// Calls <tt>has_set_seed()</tt> of the inner problem.
    /**
     * Calls the method <tt>has_set_seed()</tt> of the inner problem.
     *
     * @return a flag signalling wether the inner problem is stochastic.
     */
    bool has_set_seed() const
    {
        return m_problem.has_set_seed();
    }

    /// Calls <tt>set_seed()</tt> of the inner problem.
    /**
     * Calls the method <tt>set_seed()</tt> of the inner problem.
     *
     * @param seed seed to be set.
     *
     * @throws std::not_implemented_error if the inner problem is not stochastic.
     */
    void set_seed(unsigned seed)
    {
        return m_problem.set_seed(seed);
    }

    /// Problem's thread safety level.
    /**
     * The thread safety of a meta-problem is defined by the thread safety of the inner pagmo::problem.
     *
     * @return the thread safety level of the inner pagmo::problem.
     */
    thread_safety get_thread_safety() const
    {
        return m_problem.get_thread_safety();
    }

    /// Getter for the inner problem.
    /**
     * Returns a const reference to the inner pagmo::problem.
     *
     * @return a const reference to the inner pagmo::problem.
     */
    const problem &get_inner_problem() const
    {
        return m_problem;
    }

    /// Getter for the inner problem.
    /**
     * Returns a reference to the inner pagmo::problem.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The ability to extract a non const reference is provided only in order to allow to call
     *    non-const methods on the internal :cpp:class:`pagmo::problem` instance. Assigning a new
     *    :class:`pagmo::problem` via this reference is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a reference to the inner pagmo::problem.
     */
    problem &get_inner_problem()
    {
        return m_problem;
    }

    /// Object serialization.
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the inner problem and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_problem, m_weight, m_z, m_method, m_adapt_ideal);
    }

private:
    // Inner problem
    problem m_problem;
    // decomposition weight
    vector_double m_weight;
    // decomposition reference point (only relevant/used for tchebycheff and boundary interception)
    mutable vector_double m_z;
    // decomposition method
    std::string m_method;
    // adapts the decomposition reference point whenever a better point is computed
    bool m_adapt_ideal;
};
} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::decompose)

#endif
