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

#ifndef PAGMO_PROBLEMS_DECOMPOSE_HPP
#define PAGMO_PROBLEMS_DECOMPOSE_HPP

#include <string>
#include <type_traits>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

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
class PAGMO_DLL_PUBLIC decompose
{
public:
    // Default constructor.
    decompose();

private:
    // Enabler for the ctor from UDP or problem. In this case we allow construction from type problem.
    template <typename T>
    using ctor_enabler = enable_if_t<std::is_constructible<problem, T &&>::value, int>;
    void generic_ctor_impl(const vector_double &, const vector_double &, const std::string &);

public:
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
        generic_ctor_impl(weight, z, method);
    }

    // Fitness computation.
    vector_double fitness(const vector_double &) const;

    // Fitness of the original problem.
    vector_double original_fitness(const vector_double &) const;

    /// Number of objectives.
    /**
     * @return one.
     */
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }

    // Integer dimension
    vector_double::size_type get_nix() const;

    // Box-bounds.
    std::pair<vector_double, vector_double> get_bounds() const;

    // Gets the current reference point.
    vector_double get_z() const;

    // Problem name.
    std::string get_name() const;

    // Extra information.
    std::string get_extra_info() const;

    // Calls <tt>has_set_seed()</tt> of the inner problem.
    bool has_set_seed() const;

    // Calls <tt>set_seed()</tt> of the inner problem.
    void set_seed(unsigned);

    // Problem's thread safety level.
    thread_safety get_thread_safety() const;

    // Getter for the inner problem.
    const problem &get_inner_problem() const;

    // Getter for the inner problem.
    problem &get_inner_problem();

    // Object serialization.
    template <typename Archive>
    void serialize(Archive &, unsigned);

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

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::decompose)

#endif
