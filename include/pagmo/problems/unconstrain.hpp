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

#ifndef PAGMO_PROBLEMS_UNCONSTRAIN_HPP
#define PAGMO_PROBLEMS_UNCONSTRAIN_HPP

#include <string>
#include <type_traits>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// The unconstrain meta-problem
/**
 * This meta-problem transforms a constrained problem into an unconstrained problem applying one of the following
 * methods:
 *    - Death penalty: simply penalizes all objectives by the same high value if the fitness vector is infeasible as
 * checked by pagmo::problem::feasibility_f().
 *    - Kuri's death penalty: defined by Angel Kuri Morales et al., penalizes all objectives according to the rate of
 * satisfied constraints.
 *    - Weighted violations penalty: penalizes all objectives by the weighted sum of the constraint violations.
 *    - Ignore the constraints: simply ignores the constraints.
 *    - Ignore the objectives: ignores the objectives and defines as a new single objective the overall constraints
 * violation (i.e. the sum of the L2 norms of the equalities and inequalities violations)
 *
 * See: Coello Coello, C. A. (2002). Theoretical and numerical constraint-handling techniques used with evolutionary
 * algorithms: a survey of the state of the art. Computer methods in applied mechanics and engineering, 191(11),
 * 1245-1287.
 *
 * See: Kuri Morales, A. and Quezada, C.C. A Universal eclectic genetic algorithm for constrained optimization,
 * Proceedings 6th European Congress on Intelligent Techniques & Soft Computing, EUFIT'98, 518-522, 1998.
 */
class PAGMO_DLL_PUBLIC unconstrain
{
    // Enabler for the ctor from UDP or problem. In this case we allow construction from type problem.
    template <typename T>
    using ctor_enabler = enable_if_t<detail::conjunction<detail::negation<std::is_same<unconstrain, uncvref_t<T>>>,
                                                         std::is_constructible<problem, T &&>>::value,
                                     int>;
    // Implementation of the generic ctor.
    void generic_ctor_impl(const std::string &, const vector_double &);

public:
    // Default constructor
    unconstrain();

    /// Constructor from UDP and unconstrain method
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``T`` can be used to construct a :cpp:class:`pagmo::problem`
     *    and if ``T``, after the removal of reference and cv qualifiers, is not :cpp:class:`pagmo::unconstrain`.
     *
     * \endverbatim
     *
     * Wraps a user-defined problem so that its constraints will be removed
     *
     * @param p a user-defined problem or a pagmo::problem.
     * @param method an <tt>std::string</tt> containing the name of the method to be used t remove the constraints: one
     * of "death penalty", "kuri", "weighted", "ignore_c" or "ignore_o".
     * @param weights an <tt>std::vector</tt> containing the weights in case "weighted" is selected as method.
     *
     * @throws std::invalid_argument if the length of \p weights is
     * not equal to the problem constraint dimension \f$ n_{ec} + n_{ic}\f$ when \p method is "weighted", if the
     * \p method is not one of "death penalty", "kuri", "weighted", "ignore_c" or "ignore_o", if the \p weights vector
     * is not empty and the \p method is not "weighted" or if \p is already unconstrained
     * @throws unspecified any exception thrown by the pagmo::problem constructor
     */
    template <typename T, ctor_enabler<T> = 0>
    explicit unconstrain(T &&p, const std::string &method = "death penalty",
                         const vector_double &weights = vector_double())
        : m_problem(std::forward<T>(p)), m_weights(weights)
    {
        generic_ctor_impl(method, weights);
    }

    // Fitness.
    vector_double fitness(const vector_double &) const;

    // Number of objectives.
    vector_double::size_type get_nobj() const;

    // Integer dimension
    vector_double::size_type get_nix() const;

    // Box-bounds.
    std::pair<vector_double, vector_double> get_bounds() const;

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

    // Problem name.
    std::string get_name() const;

    // Extra info.
    std::string get_extra_info() const;

    // Object serialization.
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // The inner problem
    problem m_problem;
    // types of unconstrain methods
    enum class method_type {
        DEATH,    ///< "death penalty"
        KURI,     ///< "kuri"
        WEIGHTED, ///< "weighted"
        IGNORE_C, ///< "ignore_c"
        IGNORE_O  ///< "ignore_o"
    };

    // method used to unconstrain the problem
    method_type m_method;
    // weights vector
    vector_double m_weights;
};
} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::unconstrain)

#endif
