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

#ifndef PAGMO_PROBLEM_UNCONSTRAIN_HPP
#define PAGMO_PROBLEM_UNCONSTRAIN_HPP

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <type_traits>

#include "../exceptions.hpp"
#include "../io.hpp"
#include "../problem.hpp"
#include "../serialization.hpp"
#include "../type_traits.hpp"
#include "../types.hpp"

namespace pagmo
{

/// The unconstrain meta-problem
/**
 * This meta-problem transforms a constrained problem into an unconstrained problem applying one of the following
 * methods:
 *    - Death penalty: simply penalizes all objectives by the same high value if the fitness vector is infeasible as
 * checked by pagmo::problem::fesibility_f.
 *    - Kuri's death penalty: defined by Angel Kuri Morales et al., penalizes all objectives according to the rate of
 * satisfied constraints.
 *    - Weighted violations penalty: penalizes all objectives by the weighted sum of the constraint violations.
 *    - Ignore the constraints: simply ignores the constraints.
 *    - Ignore the objectives: ignores the objectives and defines as a new single objective the overall constraint
 * violation.
 *
 * See: Coello Coello, C. A. (2002). Theoretical and numerical constraint-handling techniques used with evolutionary
 * algorithms: a survey of the state of the art. Computer methods in applied mechanics and engineering, 191(11),
 * 1245-1287.
 * See: Kuri Morales, A. and Quezada, C.C. A Universal eclectic genetic algorithm for constrained optimization,
 * Proceedings 6th European Congress on Intelligent Techniques & Soft Computing, EUFIT'98, 518-522, 1998.
 */
class unconstrain : public problem
{
    // Enabler for the UDP ctor.
    template <typename T>
    using ctor_enabler
        = enable_if_t<std::is_constructible<problem, T &&>::value && !std::is_same<uncvref_t<T>, problem>::value, int>;

public:
    /// Default constructor
    /**
     * The default constructor will initialize a pagmo::null_problem unconstrained via the death penalty method.
     */
    unconstrain() : problem(null_problem{2, 3, 4}), m_method(method_type::DEATH), m_weights()
    {
    }

    /// Constructor from UDP and unconstrain method
    /**
     * **NOTE** This constructor is enabled only if \p T can be used to construct a pagmo::problem,
     * and \p T is not pagmo::problem.
     *
     * Wraps a user-defined problem so that its constraints will be removed
     *
     * @param p a user-defined problem.
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
    explicit unconstrain(T &&p, const std::string &method, const vector_double &weights)
        : problem(std::forward<T>(p)), m_weights(weights)
    {
        // The number of constraints in the original udp
        m_nec = static_cast<const problem *>(this)->get_nec();
        m_nic = static_cast<const problem *>(this)->get_nic();
        m_nc = m_nec + m_nic;
        // 1 - We throw if the original problem is unconstrained
        if (m_nc == 0u) {
            pagmo_throw(std::invalid_argument, "Unconstrain can only be applied to constrained problems");
        }
        // 2 - We throw if the method weighted is selected but the weight vector has the wrong size
        if (weights.size() != m_nc && method == "weighted") {
            pagmo_throw(std::invalid_argument, "Length of weight vector is: " + std::to_string(weights.size())
                                                   + " while the problem constraints are: " + std::to_string(m_nc));
        }
        // 3 - We throw if the method selected is not supported
        if (method != "death penalty" && method != "kuri" && method != "weighted" && method != "ignore_c"
            && method != "ignore_o") {
            pagmo_throw(std::invalid_argument, "The method " + method + " is not supported (did you mispell?)");
        }
        // 4 - We throw if a non empty weight vector is passed but the method weghted is not selected
        if (weights.size() != 0u && method != "weighted") {
            pagmo_throw(std::invalid_argument,
                        "The weight vector needs to be empty to use the unconstrain method " + method);
        }
        // 5 - We store the method in a more efficient enum type and the number of objectives of the orginal udp
        std::map<std::string, method_type> my_map
            = {{"death penalty", 0}, {"kuri", 1}, {"weighted", 2}, {"ignore_c", 3}, {"ignore_o", 4}};
        m_method = my_map[method];
        if (method != "ignore_o") {
            m_nobj = static_cast<const problem *>(this)->get_nobj();
        } else {
            m_nobj = 1u;
        }
    }

    /// Fitness
    /**
     * The unconstrained fitness computation is made
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers,
     * or by problem::fitness().
     */
    vector_double fitness(const vector_double &x) const
    {
        auto original_fitness = static_cast<const problem *>(this)->fitness(x);
        vector_double retval;
        switch (m_method) {
            case 0: // death penalty
                // copy the objectives
                retval = vector_double(original_fitness.begin(), original_fitness.begin() + m_nobj);
                // penalize them if unfeasible
                if (!static_cast<const problem *>(this)->feasibility_f(original_fitness)) {
                    std::fill(retval.begin(), retval.end(), std::numeric_limits<double>::max());
                }
                break;
            case 1: // kuri's penalty'
            {
                // copy the objectives
                retval = vector_double(original_fitness.begin(), original_fitness.begin() + m_nobj);
                // penalize them if unfeasible
                if (!static_cast<const problem *>(this)->feasibility_f(original_fitness)) {
                    // get the tolerances
                    auto c_tol = static_cast<const problem *>(this)->get_c_tol();
                    // compute the number of equality constraints satisfied
                    auto sat_ec = detail::test_eq_constraints(original_fitness.begin() + m_nobj,
                                                              original_fitness.begin() + m_nobj + m_nec, c_tol.begin())
                                      .first;
                    // compute the number of inequality constraints violated
                    auto sat_ic = detail::test_ineq_constraints(original_fitness.begin() + m_nobj + m_nec,
                                                                original_fitness.end(), c_tol.begin() + m_nec)
                                      .first;
                    // sets the Kuri penalization
                    auto penalty = std::numeric_limits<double>::max() * (1. - (double)(sat_ec + sat_ic) / (double)m_nc);
                    std::fill(retval.begin(), retval.end(), penalty);
                }
            } break;
            case 2: // weighted
            {
                // copy the objectives
                retval = vector_double(original_fitness.begin(), original_fitness.begin() + m_nobj);
                // copy the constraints (NOTE: not necessary remove)
                vector_double c(original_fitness.begin() + m_nobj, original_fitness.end());
                // get the tolerances
                auto c_tol = static_cast<const problem *>(this)->get_c_tol();
                // modify constraints to account for the tolerance and be violated if positive
                auto penalty = 0.;
                for (decltype(m_nc) i = 0u; i < m_nc; ++i) {
                    if (i < m_nec) {
                        c[i] = std::abs(c[i]) - c_tol[i];
                    } else {
                        c[i] = c[i] - c_tol[i];
                    }
                }
                for (decltype(m_nc) i = 0u; i < m_nc; ++i) {
                    if (c[i] > 0.) {
                        penalty += m_weights[i] * c[i];
                    }
                }
                // penalizing the objectives
                for (double &value : retval) {
                    value += penalty;
                }
            } break;
            case 3: // ignore_c
            {
                retval = vector_double(original_fitness.begin(), original_fitness.begin() + m_nobj);
            } break;
            case 4: // ignore_o
            {
                // get the tolerances
                auto c_tol = static_cast<const problem *>(this)->get_c_tol();
                // compute the norm of the violation on the equalities
                auto norm_ec = detail::test_eq_constraints(original_fitness.begin() + m_nobj,
                                                           original_fitness.begin() + m_nobj + m_nec, c_tol.begin())
                                   .second;
                // compute the norm of the violation on theinequalities
                auto norm_ic = detail::test_ineq_constraints(original_fitness.begin() + m_nobj + m_nec,
                                                             original_fitness.end(), c_tol.begin() + m_nec)
                                   .second;
                retval = vector_double(1, norm_ec + norm_ic);
            } break;
        }
        return retval;
    }

    /// Problem name
    /**
     * This method will add <tt>[unconstrained]</tt> to the name provided by the UDP.
     *
     * @return a string containing the problem name.
     *
     * @throws unspecified any exception thrown by problem::get_name() or memory errors in standard classes.
     */
    std::string get_name() const
    {
        return static_cast<const problem *>(this)->get_name() + " [unconstrained]";
    }

    /// Extra info
    /**
     * This method will append a description of the unconstrain method to the extra info provided
     * by the UDP.
     *
     * @return a string containing extra info on the problem.
     *
     * @throws unspecified any exception thrown by problem::get_extra_info(), the public interface of
     * \p std::ostringstream or memory errors in standard classes.
     */
    std::string get_extra_info() const
    {
        std::ostringstream oss;
        std::map<method_type, std::string> my_map = {{method_type::DEATH, "death penalty"},
                                                     {method_type::KURI, "kuri"},
                                                     {method_type::WEIGHTED, "weighted"},
                                                     {method_type::IGNORE_C, "ignore_c"},
                                                     {method_type::IGNORE_O, "ignore_o"}};
        stream(oss, "\n\tMethod: ", my_map[m_method]);
        if (m_method == method_type::WEIGHTED) {
            stream(oss, "\n\tWeight vector: ", m_weights);
        }
        return static_cast<const problem *>(this)->get_extra_info() + oss.str();
    }

    /// Object serialization
    /**
     * This method will save/load \p this into/from the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<problem>(this), m_method, m_weights, m_nobj, m_nec, m_nic, m_nc);
    }

private:
    // Delete all that we do not want to inherit from problem
    // A - Common to all meta
    vector_double::size_type get_nx() const = delete;
    vector_double::size_type get_nf() const = delete;
    vector_double::size_type get_nc() const = delete;
    unsigned long long get_fevals() const = delete;
    unsigned long long get_gevals() const = delete;
    unsigned long long get_hevals() const = delete;
    vector_double::size_type get_gs_dim() const = delete;
    std::vector<vector_double::size_type> get_hs_dim() const = delete;
    bool is_stochastic() const = delete;
    // These are methods brought in by the inheritance from pagmo::problem: they do not have any effect
    // and they are just confusing to see.
    void set_c_tol(const vector_double &) = delete;
    vector_double get_c_tol() const = delete;
    bool feasibility_x(const vector_double &) const = delete;
    bool feasibility_f(const vector_double &) const = delete;

// The CI using gcc 4.8 fails to compile this delete, excluding it in that case does not harm
// it would just result in a "weird" behaviour in case the user would try to stream this object
#if __GNUC__ > 4
    // NOTE: We delete the streaming operator overload called with unconstrain, otherwise the inner prob would stream
    // NOTE: If a streaming operator is wanted for this class remove the line below and implement it
    friend std::ostream &operator<<(std::ostream &, const unconstrain &) = delete;
#endif
    template <typename Archive>
    void save(Archive &) const = delete;
    template <typename Archive>
    void load(Archive &) = delete;

    // B - Specific to the unconstrain
    // An unconstrained problem does not have gradients (most methods are not differentiable)
    vector_double gradient(const vector_double &dv) const = delete;
    // deleting has_gradient allows the automatic detection of gradients to see that unconstrain does not have any
    // regardless of whether the class its build from has them. A unconstrain problem will thus never have gradients
    bool has_gradient() const = delete;
    // deleting has_gradient_sparsity/gradient_sparsity allows the automatic detection of gradient_sparsity to see that
    // unconstrain does have an implementation for it. The sparsity will thus always be dense and referred to a problem
    // with one objective
    bool has_gradient_sparsity() const = delete;
    sparsity_pattern gradient_sparsity() const = delete;
    // An unconstrained problem does not have hessians (most methods are not differentiable)
    std::vector<vector_double> hessians(const vector_double &dv) const = delete;
    // deleting has_hessians allows the automatic detection of hessians to see that unconstrain does not have any
    // regardless of whether the class its build from has them. An unconstrained problem will thus never have hessians
    bool has_hessians() const = delete;
    // deleting has_hessians_sparsity/hessians_sparsity allows the automatic detection of hessians_sparsity to see that
    // unconstrain does have an implementation for it. The hessians_sparsity will thus always be dense
    // (unconstrain::hessians_sparsity) and referred to a problem with one objective
    bool has_hessians_sparsity() const = delete;
    std::vector<sparsity_pattern> hessians_sparsity() const = delete;
    // No need for these the default implementation of these in pagmo::problem already returns zero.
    vector_double::size_type get_nec() const = delete;
    vector_double::size_type get_nic() const = delete;

    /// types of unconstrain methods
    enum method_type {
        DEATH = 0,    ///< "death penalty"
        KURI = 1,     ///< "kuri"
        WEIGHTED = 2, ///< "weighted"
        IGNORE_C = 3, ///< "ignore_c"
        IGNORE_O = 4  ///< "ignore_o"
    };

    /// method used to unconstrain the problem
    method_type m_method;
    /// weights vector
    vector_double m_weights;
    /// number of objectives
    vector_double::size_type m_nobj;
    /// number of constraints in the orginal udp
    vector_double::size_type m_nec;
    vector_double::size_type m_nic;
    vector_double::size_type m_nc;
};
}

PAGMO_REGISTER_PROBLEM(pagmo::unconstrain)

#endif
