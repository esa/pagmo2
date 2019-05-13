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
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#pragma GCC diagnostic ignored "-Wsuggest-attribute=const"
#endif

namespace pagmo
{

/// Default constructor
/**
 * The default constructor will initialize a pagmo::null_problem unconstrained via the death penalty method.
 */
unconstrain::unconstrain() : unconstrain(null_problem{2, 3, 4}, "death penalty") {}

void unconstrain::generic_ctor_impl(const std::string &method, const vector_double &weights)
{
    // The number of constraints in the original udp
    auto nec = m_problem.get_nec();
    auto nic = m_problem.get_nic();
    auto nc = nec + nic;
    // 1 - We throw if the original problem is unconstrained
    if (nc == 0u) {
        pagmo_throw(std::invalid_argument, "Unconstrain can only be applied to constrained problems, the instance of "
                                               + m_problem.get_name() + " is not one.");
    }
    // 2 - We throw if the method weighted is selected but the weight vector has the wrong size
    if (weights.size() != nc && method == "weighted") {
        pagmo_throw(std::invalid_argument, "Length of weight vector is: " + std::to_string(weights.size())
                                               + " while the problem constraints are: " + std::to_string(nc));
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
    std::map<std::string, method_type> my_map = {{"death penalty", method_type::DEATH},
                                                 {"kuri", method_type::KURI},
                                                 {"weighted", method_type::WEIGHTED},
                                                 {"ignore_c", method_type::IGNORE_C},
                                                 {"ignore_o", method_type::IGNORE_O}};
    m_method = my_map[method];
}

/// Fitness.
/**
 * The unconstrained fitness computation.
 *
 * @param x the decision vector.
 *
 * @return the fitness of \p x.
 *
 * @throws unspecified any exception thrown by memory errors in standard containers,
 * or by problem::fitness().
 */
vector_double unconstrain::fitness(const vector_double &x) const
{
    // some quantities from the orginal udp
    auto original_fitness = m_problem.fitness(x);
    auto nobj = m_problem.get_nobj();
    auto nec = m_problem.get_nec();
    auto nic = m_problem.get_nic();
    auto nc = nec + nic;

    // the different methods
    vector_double retval;
    switch (m_method) {
        case method_type::DEATH: {
            // copy the objectives
            retval = vector_double(original_fitness.data(), original_fitness.data() + nobj);
            // penalize them if unfeasible
            if (!m_problem.feasibility_f(original_fitness)) {
                std::fill(retval.begin(), retval.end(), std::numeric_limits<double>::max());
            }
        } break;
        case method_type::KURI: {
            // copy the objectives
            retval = vector_double(original_fitness.data(), original_fitness.data() + nobj);
            // penalize them if unfeasible
            if (!m_problem.feasibility_f(original_fitness)) {
                // get the tolerances
                auto c_tol = m_problem.get_c_tol();
                // compute the number of equality constraints satisfied
                auto sat_ec = detail::test_eq_constraints(original_fitness.data() + nobj,
                                                          original_fitness.data() + nobj + nec, c_tol.data())
                                  .first;
                // compute the number of inequality constraints violated
                auto sat_ic = detail::test_ineq_constraints(original_fitness.data() + nobj + nec,
                                                            original_fitness.data() + original_fitness.size(),
                                                            c_tol.data() + nec)
                                  .first;
                // sets the Kuri penalization
                auto penalty = std::numeric_limits<double>::max()
                               * (1. - static_cast<double>(sat_ec + sat_ic) / static_cast<double>(nc));
                std::fill(retval.begin(), retval.end(), penalty);
            }
        } break;
        case method_type::WEIGHTED: {
            // copy the objectives
            retval = vector_double(original_fitness.data(), original_fitness.data() + nobj);
            // copy the constraints (NOTE: not necessary remove)
            vector_double c(original_fitness.data() + nobj, original_fitness.data() + original_fitness.size());
            // get the tolerances
            auto c_tol = m_problem.get_c_tol();
            // modify constraints to account for the tolerance and be violated if positive
            auto penalty = 0.;
            for (decltype(nc) i = 0u; i < nc; ++i) {
                if (i < nec) {
                    c[i] = std::abs(c[i]) - c_tol[i];
                } else {
                    c[i] = c[i] - c_tol[i];
                }
            }
            for (decltype(nc) i = 0u; i < nc; ++i) {
                if (c[i] > 0.) {
                    penalty += m_weights[i] * c[i];
                }
            }
            // penalizing the objectives
            for (double &value : retval) {
                value += penalty;
            }
        } break;
        case method_type::IGNORE_C: {
            retval = vector_double(original_fitness.data(), original_fitness.data() + nobj);
        } break;
        case method_type::IGNORE_O: {
            // get the tolerances
            auto c_tol = m_problem.get_c_tol();
            // and the number of objectives in the original problem
            auto n_obj_orig = m_problem.get_nobj();
            // compute the norm of the violation on the equalities
            auto norm_ec = detail::test_eq_constraints(original_fitness.data() + n_obj_orig,
                                                       original_fitness.data() + n_obj_orig + nec, c_tol.data())
                               .second;
            // compute the norm of the violation on theinequalities
            auto norm_ic
                = detail::test_ineq_constraints(original_fitness.data() + n_obj_orig + nec,
                                                original_fitness.data() + original_fitness.size(), c_tol.data() + nec)
                      .second;
            retval = vector_double(1, norm_ec + norm_ic);
        } break;
    }
    return retval;
}

/// Number of objectives.
/**
 * @return the number of objectives of the inner problem.
 */
vector_double::size_type unconstrain::get_nobj() const
{
    if (m_method != method_type::IGNORE_O) {
        return m_problem.get_nobj();
    } else {
        return 1u;
    }
}

/// Integer dimension
/**
 * @return the integer dimension of the inner problem.
 */
vector_double::size_type unconstrain::get_nix() const
{
    return m_problem.get_nix();
}

/// Box-bounds.
/**
 * Forwards the bounds computations to the inner pagmo::problem.
 *
 * @return the lower and upper bounds for each of the decision vector components.
 *
 * @throws unspecified any exception thrown by <tt>problem::get_bounds()</tt>.
 */
std::pair<vector_double, vector_double> unconstrain::get_bounds() const
{
    return m_problem.get_bounds();
}

/// Calls <tt>has_set_seed()</tt> of the inner problem.
/**
 * Calls the method <tt>has_set_seed()</tt> of the inner problem.
 *
 * @return a flag signalling wether the inner problem is stochastic.
 */
bool unconstrain::has_set_seed() const
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
void unconstrain::set_seed(unsigned seed)
{
    return m_problem.set_seed(seed);
}

/// Problem's thread safety level.
/**
 * The thread safety of a meta-problem is defined by the thread safety of the inner pagmo::problem.
 *
 * @return the thread safety level of the inner pagmo::problem.
 */
thread_safety unconstrain::get_thread_safety() const
{
    return m_problem.get_thread_safety();
}

/// Getter for the inner problem.
/**
 * Returns a const reference to the inner pagmo::problem.
 *
 * @return a const reference to the inner pagmo::problem.
 */
const problem &unconstrain::get_inner_problem() const
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
 *    :cpp:class:`pagmo::problem` via this reference is undefined behaviour.
 *
 * \endverbatim
 *
 * @return a reference to the inner pagmo::problem.
 */
problem &unconstrain::get_inner_problem()
{
    return m_problem;
}

/// Problem name.
/**
 * This method will add <tt>[unconstrained]</tt> to the name provided by the inner problem.
 *
 * @return a string containing the problem name.
 *
 * @throws unspecified any exception thrown by problem::get_name() or memory errors in standard classes.
 */
std::string unconstrain::get_name() const
{
    return m_problem.get_name() + " [unconstrained]";
}

/// Extra info.
/**
 * This method will append a description of the unconstrain method to the extra info provided
 * by the inner problem.
 *
 * @return a string containing extra info on the problem.
 *
 * @throws unspecified any exception thrown by problem::get_extra_info(), the public interface of
 * \p std::ostringstream or memory errors in standard classes.
 */
std::string unconstrain::get_extra_info() const
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
    return m_problem.get_extra_info() + oss.str();
}

/// Object serialization.
/**
 * This method will save/load \p this into/from the archive \p ar.
 *
 * @param ar target archive.
 *
 * @throws unspecified any exception thrown by the serialization of the inner problem and of primitive types.
 */
template <typename Archive>
void unconstrain::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_problem, m_method, m_weights);
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::unconstrain)
