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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>

namespace pagmo
{

/** Compares two fitness vectors in a single-objective, constrained, case (from a vector of tolerances)
 *
 * Comparison between two fitness vectors (assuming a single-objective optimization)
 * with respect to the following strict ordering:
 * - \f$f_1 \prec f_2\f$ if \f$f_1\f$ is feasible and \f$f_2\f$ is not.
 * - \f$f_1 \prec f_2\f$ if \f$f_1\f$ is they are both infeasible, but \f$f_1\f$
 * violates fewer constraints than \f$f_2\f$, or in case they both violate the same
 * number of constraints, if the \f$L_2\f$ norm of the overall constraint violation
 is smaller.
 * - \f$f_1 \prec f_2\f$ if both fitness vectors are feasible and the objective value
 * in \f$f_1\f$ is smaller than the objectve value in \f$f_2\f$
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The fitness vectors are assumed to contain exactly one objective, ``neq`` equality
 *    constraints and the rest (if any) inequality constraints
 *
 * \endverbatim
 *
 * @param f1 first fitness vector
 * @param f2 second fitness vector
 * @param neq number of equality constraints
 * @param tol a vector_double containing the tolerances to be accounted for in the constraints
 *
 * @return true if \p f1 is "better" than \p f2
 *
 * @throws std::invalid_argument If \p f1 and \p f2 do not have equal size \f$n\f$
 * @throws std::invalid_argument If \p f1 does not have at least size 1
 * @throws std::invalid_argument If \p neq is larger than \f$n - 1\f$ (too many constraints)
 * @throws std::invalid_argument If the size of the \p tol is not exactly the size of \p f1 - 1
 */
bool compare_fc(const vector_double &f1, const vector_double &f2, vector_double::size_type neq,
                const vector_double &tol)
{
    // 1 - The two fitness must have the same dimension
    if (f1.size() != f2.size()) {
        pagmo_throw(std::invalid_argument, "Fitness dimensions should be equal: " + std::to_string(f1.size())
                                               + " != " + std::to_string(f2.size()));
    }
    // 2 - The dimension of the fitness vectors must be at least 1
    if (f1.size() < 1u) {
        pagmo_throw(std::invalid_argument, "Fitness dimension should be at least 1 to compare: a dimension of "
                                               + std::to_string(f1.size()) + " was detected. ");
    }
    // 3 - The dimension of the tolerance vector must be that of the fitness minus one
    if (f1.size() - 1u != tol.size()) {
        pagmo_throw(std::invalid_argument,
                    "Tolerance vector dimension is detected to be: " + std::to_string(tol.size())
                        + ", while the fitness dimension is: " + std::to_string(f1.size())
                        + ", I was expecting the tolerance vector dimension to be: " + std::to_string(f1.size() - 1u));
    }
    // 4 - The number of equality constraints must be at most f1.size()-1
    if (neq > f1.size() - 1u) {
        pagmo_throw(std::invalid_argument,
                    "Number of equality constraints declared: " + std::to_string(neq)
                        + " while fitness vector has dimension: " + std::to_string(f1.size())
                        + "(it must be striclty smaller as the objfun is assumed to be at position 0)");
    }

    auto c1eq = detail::test_eq_constraints(f1.data() + 1, f1.data() + 1 + neq, tol.data());
    auto c1ineq = detail::test_ineq_constraints(f1.data() + 1 + neq, f1.data() + f1.size(), tol.data() + neq);
    auto n1 = c1eq.first + c1ineq.first;
    auto l1 = c1eq.second + c1ineq.second;

    auto c2eq = detail::test_eq_constraints(f2.data() + 1, f2.data() + 1 + neq, tol.data());
    auto c2ineq = detail::test_ineq_constraints(f2.data() + 1 + neq, f2.data() + f2.size(), tol.data() + neq);
    auto n2 = c2eq.first + c2ineq.first;
    auto l2 = std::sqrt(c2eq.second * c2eq.second + c2ineq.second * c2ineq.second);
    if (n1 == n2) {                 // same number of constraints satistfied
        if (n1 == f1.size() - 1u) { // fitness decides
            return detail::less_than_f(f1[0], f2[0]);
        } else { // l2 norm decides
            return detail::less_than_f(l1, l2);
        }
    } else { // number of constraints satisfied decides
        return n1 > n2;
    }
}

/** Compares two fitness vectors in a single-objective, constrained, case (from a scalar tolerance)
 *
 * @param f1 first fitness vector
 * @param f2 second fitness vector
 * @param neq number of equality constraints
 * @param tol a vector_double containing the tolerances to be accounted for in the constraints
 *
 * @return true if \p f1 is "better" than \p f2
 *
 * @throws std::invalid_argument If \p f1 and \p f2 do not have equal size \f$n\f$
 * @throws std::invalid_argument If \p f1 does not have at least size 1
 * @throws std::invalid_argument If \p neq is larger than \f$n - 1\f$ (too many constraints)
 */
bool compare_fc(const vector_double &f1, const vector_double &f2, vector_double::size_type neq, double tol)
{
    // 1 - The dimension of the fitness vector must be at least 1 (this check
    // cannot be removed and delegated to the other overload as f1.size()-1u is used)
    if (f1.size() < 1u) {
        pagmo_throw(std::invalid_argument, "Fitness dimension should be at least 1 to compare: a dimension of "
                                               + std::to_string(f1.size()) + " was detected. ");
    }
    return compare_fc(f1, f2, neq, vector_double(f1.size() - 1u, tol));
}

/** Sorts a population in a single-objective, constrained, case (from a vector of tolerances)
 *
 * Sorts a population (intended here as an <tt>std::vector<vector_double></tt>
 * containing single objective fitness vectors)
 * with respect to the following strict ordering:
 * - \f$f_1 \prec f_2\f$ if \f$f_1\f$ is feasible and \f$f_2\f$ is not.
 * - \f$f_1 \prec f_2\f$ if \f$f_1\f$ is they are both infeasible, but \f$f_1\f$
 * violates fewer constraints than \f$f_2\f$, or in case they both violate the same
 * number of constraints, if the \f$L_2\f$ norm of the overall constraint violation
 * is smaller.
 * - \f$f_1 \prec f_2\f$ if both fitness vectors are feasible and the objective value
 * in \f$f_1\f$ is smaller than the objectve value in \f$f_2\f$
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The fitness vectors are assumed to contain exactly one objective, ``neq`` equality
 *    constraints and the rest (if any) inequality constraints
 *
 * \endverbatim
 *
 * @param input_f an <tt>std::vector</tt> of fitness vectors (containing objectives and constraints)
 * @param neq number of equality constraints
 * @param tol a vector_double containing tolerances to be accouted for in the constraints
 *
 * @return an <tt>std::vector</tt> of indexes containing the sorted population
 *
 * @throws std::invalid_argument If the input fitness vectors do not have all the same size \f$n >=1\f$
 * @throws std::invalid_argument If \p neq is larger than \f$n - 1\f$ (too many constraints)
 * @throws std::invalid_argument If the size of the \p tol is not \f$n - 1\f$
 *
 */
std::vector<pop_size_t> sort_population_con(const std::vector<vector_double> &input_f, vector_double::size_type neq,
                                            const vector_double &tol)
{
    auto N = input_f.size();
    // Corner cases
    if (N < 2u) {
        if (N == 0u) {
            return {};
        }
        if (N == 1u) {
            return {0u};
        }
    }

    // Create the indexes 0....N-1
    std::vector<pop_size_t> retval(N);
    std::iota(retval.begin(), retval.end(), pop_size_t(0));
    // Sort the indexes
    std::sort(retval.begin(), retval.end(), [&input_f, &neq, &tol](pop_size_t idx1, pop_size_t idx2) {
        return compare_fc(input_f[idx1], input_f[idx2], neq, tol);
    });
    return retval;
}

/// Sorts a population in a single-objective, constrained, case (from a scalar tolerance)
/**
 *
 * @param input_f an <tt>std::vector</tt> of fitness vectors (containing objectives and constraints)
 * @param neq number of equality constraints
 * @param tol scalar tolerance to be accouted for in the constraints
 *
 * @return an <tt>std::vector</tt> of indexes containing the sorted population
 *
 * @throws std::invalid_argument If the input fitness vectors do not have all the same size \f$n >=1\f$
 * @throws std::invalid_argument If \p neq is larger than \f$n - 1\f$ (too many constraints)
 */
std::vector<pop_size_t> sort_population_con(const std::vector<vector_double> &input_f, vector_double::size_type neq,
                                            double tol)
{
    auto N = input_f.size();
    // Corner cases
    if (N < 2u) {
        if (N == 0u) {
            return {};
        }
        if (N == 1u) {
            return {0u};
        }
    }
    // Now we are sure input_f is not empty and has size at least 2
    auto M = input_f[0].size();
    // 2 - The dimension of the fitness vectors must be at least 1
    if (M < 1u) {
        pagmo_throw(std::invalid_argument, "Fitness dimension should be at least 1 to sort: a dimension of "
                                               + std::to_string(M) + " was detected. ");
    }
    vector_double tol_vector(M - 1u, tol);
    return sort_population_con(input_f, neq, tol_vector);
}

} // namespace pagmo
