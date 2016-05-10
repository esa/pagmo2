#ifndef PAGMO_CONSTRAINED_HPP
#define PAGMO_CONSTRAINED_HPP

/** \file constrained.hpp
 * \brief Constrained optimization utilities.
 *
 * This header contains utilities useful for constrained optimization
 */

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "../exceptions.hpp"
#include "../io.hpp"
#include "../types.hpp"

namespace pagmo{

namespace detail {

/// Tests equality constraints against some tolerance vector. Returns number of constraints satisfied and the L2 norm of the violation
template <typename It>
std::pair<vector_double::size_type, double> test_eq_constraints(It ceq_first, It ceq_last, It tol_first)
{
    // Main computation
    double l2=0.;
    vector_double::size_type n =0u;
    while(ceq_first!=ceq_last) {
        auto err = std::max(std::abs(*ceq_first++) - *tol_first++, 0.);
        l2 += err*err;
        if (err <= 0.) {
            ++n;
        }
    }
    return std::pair<vector_double::size_type, double>(n, std::sqrt(l2));
}

/// Tests inequality constraints against some tolerance vector. Returns number of constraints satisfied and the L2 norm of the violation
template <typename It>
std::pair<vector_double::size_type, double> test_ineq_constraints(It cineq_first, It cineq_last, It tol_first)
{
    // Main computation
    double l2=0.;
    vector_double::size_type n =0u;
    while(cineq_first!=cineq_last) {
        auto err = std::max(*cineq_first++ - *tol_first++, 0.);
        l2 += err*err;
        if (err <= 0.) {
            ++n;
        }
    }
    return std::pair<vector_double::size_type, double>(n, std::sqrt(l2));
}

} // detail namespace

/** Sorts a population in a constrained optimization case
 *
 * Sorts a population (intended here as an <tt>std::vector<vector_double></tt>
 * containing single objective fitness vectors)
 * with respect to the following strict ordering:
 * - \f$f_1 \prec f_2\f$ if \f$f_1\f$ is feasible and \f$f_2\f$ is not.
 * - \f$f_1 \prec f_2\f$ if \f$f_1\f$ is they are both infeasible, but \f$f_1\f$
 * violates less constraints than \f$f_2\f$, or in case they both violate the same
 * number of constraints, if the \f$L_2\f$ norm of the overall constraint violation
 is smaller.
 * - \f$f_1 \prec f_2\f$ if both fitness vectors are feasible and the objective value
 * in \f$f_1\f$ is smaller than the objectve value in \f$f_2\f$
 *
 * @note: the fitness vectors are assumed to contain exactly one objective, \p neq equality
 * constraints and the rest (if any) inequality constraints
 *
 * @param[in] input_f an <tt>std::vector</tt> of fitness vectors (containing objectives and constraints)
 * @param[in] neq number of equality constraints
 * @param[in] tol a vector_double containing tolerances to be accouted for in the constraints
 *
 * @return an <tt>std::vector</tt> of indexes containing the sorted population
 *
 * @throws std::invalid_argument If the input fitness vectors do not have all the same size \f$n\f$
 * @throws std::invalid_argument If \p neq is larger than \f$n - 1\f$ (too many constraints)
 * @throws std::invalid_argument If the size of the \p tol is not exactly the size of \p input_f - 1
 *
 */
std::vector<vector_double::size_type> sort_population_con(const std::vector<vector_double> &input_f, vector_double::size_type neq, const vector_double &tol)
{
    auto N = input_f.size();
    /// Corner cases
    if (N < 2u) { // corner cases
        if (N == 0u) {
            return {};
        }
        if (N == 1u) {
            return {0u};
        }
    }
    // Now we are sure input_f is not empty and has size at least 2
    auto M = input_f[0].size();
    // Santity Checks
    // 1 - All fitness vectors must have the same size
    for (decltype(N) i = 1u; i < N; ++i) {
        if (input_f[i].size() != M) {
            pagmo_throw(std::invalid_argument, "The fitness vector at position: "
                + std::to_string(i) + " has dimension "
                + std::to_string(input_f[i].size()) + " while I was expecting: "
                + std::to_string(M) + "(first element dimension)"
            );
        }
    }
    // 2 - The dimension of the fitness vectors mus be at least 1 
    if (M < 1u) {
        pagmo_throw(std::invalid_argument, "Fitness dimension should be at least 1 to sort: a dimension of "
            + std::to_string(M) + " was detected. "
        );
    }
    // Now we are sure M has size at least 2
    // 3 - The number of equality constraints must be at most input_f[0].size()-1
    if (neq > M - 1u) {
        pagmo_throw(std::invalid_argument, "Number of equality constraints declared: "
            + std::to_string(neq) + " while fitness vector has dimension: "
            + std::to_string(M) + "(it must be striclty smaller as the objfun is assumed to be at position 0)"
        );
    }
    // 4 - The tolerance vector size must be input_f.size[0]()-1u
    if (tol.size() != M-1u) {
        pagmo_throw(std::invalid_argument, "Tolerance vector dimension: "
            + std::to_string(tol.size()) + " while it must be: "
            + std::to_string(M-1u)
        );
    }

    // Create the indexes 0....N-1
    std::vector<vector_double::size_type> retval(N);
    std::iota(retval.begin(), retval.end(), vector_double::size_type(0u));
    // Sort the indexes
    std::sort(retval.begin(), retval.end(), [&input_f, &neq, &tol] (auto idx1, auto idx2)
    {
        auto c1eq = detail::test_eq_constraints(input_f[idx1].data()+1, input_f[idx1].data()+1+neq, tol.data());
        auto c1ineq = detail::test_ineq_constraints(input_f[idx1].data()+1+neq, input_f[idx1].data()+input_f[idx1].size(), tol.data() + neq);
        auto n1 = c1eq.first+c1ineq.first;
        auto l1 = c1eq.second+c1ineq.second;

        auto c2eq = detail::test_eq_constraints(input_f[idx2].data()+1, input_f[idx2].data()+1+neq, tol.data());
        auto c2ineq = detail::test_ineq_constraints(input_f[idx2].data()+1+neq, input_f[idx2].data()+input_f[idx2].size(), tol.data() + neq);
        auto n2 = c2eq.first + c2ineq.first;
        auto l2 = std::sqrt(c2eq.second*c2eq.second + c2ineq.second*c2ineq.second);
        if (n1 == n2) { // same number of constraints satistfied
            if (n1 == input_f[0].size() - 1u) { // fitness decides
                return input_f[idx1][0] < input_f[idx2][0];
            } else { // l2 norm decides
                return l1 < l2;
            }
        } else { // number of constraints satisfied decides
            return n1 > n2;
        }
    });
    return retval;
}

/// Sorts a population in a constrained optimization case (from one tolerance valid for all)
std::vector<vector_double::size_type> sort_population_con(const std::vector<vector_double> &input_f, vector_double::size_type neq, double tol = 0.)
{
    auto N = input_f.size();
    /// Corner cases
    if (N < 2u) { // corner cases
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
            + std::to_string(M) + " was detected. "
        );
    }
    vector_double tol_vector(M - 1u, tol);
    return sort_population_con(input_f, neq, tol_vector);
}

} // namespace pagmo
#endif
