#ifndef PAGMO_CONSTRAINED_HPP
#define PAGMO_CONSTRAINED_HPP

/** \file constrained.hpp
 * \brief Constrained optimization utilities.
 *
 * This header contains utilities useful for constrained optimization
 */

#include <stdexcept>
#include <string>
#include <utility>

#include "../types.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"


namespace pagmo{

namespace detail {

std::pair<vector_double::size_type, double> test_eq_constraints(vector_double::const_iterator ceq_first, vector_double::const_iterator ceq_last, vector_double::const_iterator tol_first) 
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

std::pair<vector_double::size_type, double> test_ineq_constraints(vector_double::const_iterator cineq_first, vector_double::const_iterator cineq_last, vector_double::const_iterator tol_first) 
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

/** Equality constraints test
 *
 * Tests an equality constraint vector, counting the number of constraints
 * violated \f$ n\f$ (allowing for a certain tolerance) and computing
 * the \f$L_2\f$ norm of the violation \f$ l\f$ (discounting the
 * tolerance for each constraint).
 *
 * @note Equality constraints in PaGMO are all written in the form \f$c(x) = 0\f$.
 * A given constraint is thus satisfied if \f$|c(x)| \le \epsilon \f$, where 
 * \f$\epsilon \f$ is the tolerance set for that constraint.
 *
 * @param[in] ceq A vector_double containing the constraints to be tested
 * @param[in] tol A vector_double containing the tolerance to consider for each constraint
 * @returns an <tt>std::pair</tt> containing \f$ n\f$ and \f$ l\f$
 *
 * @throws std::invalid_argument if the constraint vector and the tolerance vector
 * have different sizes
 */

std::pair<vector_double::size_type, double> test_eq_constraints(const vector_double &ceq, const vector_double &tol = {}) 
{
    auto tol_copy(tol);
    // By default the tolerance vector is set to zero
    if (tol_copy.size() == 0u) {
        tol_copy = vector_double(ceq.size(), 0.);
    }
    // Check that tolerance vector has the same size as the constraint vector
    if (tol_copy.size() != ceq.size())
    {
        pagmo_throw(std::invalid_argument, "Tolerance vector (dimension " + std::to_string(tol_copy.size()) + ") is inconsistent with constraints vector (dimension " + std::to_string(ceq.size()) + ")");
    }
    // Corner case
    if (ceq.size() == 0u) {
        return {0u, 0};
    }
    // Main computation
    return detail::test_eq_constraints(ceq.begin(), ceq.end(), tol_copy.begin());
} 

/** Equality constraints test (overload)
 *
 * Tests an equality constraint vector, counting the number of constraints
 * violated \f$ n\f$ (allowing for a certain tolerance) and computing
 * the \f$L_2\f$ norm of the violation \f$ l\f$ (discounting the
 * tolerance for each constraint).
 *
 * @note Calls pagmo::test_eq_constraints setting the tolerance vector as uniform
 *
 * @param[in] ceq A vector_double containing the constraints to be tested
 * @param[in] tol A double containing the tolerance to consider for all constraints
 * @returns an <tt>std::pair</tt> containing \f$ n\f$ and \f$ l\f$
 */
std::pair<vector_double::size_type, double> test_eq_constraints(const vector_double &ceq, double tol)
{
    vector_double tol_vector(ceq.size(), tol);
    return detail::test_eq_constraints(ceq.begin(), ceq.end(), tol_vector.begin());
}

/** Inequality constraints test
 *
 * Tests an inequality constraint vector, counting the number of constraints
 * violated \f$ n\f$ (allowing for a certain tolerance) and computing
 * the \f$L_2\f$ norm of the violation \f$ l\f$ (discounting the
 * tolerance for each constraint).
 *
 * @note Equality constraints in PaGMO are all written in the form \f$c(x) \le 0\f$.
 * A given constraint is thus satisfied if \f$c(x) \le \epsilon \f$, where 
 * \f$\epsilon \f$ is the tolerance set for that constraint.
 *
 * @param[in] ceq A vector_double containing the inequality constraints to be tested
 * @param[in] tol A vector_double containing the tolerance to consider for each inequality constraint
 * @returns an <tt>std::pair</tt> containing \f$ n\f$ and \f$ l\f$
 *
 * @throws std::invalid_argument if the inequality constraint vector and the tolerance vector
 * have different sizes
 */
std::pair<vector_double::size_type, double> test_ineq_constraints(const vector_double &ceq, const vector_double &tol = {}) 
{
    auto tol_copy(tol);
    // By default the tolerance vector is set to zero
    if (tol_copy.size() == 0u) {
        tol_copy = vector_double(ceq.size(), 0.);
    }
    // Check that tolerance vector has the same size as the constraint vector
    if (tol_copy.size() != ceq.size())
    {
        pagmo_throw(std::invalid_argument, "Tolerance vector (dimension " + std::to_string(tol_copy.size()) + ") is inconsistent with constraints vector (dimension " + std::to_string(ceq.size()) + ")");
    }
    // Corner case
    if (ceq.size() == 0u) {
        return {0u, 0};
    }
    // Main computation
    double l2=0.;
    vector_double::size_type n =0u;
    for (decltype(ceq.size()) i = 0u; i < ceq.size(); ++i) {
        auto err = std::max(ceq[i] - tol_copy[i], 0.);
        l2 += err*err;
        if (err <= 0.) {
            ++n;
        }
    }
    return detail::test_ineq_constraints(ceq.begin(), ceq.end(), tol_copy.begin());
}

/** Inequality constraints test (overload)
 *
 * Tests an inequality constraint vector, counting the number of  inequalityconstraints
 * violated \f$ n\f$ (allowing for a certain tolerance) and computing
 * the \f$L_2\f$ norm of the violation \f$ l\f$ (discounting the
 * tolerance for each inequality constraint).
 *
 * @note Calls pagmo::test_ineq_constraints setting the tolerance vector as uniform
 *
 * @param[in] ceq A vector_double containing the inequality constraints to be tested
 * @param[in] tol A double containing the tolerance to consider for all inequality constraints
 * @returns an <tt>std::pair</tt> containing \f$ n\f$ and \f$ l\f$
 */
std::pair<vector_double::size_type, double> test_ineq_constraints(const vector_double &ceq, double tol)
{
    vector_double tol_vector(ceq.size(), tol);
    return detail::test_ineq_constraints(ceq.begin(), ceq.end(), tol_vector.begin());
}

} // namespace pagmo
#endif