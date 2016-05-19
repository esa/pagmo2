#ifndef PAGMO_UTILS_GENERIC_HPP
#define PAGMO_UTILS_GENERIC_HPP

/** \file generic.hpp
 * \brief Utilities of general interest
 *
 * This header contains utilities useful in general for PaGMO purposes
 */

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "../exceptions.hpp"
#include "../problem.hpp"
#include "../rng.hpp"
#include "../types.hpp"

namespace pagmo{
/// Generates a random number within some lower and upper bounds
/**
 * Creates a random number within some lower and upper bounds. If
 * both the lower and upper bounds are finite numbers, then the \f$i\f$-th
 * component of the randomly generated decision_vector will be such that
 * \f$lb_i \le x_i < ub_i\f$. If \f$lb_i == ub_i\f$ then \f$lb_i\f$ is
 * returned
 *
 * @note: This has to be preferred to std::uniform_real<double>(r_engine) as it
 * performs checks that avoid undefined behaviour in PaGMO.
 *
 * Example:
 *
 * @code
 * std::mt19937 r_engine(32u);
 * auto x = decision_vector({{1,3},{3,5}}, r_engine); // a random vector
 * auto x = decision_vector({{1,3},{1,3}}, r_engine); // the vector {1,3}
 * @endcode
 *
 * @param[in] lb lower bound
 * @param[in] ub upper bound
 * @param[in] r_engine a <tt>std::mt19937</tt> random engine
 *
 * @throws std::invalid_argument if:
 * - the bounds are not of equal length, they have zero size, they contain NaNs or infs,
 *   or \f$ \mathbf{ub} \le \mathbf {lb}\f$,
 * - if \f$ub_i-lb_i\f$ is larger than implementation-defined value
 *
 * @returns a vector_double containing a random decision vector
 */
double uniform_real_from_range(double lb, double ub, detail::random_engine_type &r_engine)
{
    // NOTE: see here for the requirements for floating-point RNGS:
    // http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution/uniform_real_distribution

    // 0 - Check that lb is <= ub
    if (lb > ub) {
        pagmo_throw(std::invalid_argument,"Lower bounds are greater than upper bounds. Cannot generate a random pint in [lb, ub]");
    }

    // 1 - Forbid random generation when bounds are infinite.
    if (std::isinf(lb) || std::isinf(ub)) {
        pagmo_throw(std::invalid_argument,"Cannot generate a random point if (inf bounds detected)");
    }
    // 2 - Bounds cannot be too large.
    const auto delta = ub - lb;
    if (!std::isfinite(delta) || delta > std::numeric_limits<double>::max()) {
        pagmo_throw(std::invalid_argument,"Cannot generate a random point within bounds "
            "that are too large");
    }
    double retval;
    // 3 - If the bounds are equal we don't call the RNG, as that would be undefined behaviour.
    if (lb == ub) {
        retval = lb;
    } else {
        retval = std::uniform_real_distribution<double>(lb, ub)(r_engine);
    }
    return retval;
}

/// Generates a random decision vector
/**
 * Creates a random decision vector within some bounds. If
 * both the lower and upper bounds are finite numbers, then the \f$i\f$-th
 * component of the randomly generated decision_vector will be such that
 * \f$lb_i \le x_i < ub_i\f$. If \f$lb_i == ub_i\f$ then \f$lb_i\f$ is
 * returned
 *
 * Example:
 *
 * @code
 * std::mt19937 r_engine(32u);
 * auto x = decision_vector({{1,3},{3,5}}, r_engine); // a random vector
 * auto x = decision_vector({{1,3},{1,3}}, r_engine); // the vector {1,3}
 * @endcode
 *
 * @param[in] bounds an <tt>std::pair</tt> containing the bounds
 * @param[in] r_engine a <tt>std::mt19937</tt> random engine
 *
 * @throws std::invalid_argument if:
 * - the bounds are not of equal length, they have zero size, they contain NaNs or infs,
 *   or \f$ \mathbf{ub} \le \mathbf {lb}\f$,
 * - if \f$ub_i-lb_i\f$ is larger than implementation-defined value
 *
 * @returns a vector_double containing a random decision vector
 */
vector_double decision_vector(const std::pair<vector_double, vector_double> &bounds, detail::random_engine_type &r_engine)
{
    // This will check for consistent vector lengths, non-null sizes, lb <= ub and no NaNs.
    detail::check_problem_bounds(bounds);
    auto dim = bounds.first.size();
    vector_double retval(dim);

    for (decltype(dim) i = 0u; i < dim; ++i) {
        retval[i] = uniform_real_from_range(bounds.first[i], bounds.second[i], r_engine);
    }
    return retval;
}

/// Generates a random decision vector
/**
 * Creates a random decision vector within some bounds. If
 * both the lower and upper bounds are finite numbers, then the \f$i\f$-th
 * component of the randomly generated decision_vector will be such that
 * \f$lb_i \le x_i < ub_i\f$. If \f$lb_i == ub_i\f$ then \f$lb_i\f$ is
 * returned
 *
 * Example:
 *
 * @code
 * std::mt19937 r_engine(32u);
 * auto x = decision_vector({1,3},{3,5}, r_engine); // a random vector
 * auto x = decision_vector({1,3},{1,3}, r_engine); // the vector {1,3}
 * @endcode
 *
 * @param[in] lb a vector_double containing the lower bounds
 * @param[in] ub a vector_double containing the upper bounds
 * @param[in] r_engine a <tt>std::mt19937</tt> random engine
 *
 * @throws std::invalid_argument if:
 * - the bounds are not of equal length, they contain NaNs or infs, or \f$ \mathbf{ub} \le \mathbf {lb}\f$,
 * - if \f$ub_i-lb_i\f$ is larger than implementation-defined value
 *
 * @returns a vector_double containing a random decision vector
 */
vector_double decision_vector(const vector_double &lb, const vector_double &ub, detail::random_engine_type &r_engine)
{
    return decision_vector({lb, ub}, r_engine);
}

} // namespace pagmo

#endif
