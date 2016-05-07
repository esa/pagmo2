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
 * auto x = decision_vector({{1,3},{3,5}});       // a random vector
 * auto x = decision_vector({{1,3},{3,5}}, 1234); // a random vector with seed 1234
 * auto x = decision_vector({{1,3},{1,3}});       // the vector {1,3}
 * @endcode
 *
 * @param[in] bounds an <tt>std::pair</tt> containing the bounds
 * @param[in] seed seed to the internal random engine used
 *
 * @throws std::invalid_argument if:
 * - the bounds are not of equal length, they have zero size, they contain NaNs or infs,
 *   or \f$ \mathbf{ub} \le \mathbf {lb}\f$,
 * - if \f$ub_i-lb_i\f$ is larger than implementation-defined value
 *
 * @returns a vector_double containing a random decision vector
 */
vector_double decision_vector(const std::pair<vector_double, vector_double> &bounds, unsigned int seed = pagmo::random_device::next())
{
    // This will check for consistent vector lengths, non-null sizes, lb <= ub and no NaNs.
    detail::check_problem_bounds(bounds);
    auto dim = bounds.first.size();
    vector_double retval(dim);
    detail::random_engine_type r_engine(seed);

    for (decltype(dim) i = 0u; i < dim; ++i) {
        // NOTE: see here for the requirements for floating-point RNGS:
        // http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution/uniform_real_distribution

        // 1 - Forbid random generation when bounds are infinite.
        if (std::isinf(bounds.first[i]) || std::isinf(bounds.second[i])) {
            pagmo_throw(std::invalid_argument,"Cannot generate a random individual if the problem is"
             " unbounded (inf bounds detected)");
        }
        // 2 - Bounds cannot be too large.
        const auto delta = bounds.second[i] - bounds.first[i];
        if (!std::isfinite(delta) || delta > std::numeric_limits<double>::max()) {
            pagmo_throw(std::invalid_argument,"Cannot generate a random individual if the problem bounds "
                "are too large");
        }
        // 3 - If the bounds are equal we don't call the RNG, as that would be undefined behaviour.
        if (bounds.first[i] == bounds.second[i]) {
            retval[i] = bounds.first[i];
        } else {
            retval[i] = std::uniform_real_distribution<double>(bounds.first[i], bounds.second[i])(r_engine);
        }
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
 * auto x = decision_vector({1,3},{3,5});       // a random vector
 * auto x = decision_vector({1,3},{3,5}, 1234); // a random vector with seed 1234
 * auto x = decision_vector({1,3},{1,3});       // the vector {1,3}
 * @endcode
 *
 * @param[in] lb a vector_double containing the lower bounds
 * @param[in] ub a vector_double containing the upper bounds
 * @param[in] seed seed to the internal random engine used
 *
 * @throws std::invalid_argument if:
 * - the bounds are not of equal length, they contain NaNs or infs, or \f$ \mathbf{ub} \le \mathbf {lb}\f$,
 * - if \f$ub_i-lb_i\f$ is larger than implementation-defined value
 *
 * @returns a vector_double containing a random decision vector
 */
vector_double decision_vector(const vector_double &lb, const vector_double &ub, unsigned int seed = pagmo::random_device::next())
{
    return decision_vector({lb, ub}, seed);
}

} // namespace pagmo

#endif
