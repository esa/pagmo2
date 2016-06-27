#ifndef PAGMO_UTILS_GENERIC_HPP
#define PAGMO_UTILS_GENERIC_HPP

/** \file generic.hpp
 * \brief Utilities of general interest
 *
 * This header contains utilities useful in general for PaGMO purposes
 */

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "../exceptions.hpp"
#include "../problem.hpp"
#include "../rng.hpp"
#include "../types.hpp"

namespace pagmo
{

/// Generates a random number within some lower and upper bounds
/**
 * Creates a random number within a closed range. If
 * both the lower and upper bounds are finite numbers, then the generated value
 * \f$ x \f$ will be such that \f$lb \le x < ub\f$. If \f$lb == ub\f$ then \f$lb\f$ is
 * returned
 *
 * @note: This has to be preferred to std::uniform_real<double>(r_engine) as it
 * performs checks that avoid undefined behaviour in PaGMO.
 *
 * Example:
 *
 * @code
 * std::mt19937 r_engine(32u);
 * auto x = uniform_real_from_range(3,5,r_engine); // a random value
 * auto x = uniform_real_from_range(2,2,r_engine); // the value 2.
 * @endcode
 *
 * @param[in] lb lower bound
 * @param[in] ub upper bound
 * @param[in] r_engine a <tt>std::mt19937</tt> random engine
 *
 * @throws std::invalid_argument if:
 * - the bounds contain NaNs or infs,
 *   or \f$ lb > ub \f$,
 * - if \f$ub-lb\f$ is larger than implementation-defined value
 *
 * @returns a random floating-point value
 */
double uniform_real_from_range(double lb, double ub, detail::random_engine_type &r_engine)
{
    // NOTE: see here for the requirements for floating-point RNGS:
    // http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution/uniform_real_distribution

    // 0 - Forbid random generation when bounds are not finite.
    if (!std::isfinite(lb) || !std::isfinite(ub)) {
        pagmo_throw(std::invalid_argument,"Cannot generate a random point if the bounds are not finite");
    }
    // 1 - Check that lb is <= ub
    if (lb > ub) {
        pagmo_throw(std::invalid_argument,"Lower bound is greater than upper bound. Cannot generate a random point in [lb, ub]");
    }
    // 2 - Bounds cannot be too large
    const auto delta = ub - lb;
    if (!std::isfinite(delta) || delta > std::numeric_limits<double>::max()) {
        pagmo_throw(std::invalid_argument,"Cannot generate a random point within bounds that are too large");
    }
    // 3 - If the bounds are equal we don't call the RNG, as that would be undefined behaviour.
    if (lb == ub) {
        return lb;
    }
    return std::uniform_real_distribution<double>(lb, ub)(r_engine);
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

/// Safely cast between unsigned types
/**
 * Performs a cast between unsigned types throwing if the input cannot be represented in the new type
 *
 * Example:
 * @code
 * unsigned short s = std::numeric_limits<unsigned short>::max();
 * unsigned long l = std::numeric_limits<unsigned long>::max();
 * auto res1 = safe_cast<unsigned long>(s); // Will always work
 * auto res2 = safe_cast<unsigned short>(l); // Will throw an std::overflow_error if precision is lost
 * @endcode
 *
 * @param[in] x an unsigned value \p x to be casted to \p T
 * @return the input \p x safey casted to \p T
 * @throws std::overflow_error if \p x cannot be represented by the new type
 */
template <typename T, typename U>
inline T safe_cast(const U &x)
{
    static_assert(std::is_unsigned<T>::value && std::is_unsigned<U>::value,"Safe cast can only be used on unsigned types");
    if (x > std::numeric_limits<T>::max()) {
        pagmo_throw(std::overflow_error,"Converting between unsigned types caused a loss");
    }
    return static_cast<T>(x);
}

/// Binomial coefficient
/**
 * An implementation of the binomial coefficient using gamma functions
 * @param  n first parameter \f$n\f$
 * @param  k second paramater \f$k\f$
 * @return the binomial coefficient \f$ n \choose k \f$
 */
double binomial_coefficient(vector_double::size_type n, vector_double::size_type k)
{
    if (k <= n) {
        return std::round(std::exp(std::lgamma(static_cast<double>(n) + 1.) - std::lgamma(static_cast<double>(k) + 1.) - std::lgamma(static_cast<double>(n) - static_cast<double>(k) + 1.)));
    } else {
        pagmo_throw(std::invalid_argument, "The binomial coefficient is only defined for k<=n, you requested n=" + std::to_string(n) + " and k=" + std::to_string(k));
    }
}

/// K-Nearest Neighbours
/**
 * Computes the indexes of the k nearest neighbours (euclidean distance) to each of the input points.
 * The algorithm complexity (naive implementation) is \f$ O(MN^2)\f$ where \f$N\f$ is the number of
 * points and \f$M\f$ their dimensionality
 *
 * Example:
 * @code
 * auto res = kNN({{1, 1}, {2, 2}, {3.1, 3.1}, {5, 5}}, 2u);
 * @endcode
 *
 * @param[in] points the \f$N\f$ points having dimension \f$M\f$
 * @param[in] k number of neighbours to detect
 * @return An <tt>std::vector<std::vector<population::size_type> > </tt> containing the indexes of the k nearest neighbours sorted by distance
 * @throws std::invalid_argument If the points do not all have the same dimension.
 */
std::vector<std::vector<vector_double::size_type> > kNN(const std::vector<vector_double> &points, std::vector<vector_double>::size_type k) {
    std::vector<std::vector<vector_double::size_type> > neigh_idxs;
    auto N = points.size();
    if (N == 0u) {
        return {};
    }
    auto M = points[0].size();
    if (!std::all_of(points.begin(), points.end(), [M](const auto &p){return p.size() == M;} )) {
        pagmo_throw(std::invalid_argument, "All points must have the same dimensionality for k-NN to be invoked");
    }
    // loop through the points
    for(decltype(N) i = 0u; i < N; ++i) {
        // We compute all the distances to all other points including the self
        vector_double distances;
        for(decltype(N) j = 0u; j < N; ++j) {
            double dist = 0.;
            for (decltype(M) l = 0u; l < M; ++l) {
                dist += (points[i][l] - points[j][l]) * (points[i][l] - points[j][l]);
            }
            distances.push_back(std::sqrt(dist));
        }
        // We sort the indexes with respect to the distance
        std::vector<vector_double::size_type> idxs(N);
        std::iota(idxs.begin(), idxs.end(), vector_double::size_type(0u));
        std::sort(idxs.begin(), idxs.end(), [&distances] (auto idx1, auto idx2) {return distances[idx1] < distances[idx2];});
        // We remove the first element containg the self-distance (0)
        idxs.erase(std::remove(idxs.begin(), idxs.end(), i), idxs.end());
        neigh_idxs.push_back(idxs);
    }
    // We trim to k the lists if needed
    if (k < N - 1u) {
        for (decltype(neigh_idxs.size()) i = 0u; i < neigh_idxs.size();++i) {
                neigh_idxs[i].erase(neigh_idxs[i].begin() + static_cast<int>(k), neigh_idxs[i].end());
        }
    }
    return neigh_idxs;
}

namespace detail
{
    // modifies a chromosome so that it will be in the bounds. elements that are off are resampled at random in the bounds
    void force_bounds_random(vector_double &x, const vector_double &lb, const vector_double &ub, detail::random_engine_type &r_engine)
    {
        assert(x.size()==lb.size());
        assert(x.size()==ub.size());
        for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
            if ((x[j] < lb[j]) || (x[j] > ub[j])) {
                x[j] = pagmo::uniform_real_from_range(lb[j], ub[j], r_engine);
            }
        }
    }
    // modifies a chromosome so that it will be in the bounds. Elements that are off are reflected in the bounds
    void force_bounds_reflection(vector_double &x, const vector_double &lb, const vector_double &ub)
    {
        assert(x.size()==lb.size());
        assert(x.size()==ub.size());
        for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
            while(x[j] < lb[j] || x[j] > ub[j])
            {
                if (x[j] < lb[j]) {
                    x[j] = 2 * lb[j] - x[j];
                }
                if (x[j] > ub[j]) {
                    x[j] = 2 * ub[j] - x[j];
                }
            }
        }
    }
    // modifies a chromosome so that it will be in the bounds. Elements that are off are set on the bounds
    void force_bounds_stick(vector_double &x, const vector_double &lb, const vector_double &ub)
    {
        assert(x.size()==lb.size());
        assert(x.size()==ub.size());
        for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
            if (x[j] < lb[j]) {
                x[j] = lb[j];
            }
            if (x[j] > ub[j]) {
                x[j] = ub[j];
            }
        }
    }
} // namespace detail

} // namespace pagmo

#endif
