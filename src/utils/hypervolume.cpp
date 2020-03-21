/* Copyright 2017-2020 PaGMO development team

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

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>
#include <pagmo/utils/hv_algos/hv_hv2d.hpp>
#include <pagmo/utils/hv_algos/hv_hv3d.hpp>
#include <pagmo/utils/hv_algos/hv_hvwfg.hpp>
#include <pagmo/utils/hypervolume.hpp>

namespace pagmo
{

/// Default constructor
/**
 * Initiates hypervolume with empty set of points.
 * Used for serialization purposes.
 */
hypervolume::hypervolume() : m_points(), m_copy_points(true), m_verify(false) {}

// Constructor from population
hypervolume::hypervolume(const pagmo::population &pop, bool verify) : m_copy_points(true), m_verify(verify)
{
    if (pop.get_problem().get_nc() > 0u) {
        pagmo_throw(std::invalid_argument,
                    "The problem of the population is not unconstrained."
                    "Only unconstrained populations can be used to construct hypervolume objects.");
    }
    if (pop.get_problem().get_nobj() < 2u) {
        pagmo_throw(std::invalid_argument,
                    "The problem of the population is not multiobjective."
                    "Only multi-objective populations can be used to construct hypervolume objects.");
    }
    m_points = pop.get_f();
    if (m_verify) {
        verify_after_construct();
    }
}

// Constructor from points
hypervolume::hypervolume(const std::vector<vector_double> &points, bool verify)
    : m_points(points), m_copy_points(true), m_verify(verify)
{
    if (m_verify) {
        verify_after_construct();
    }
}

/// Default copy constructor.
/**
 * The copy constructor will deep copy the input problem \p other.
 *
 * @throws unspecified any exception thrown by:
 * - memory allocation errors in standard containers,
 */
hypervolume::hypervolume(const hypervolume &) = default;

/// Default copy assignment operator
/**
 * @return a reference to \p this.
 */
hypervolume &hypervolume::operator=(const hypervolume &) = default;

/// Setter for 'copy_points' flag
/**
 *
 * It is used in cases where we are certain that we can alter the original set of points
 * from the hypervolume object.
 * This is useful when we don't want to make a copy of the points first, as most algorithms
 * alter the original set, but may result in unexpected behaviour when used incorrectly
 * (e.g. requesting the computation twice out of the same object)
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    When this flag is set to true the object can reliably be used only once to compute
 *    the hypervolume. Successive usages are undefined behaviour.
 *
 * \endverbatim
 *
 * @param copy_points boolean value stating whether the hypervolume computation may use original set
 */
void hypervolume::set_copy_points(bool copy_points)
{
    m_copy_points = copy_points;
}

/// Getter for 'copy_points' flag
/**
 * Gets the copy_points flag
 *
 * @return the copy_points flag value
 */
bool hypervolume::get_copy_points() const
{
    return m_copy_points;
}

/// Setter for the 'verify' flag
/**
 * Turns off the verification phase.
 * By default, the hypervolume object verifies whether certain characteristics of the point set hold,
 * such as valid dimension sizes or a reference point that suits the minimisation.
 * In order to optimize the computation when the rules above are certain, we can turn off that phase.
 *
 * This may result in unexpected behaviour when used incorrectly (e.g. requesting the computation
 * of empty set of points)
 *
 * @param verify boolean value stating whether the hypervolume computation is to be executed without verification
 */
void hypervolume::set_verify(bool verify)
{
    m_verify = verify;
}

/// Getter for the 'verify' flag
/**
 * Gets the verify flag
 *
 * @return the verify flag value
 */
bool hypervolume::get_verify() const
{
    return m_verify;
}

// Calculate a default reference point
vector_double hypervolume::refpoint(double offset) const
{
    // Corner case
    if (m_points.size() == 0u) {
        return {};
    }

    auto fdim = m_points[0].size();
    vector_double ref_point(m_points[0].begin(), m_points[0].end());

    for (decltype(fdim) f_idx = 0u; f_idx < fdim; ++f_idx) {
        for (std::vector<vector_double>::size_type idx = 1u; idx < m_points.size(); ++idx) {
            ref_point[f_idx] = std::max(ref_point[f_idx], m_points[idx][f_idx]);
        }
    }

    for (auto &c : ref_point) {
        c += offset;
    }

    return ref_point;
}

/// Get points
/**
 * Will return a vector containing the points as they were set up during construction of the hypervolume object.
 *
 * @return const reference to the vector containing the fitness_vectors representing the points in the hyperspace.
 */
const std::vector<vector_double> &hypervolume::get_points() const
{
    return m_points;
}

/// Compute hypervolume
/**
 * Computes hypervolume for given reference point.
 * This method chooses the hv_algorithm dynamically.
 *
 * @param r_point vector describing the reference point
 *
 * @return value representing the hypervolume
 */
double hypervolume::compute(const vector_double &r_point) const
{
    return compute(r_point, *get_best_compute(r_point));
}

/// Compute hypervolume
/**
 * Computes hypervolume for given reference point, using given pagmo::hv_algorithm object.
 *
 * @param r_point fitness vector describing the reference point
 * @param hv_algo instance of the algorithm object used for the computation
 *
 * @return the hypervolume
 */
double hypervolume::compute(const vector_double &r_point, hv_algorithm &hv_algo) const
{
    if (m_verify) {
        verify_before_compute(r_point, hv_algo);
    }
    // copy the initial set of points, as the algorithm may alter its contents
    if (m_copy_points) {
        std::vector<vector_double> points_cpy(m_points.begin(), m_points.end());
        return hv_algo.compute(points_cpy, r_point);
    } else {
        return hv_algo.compute(m_points, r_point);
    }
}

/// Compute exclusive contribution
/**
 * Computes exclusive hypervolume for given indivdual.
 *
 * @param p_idx index of the individual for whom we compute the exclusive contribution to the hypervolume
 * @param r_point fitness vector describing the reference point
 * @param hv_algo instance of the algorithm object used for the computation
 *
 * @return the exclusive contribution to the hypervolume
 */
double hypervolume::exclusive(unsigned p_idx, const vector_double &r_point, hv_algorithm &hv_algo) const
{
    if (m_verify) {
        verify_before_compute(r_point, hv_algo);
    }

    if (p_idx >= m_points.size()) {
        pagmo_throw(std::invalid_argument, "Index of the individual is out of bounds.");
    }

    // copy the initial set of points, as the algorithm may alter its contents
    if (m_copy_points) {
        std::vector<vector_double> points_cpy(m_points.begin(), m_points.end());
        return hv_algo.exclusive(p_idx, points_cpy, r_point);
    } else {
        return hv_algo.exclusive(p_idx, m_points, r_point);
    }
}

/// Compute exclusive contribution
/**
 * Computes exclusive hypervolume for given indivdual.
 * This methods chooses the hv_algorithm dynamically.
 *
 * @param p_idx index of the individual for whom we compute the exclusive contribution to the hypervolume
 * @param r_point fitness vector describing the reference point
 *
 * @return the exclusive contribution to the hypervolume
 */
double hypervolume::exclusive(unsigned p_idx, const vector_double &r_point) const
{
    return exclusive(p_idx, r_point, *get_best_exclusive(p_idx, r_point));
}

/// Contributions method
/**
 * This method returns the exclusive contribution to the hypervolume by every point.
 * The concrete algorithm can implement this feature optimally (as opposed to calling for the exclusive contributor
 * in a loop).
 *
 * @param r_point fitness vector describing the reference point
 * @param hv_algo instance of the algorithm object used for the computation
 *
 * @return vector of exclusive contributions by every point
 */
std::vector<double> hypervolume::contributions(const vector_double &r_point, hv_algorithm &hv_algo) const
{
    if (m_verify) {
        verify_before_compute(r_point, hv_algo);
    }

    // Trivial case
    if (m_points.size() == 1u) {
        std::vector<double> c;
        c.push_back(hv_algorithm::volume_between(m_points[0], r_point));
        return c;
    }

    // copy the initial set of points, as the algorithm may alter its contents
    if (m_copy_points) {
        std::vector<vector_double> points_cpy(m_points.begin(), m_points.end());
        return hv_algo.contributions(points_cpy, r_point);
    } else {
        return hv_algo.contributions(m_points, r_point);
    }
}

/// Contributions method
/**
 * This method returns the exclusive contribution to the hypervolume by every point.
 * The concrete algorithm can implement this feature optimally (as opposed to calling for the exclusive contributor
 * in a loop).
 * The hv_algorithm itself is chosen dynamically, so the best performing method is employed for given task.
 *
 * @param r_point fitness vector describing the reference point
 * @return vector of exclusive contributions by every point
 */
std::vector<double> hypervolume::contributions(const vector_double &r_point) const
{
    return contributions(r_point, *get_best_contributions(r_point));
}

/// Find the least contributing individual
/**
 * Establishes the individual contributing the least to the total hypervolume.
 *
 * @param r_point fitness vector describing the reference point
 * @param hv_algo instance of the algorithm object used for the computation
 *
 * @return index of the least contributing point
 */
unsigned long long hypervolume::least_contributor(const vector_double &r_point, hv_algorithm &hv_algo) const
{
    if (m_verify) {
        verify_before_compute(r_point, hv_algo);
    }

    // Trivial case
    if (m_points.size() == 1) {
        return 0u;
    }

    // copy the initial set of points, as the algorithm may alter its contents
    if (m_copy_points) {
        std::vector<vector_double> points_cpy(m_points.begin(), m_points.end());
        return hv_algo.least_contributor(points_cpy, r_point);
    } else {
        return hv_algo.least_contributor(m_points, r_point);
    }
}

/// Find the least contributing individual
/**
 * Establishes the individual contributing the least to the total hypervolume.
 * This method chooses the best performing hv_algorithm dynamically
 *
 * @param r_point fitness vector describing the reference point
 *
 * @return index of the least contributing point
 */
unsigned long long hypervolume::least_contributor(const vector_double &r_point) const
{
    return least_contributor(r_point, *get_best_contributions(r_point));
}

/// Find the most contributing individual
/**
 * Establish the individual contributing the most to the total hypervolume.
 *
 * @param r_point fitness vector describing the reference point
 * @param hv_algo instance of the algorithm object used for the computation
 *
 * @return index of the most contributing point
 */
unsigned long long hypervolume::greatest_contributor(const vector_double &r_point, hv_algorithm &hv_algo) const
{
    if (m_verify) {
        verify_before_compute(r_point, hv_algo);
    }

    // copy the initial set of points, as the algorithm may alter its contents
    if (m_copy_points) {
        std::vector<vector_double> points_cpy(m_points.begin(), m_points.end());
        return hv_algo.greatest_contributor(points_cpy, r_point);
    } else {
        return hv_algo.greatest_contributor(m_points, r_point);
    }
}

/// Find the most contributing individual
/**
 * Establish the individual contributing the most to the total hypervolume.
 * This method chooses the best performing hv_algorithm dynamically
 *
 * @param r_point fitness vector describing the reference point
 *
 * @return index of the most contributing point
 */
unsigned long long hypervolume::greatest_contributor(const vector_double &r_point) const
{
    return greatest_contributor(r_point, *get_best_contributions(r_point));
}

// Verify after construct method
/**
 * Verifies whether basic requirements are met for the initial set of points.
 *
 * @throws invalid_argument if point size is empty or when the dimensions among the points differ
 */
void hypervolume::verify_after_construct() const
{
    if (m_points.size() == 0) {
        pagmo_throw(std::invalid_argument, "Point set cannot be empty.");
    }
    auto f_dim = m_points[0].size();
    if (f_dim <= 1) {
        pagmo_throw(std::invalid_argument, "Points of dimension > 1 required.");
    }
    for (const auto &v : m_points) {
        if (v.size() != f_dim) {
            pagmo_throw(std::invalid_argument, "All point set dimensions must be equal.");
        }
    }
    for (const auto &point : m_points) {
        for (auto value : point) {
            if (std::isnan(value)) {
                pagmo_throw(std::invalid_argument, "A nan value has been encountered in the hypervolume points. Cannot construct the hypervolume object");
            }
        }
    }
}

// Verify before compute method
/**
 * Verifies whether reference point and the hypervolume method meet certain criteria.
 *
 * @param r_point vector describing the reference point
 * @param hv_algo instance of the algorithm object used for the computation
 *
 * @throws value_error if reference point's and point set dimension do not agree
 */
void hypervolume::verify_before_compute(const vector_double &r_point, hv_algorithm &hv_algo) const
{
    if (m_points[0].size() != r_point.size()) {
        pagmo_throw(std::invalid_argument, "Point set dimensions and reference point dimension must be equal.");
    }
    hv_algo.verify_before_compute(m_points, r_point);
}

namespace detail
{

// Expected number of operations
/**
 * Returns the expected average amount of elementary operations necessary to compute the hypervolume
 * for a given front size \f$n\f$ and dimension size \f$d\f$
 * This method is used by the approximated algorithms that fall back to exact computation.
 *
 * @param n size of the front
 * @param d dimension size
 *
 * @return expected number of operations
 */
double expected_hv_operations(vector_double::size_type n, vector_double::size_type d)
{
    if (d <= 3u) {
        return static_cast<double>(d) * static_cast<double>(n) * std::log(n); // hv3d
    } else if (d == 4u) {
        return 4.0 * static_cast<double>(n) * static_cast<double>(n); // hv4d
    } else {
        return 0.0005 * static_cast<double>(d)
               * std::pow(static_cast<double>(n), static_cast<double>(d) * 0.5); // exponential complexity
    }
}

} // namespace detail

/// Chooses the best algorithm to compute the hypervolume
/**
 * Returns the best method for given hypervolume computation problem.
 * As of yet, only the dimension size is taken into account.
 *
 * @param r_point reference point for the vector of points
 *
 * @return an std::shared_ptr to the selected algorithm
 */
std::shared_ptr<hv_algorithm> hypervolume::get_best_compute(const vector_double &r_point) const
{
    auto fdim = r_point.size();

    if (fdim == 2u) {
        return hv2d().clone();
    } else if (fdim == 3u) {
        return hv3d().clone();
    } else {
        return hvwfg().clone();
    }
}

/// Chooses the best algorithm to compute the hypervolume
/**
 * Returns the best method for given hypervolume computation problem.
 * As of yet, only the dimension size is taken into account.
 *
 * @param p_idx index of the point for which the exclusive contribution is to be computed
 * @param r_point reference point for the vector of points
 *
 * @return an std::shared_ptr to the selected algorithm
 */
std::shared_ptr<hv_algorithm> hypervolume::get_best_exclusive(const unsigned p_idx, const vector_double &r_point) const
{
    (void)p_idx;
    // Exclusive contribution and compute method share the same "best" set of algorithms.
    return hypervolume::get_best_compute(r_point);
}

/// Chooses the best algorithm to compute the hypervolume
/**
 * Returns the best method for given hypervolume computation problem.
 * As of yet, only the dimension size is taken into account.
 *
 * @param r_point reference point for the vector of points
 *
 * @return an std::shared_ptr to the selected algorithm
 */
std::shared_ptr<hv_algorithm> hypervolume::get_best_contributions(const vector_double &r_point) const
{
    auto fdim = r_point.size();

    if (fdim == 2u) {
        return hv2d().clone();
    } else if (fdim == 3u) {
        return hv3d().clone();
    } else {
        return hvwfg().clone();
    }
}

} // namespace pagmo
