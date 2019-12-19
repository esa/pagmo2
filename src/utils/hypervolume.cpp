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

#include <stdexcept>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>
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
 * @param other the hypervolume object to be copied.
 *
 * @throws unspecified any exception thrown by:
 * - memory allocation errors in standard containers,
 */
hypervolume::hypervolume(const hypervolume &other) = default;

/// Default copy assignment operator
/**
 * @param other the assignment target.
 *
 * @return a reference to \p this.
 */
hypervolume &hypervolume::operator=(const hypervolume &other) = default;

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

} // namespace pagmo
