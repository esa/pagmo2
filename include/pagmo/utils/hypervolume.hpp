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

#ifndef PAGMO_UTILS_HYPERVOLUME_HPP
#define PAGMO_UTILS_HYPERVOLUME_HPP

#include <memory>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>

namespace pagmo
{

/// Hypervolume
/**
 * This class encapsulate various utilities used to compute the hyervolume of a set of
 * points or the various exclusive contributions.
 *
 * The main API consists of the five methods:
 *
 * - hypervolume::compute - returns the total hypervolume of the set of points
 * - hypervolume::exclusive - returns the exclusive volume contributed by a given point
 * - hypervolume::least_contributor - returns the index of the point contributing the least volume
 * - hypervolume::greatest_contributor - returns the index of the point contributing the most volume
 * - hypervolume::contributions - returns the vector of exclusive contributions for each of the points.
 *
 * Each of the methods can be called passing them a reference point (and an index where needed)
 * and will, internally, select the most efficient exact Hypervolume algorithm able to compute
 * the requested quantity. A pagmo::hv_algorithm can also be passed as optional argument, in which case
 * it will be used to perform the computations.
 *
 */
class PAGMO_DLL_PUBLIC hypervolume
{
public:
    // Default constructor
    hypervolume();

    /// Constructor from population
    /**
     * Constructs a hypervolume object, where points are elicited from a pagmo::population object.
     *
     * @param pop a pagmo::population
     * @param verify flag stating whether the points should be verified for consistency after the construction.
     *
     * Example:
     *
     * @code
     * population pop(zdt{1u}, 20u);
     * hypervolume hv(pop);
     * hypervolume hv2(pop, false);
     * @endcode
     *
     * @throw std::invalid_argument if the population contains a problem that is constrained or single-objective
     */
    hypervolume(const pagmo::population &pop, bool verify = false);

    /// Constructor from points
    /**
     * Constructs a hypervolume object from a provided set of points.
     *
     * @param points vector of points for which the hypervolume is computed
     * @param verify flag stating whether the points should be verified after the construction.
     *        This regulates validation for further computation as well, use 'set_verify'
     *        flag to alter it later.
     *
     * Example:
     * @code
     * hypervolume hv({{2,3},{3,4}});
     * hypervolume hv2({{2,3},{3,4}}, false);
     * @endcode
     */
    hypervolume(const std::vector<vector_double> &points, bool verify = true);

    // Copy constructor.
    hypervolume(const hypervolume &);

    // Copy assignment operator.
    hypervolume &operator=(const hypervolume &);

    // Setter for 'copy_points' flag
    void set_copy_points(bool);

    // Getter for 'copy_points' flag
    bool get_copy_points() const;

    // Setter for the 'verify' flag
    void set_verify(bool);

    // Getter for the 'verify' flag
    bool get_verify() const;

    /// Calculate a default reference point
    /**
     * Calculates a mock refpoint by taking the maximum in each dimension over all points saved
     * in the hypervolume object.
     * The result is a point that is necessarily dominated by all other points, frequently used
     * for hypervolume computations.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This point is different from the one computed by :cpp:func:`pagmo::nadir()` as only the non dominated front
     *    is considered in that method (also its complexity is thus higher)
     *
     * \endverbatim
     *
     * @param offset value that can be added to each objective to assure strict domination
     *
     * @return reference point
     */
    vector_double refpoint(double offset = 0.0) const;

    // Get points
    const std::vector<vector_double> &get_points() const;

    // Choose the best algorithm to compute the hypervolume the actual implementation is given
    // in another headers as to not create a circular dependency problem
    std::shared_ptr<hv_algorithm> get_best_compute(const vector_double &) const;
    std::shared_ptr<hv_algorithm> get_best_exclusive(const unsigned, const vector_double &) const;
    std::shared_ptr<hv_algorithm> get_best_contributions(const vector_double &) const;

    // Compute hypervolume
    double compute(const vector_double &) const;

    // Compute hypervolume
    double compute(const vector_double &, hv_algorithm &) const;

    // Compute exclusive contribution
    double exclusive(unsigned, const vector_double &, hv_algorithm &) const;

    // Compute exclusive contribution
    double exclusive(unsigned, const vector_double &) const;

    // Contributions method
    std::vector<double> contributions(const vector_double &, hv_algorithm &) const;

    // Contributions method
    std::vector<double> contributions(const vector_double &) const;

    // Find the least contributing individual
    unsigned long long least_contributor(const vector_double &, hv_algorithm &) const;

    // Find the least contributing individual
    unsigned long long least_contributor(const vector_double &) const;

    // Find the most contributing individual
    unsigned long long greatest_contributor(const vector_double &, hv_algorithm &) const;

    // Find the most contributing individual
    unsigned long long greatest_contributor(const vector_double &) const;

    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, m_points, m_copy_points, m_verify);
    }

private:
    // Verify after construct method
    PAGMO_DLL_LOCAL void verify_after_construct() const;

    // Verify before compute method
    PAGMO_DLL_LOCAL void verify_before_compute(const vector_double &, hv_algorithm &) const;

private:
    mutable std::vector<vector_double> m_points;
    bool m_copy_points;
    bool m_verify;
};

namespace detail
{

// Expected number of operations
PAGMO_DLL_PUBLIC double expected_hv_operations(vector_double::size_type, vector_double::size_type);

} // end namespace detail

} // end namespace pagmo

#endif
