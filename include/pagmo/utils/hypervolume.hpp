/*****************************************************************************
*   Copyright (C) 2004-2015 The PaGMO development team,                     *
*   Advanced Concepts Team (ACT), European Space Agency (ESA)               *
*                                                                           *
*   https://github.com/esa/pagmo                                            *
*                                                                           *
*   act@esa.int                                                             *
*                                                                           *
*   This program is free software; you can redistribute it and/or modify    *
*   it under the terms of the GNU General Public License as published by    *
*   the Free Software Foundation; either version 2 of the License, or       *
*   (at your option) any later version.                                     *
*                                                                           *
*   This program is distributed in the hope that it will be useful,         *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
*   GNU General Public License for more details.                            *
*                                                                           *
*   You should have received a copy of the GNU General Public License       *
*   along with this program; if not, write to the                           *
*   Free Software Foundation, Inc.,                                         *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.               *
*****************************************************************************/

#ifndef PAGMO_UTIL_HYPERVOLUME_H
#define PAGMO_UTIL_HYPERVOLUME_H

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
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
class hypervolume
{
public:
    /// Default constructor
    /**
    * Initiates hypervolume with empty set of points.
    * Used for serialization purposes.
    */
    hypervolume() : m_points(), m_copy_points(true), m_verify(false)
    {
    }

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
    hypervolume(const pagmo::population &pop, bool verify = false) : m_copy_points(true), m_verify(verify)
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
    hypervolume(const std::vector<vector_double> &points, bool verify = true)
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
    hypervolume(const hypervolume &other) = default;

    /// Default copy assignment operator
    /**
     * @param other the assignment target.
     *
     * @return a reference to \p this.
     */
    hypervolume &operator=(const hypervolume &other) = default;

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
    void set_copy_points(bool copy_points)
    {
        m_copy_points = copy_points;
    }

    /// Getter for 'copy_points' flag
    /**
    * Gets the copy_points flag
    *
    * @return the copy_points flag value
     */
    bool get_copy_points() const
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
    void set_verify(bool verify)
    {
        m_verify = verify;
    }

    /// Getter for the 'verify' flag
    /**
     * Gets the verify flag
     *
     * @return the verify flag value
     */
    bool get_verify() const
    {
        return m_verify;
    }

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
    vector_double refpoint(double offset = 0.0) const
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
    const std::vector<vector_double> &get_points() const
    {
        return m_points;
    }

    // Choose the best algorithm to compute the hypervolume the actual implementation is given
    // in another headers as to not create a circular dependency problem
    std::shared_ptr<hv_algorithm> get_best_compute(const vector_double &r_point) const;
    std::shared_ptr<hv_algorithm> get_best_exclusive(const unsigned int p_idx, const vector_double &r_point) const;
    std::shared_ptr<hv_algorithm> get_best_contributions(const vector_double &r_point) const;

    /// Compute hypervolume
    /**
    * Computes hypervolume for given reference point.
    * This method chooses the hv_algorithm dynamically.
    *
    * @param r_point vector describing the reference point
    *
    * @return value representing the hypervolume
    */
    double compute(const vector_double &r_point) const
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
    double compute(const vector_double &r_point, hv_algorithm &hv_algo) const
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
    double exclusive(unsigned int p_idx, const vector_double &r_point, hv_algorithm &hv_algo) const
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
    double exclusive(unsigned int p_idx, const vector_double &r_point) const
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
    std::vector<double> contributions(const vector_double &r_point, hv_algorithm &hv_algo) const
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
    std::vector<double> contributions(const vector_double &r_point) const
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
    unsigned long long least_contributor(const vector_double &r_point, hv_algorithm &hv_algo) const
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
    unsigned long long least_contributor(const vector_double &r_point) const
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
    unsigned long long greatest_contributor(const vector_double &r_point, hv_algorithm &hv_algo) const
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
    unsigned long long greatest_contributor(const vector_double &r_point) const
    {
        return greatest_contributor(r_point, *get_best_contributions(r_point));
    }

    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_points, m_copy_points, m_verify);
    }

private:
    mutable std::vector<vector_double> m_points;
    bool m_copy_points;
    bool m_verify;

    /// Verify after construct method
    /**
    * Verifies whether basic requirements are met for the initial set of points.
    *
    * @throws invalid_argument if point size is empty or when the dimensions among the points differ
    */
    void verify_after_construct() const
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
    }

    /// Verify before compute method
    /**
    * Verifies whether reference point and the hypervolume method meet certain criteria.
    *
    * @param r_point vector describing the reference point
    * @param hv_algo instance of the algorithm object used for the computation
    *
    * @throws value_error if reference point's and point set dimension do not agree
    */
    void verify_before_compute(const vector_double &r_point, hv_algorithm &hv_algo) const
    {
        if (m_points[0].size() != r_point.size()) {
            pagmo_throw(std::invalid_argument, "Point set dimensions and reference point dimension must be equal.");
        }
        hv_algo.verify_before_compute(m_points, r_point);
    }
};

namespace detail
{
/// Expected number of operations
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
inline double expected_hv_operations(vector_double::size_type n, vector_double::size_type d)
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
} // end namespace detail
} // end namespace pagmo

#endif
