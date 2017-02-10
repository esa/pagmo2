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

#ifndef PAGMO_UTIL_hv2d_H
#define PAGMO_UTIL_hv2d_H

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../exceptions.hpp"
#include "../../io.hpp"
#include "../../population.hpp"
#include "../../types.hpp"
#include "../hypervolume.hpp"
#include "hv_algorithm.hpp"

namespace pagmo
{

/// hv2d hypervolume algorithm class
/**
 * This is the class containing the implementation of the hypervolume algorithm for the 2-dimensional fronts.
 * This method achieves the lower bound of n*log(n) time by sorting the initial set of points and then computing the
 * partial areas linearly.
 *
 */
class hv2d : public hv_algorithm
{
public:
    /// Constructor
    hv2d(const bool initial_sorting = true) : m_initial_sorting(initial_sorting)
    {
    }

    /// Compute hypervolume method.
    /**
    * This method should be used both as a solution to 2-dimensional cases, and as a general termination method for
    * algorithms that reduce n-dimensional problem to 2D.
    *
    * Computational complexity: n*log(n)
    *
    * @param[in] points vector of points containing the 2-dimensional points for which we compute the hypervolume
    * @param[in] r_point reference point for the points
    *
    * @return hypervolume
    */
    double compute(std::vector<vector_double> &points, const vector_double &r_point) const
    {
        if (points.size() == 0u) {
            return 0.0;
        } else if (points.size() == 1u) {
            return hv_algorithm::volume_between(points[0], r_point);
        }

        if (m_initial_sorting) {
            sort(points.begin(), points.end(),
                 [](const vector_double &v1, const vector_double &v2) { return v1[1] < v2[1]; });
        }

        double hypervolume = 0.0;

        // width of the sweeping line
        double w = r_point[0] - points[0][0];
        for (decltype(points.size()) idx = 0u; idx < points.size() - 1u; ++idx) {
            hypervolume += (points[idx + 1u][1] - points[idx][1]) * w;
            w = std::max(w, r_point[0] - points[idx + 1u][0]);
        }
        hypervolume += (r_point[1] - points[points.size() - 1u][1]) * w;

        return hypervolume;
    }

    /// Compute hypervolume method.
    /**
    * This method should be used both as a solution to 2-dimensional cases, and as a general termination method for
    * algorithms that reduce n-dimensional problem to 2d.
    * This method is overloaded to work with arrays of double, in order to provide other algorithms that internally work
    * with arrays (such as hv_algorithm::wfg) with an efficient computation.
    *
    * Computational complexity: n*log(n)
    *
    * @param[in] points array of 2-dimensional points
    * @param[in] n_points number of points
    * @param[in] r_point 2-dimensional reference point for the points
    *
    * @return hypervolume
    */
    double compute(double **points, vector_double::size_type n_points, double *r_point) const
    {
        if (n_points == 0u) {
            return 0.0;
        } else if (n_points == 1u) {
            return volume_between(points[0], r_point, 2);
        }

        if (m_initial_sorting) {
            std::sort(points, points + n_points, [](double *a, double *b) { return a[1] < b[1]; });
        }

        double hypervolume = 0.0;

        // width of the sweeping line
        double w = r_point[0] - points[0][0];
        for (decltype(n_points) idx = 0; idx < n_points - 1u; ++idx) {
            hypervolume += (points[idx + 1u][1] - points[idx][1]) * w;
            w = std::max(w, r_point[0] - points[idx + 1u][0]);
        }
        hypervolume += (r_point[1] - points[n_points - 1u][1]) * w;

        return hypervolume;
    }

    /// Contributions method
    /**
    * Computes the contributions of each point by invoking the HV3D algorithm with mock third dimension.
    *
    * @param[in] points vector of points containing the 2-dimensional points for which we compute the hypervolume
    * @param[in] r_point reference point for the points
    * @return vector of exclusive contributions by every point
    */
    std::vector<double> contributions(std::vector<vector_double> &points, const vector_double &r_point) const;

    /// Clone method.
    std::shared_ptr<hv_algorithm> clone() const
    {
        return std::shared_ptr<hv_algorithm>(new hv2d(*this));
    }

    /// Verify input method.
    /**
    * Verifies whether the requested data suits the hv2d algorithm.
    *
    * @param[in] points vector of points containing the d dimensional points for which we compute the hypervolume
    * @param[in] r_point reference point for the vector of points
    *
    * @throws value_error when trying to compute the hypervolume for the dimension other than 3 or non-maximal reference
    * point
    */
    void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const
    {
        if (r_point.size() != 2u) {
            pagmo_throw(std::invalid_argument, "Algorithm hv2d works only for 2-dimensional cases.");
        }

        hv_algorithm::assert_minimisation(points, r_point);
    }

    /// Algorithm name
    std::string get_name() const
    {
        return "hv2d algorithm";
    }

private:
    // Flag stating whether the points should be sorted in the first step of the algorithm.
    const bool m_initial_sorting;
};
}

#endif
