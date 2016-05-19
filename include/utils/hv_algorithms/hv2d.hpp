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

#ifndef PAGMO_UTIL_HV_ALGORITHMS_HV2D_H
#define PAGMO_UTIL_HV_ALGORITHMS_HV2D_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "../../io.hpp"
#include "../../types.hpp"
#include "../../exceptions.hpp"

#include "hv_algorithm.hpp"
#include "hv3d.hpp"


namespace pagmo { 

/// hv2d hypervolume algorithm class
/**
 * This is the class containing the implementation of the hypervolume algorithm for the 2-dimensional fronts.
 * This method achieves the lower bound of n*log(n) time by sorting the initial set of points and then computing the partial areas linearly.
 *
 * @author Krzysztof Nowak (kn@kiryx.net)
 */
class hv2d : public hv_algorithm
{
public:
	/// Constructor
	hv2d(const bool initial_sorting = true) : m_initial_sorting(initial_sorting) { }

	/// Compute hypervolume method.
	/**
	* This method should be used both as a solution to 2-dimensional cases, and as a general termination method for algorithms that reduce n-dimensional problem to 2D.
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
		if (points.size() == 0) {
			return 0.0;
		}
		else if (points.size() == 1) {
			return hv_algorithm::volume_between(points[0], r_point);
		}

		if (m_initial_sorting) {
			sort(points.begin(), points.end(), vector_double_cmp(1, '<'));
		}

		double hypervolume = 0.0;

		// width of the sweeping line
		double w = r_point[0] - points[0][0];
		for (unsigned int idx = 0; idx < points.size() - 1; ++idx) {
			hypervolume += (points[idx + 1][1] - points[idx][1]) * w;
			w = std::max(w, r_point[0] - points[idx + 1][0]);
		}
		hypervolume += (r_point[1] - points[points.size() - 1][1]) * w;

		return hypervolume;
	}


	/// Compute hypervolume method.
	/**
	* This method should be used both as a solution to 2-dimensional cases, and as a general termination method for algorithms that reduce n-dimensional problem to 2d.
	* This method is overloaded to work with arrays of double, in order to provide other algorithms that internally work with arrays (such as hv_algorithm::wfg) with an efficient computation.
	*
	* Computational complexity: n*log(n)
	*
	* @param[in] points array of 2-dimensional points
	* @param[in] n_points number of points
	* @param[in] r_point 2-dimensional reference point for the points
	*
	* @return hypervolume
	*/
	double compute(double** points, unsigned int n_points, double* r_point) const
	{
		if (n_points == 0) {
			return 0.0;
		}
		else if (n_points == 1) {
			return volume_between(points[0], r_point, 2);
		}

		if (m_initial_sorting) {
			std::sort(points, points + n_points, hv2d::cmp_double_2d);
		}

		double hypervolume = 0.0;

		// width of the sweeping line
		double w = r_point[0] - points[0][0];
		for (unsigned int idx = 0; idx < n_points - 1; ++idx) {
			hypervolume += (points[idx + 1][1] - points[idx][1]) * w;
			w = std::max(w, r_point[0] - points[idx + 1][0]);
		}
		hypervolume += (r_point[1] - points[n_points - 1][1]) * w;

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
	std::vector<double> contributions(std::vector<vector_double> &points, const vector_double &r_point) const
	{
		std::vector<vector_double> new_points(points.size(), vector_double(3, 0.0));
		vector_double new_r(r_point);
		new_r.push_back(1.0);

		for (unsigned int i = 0; i < points.size(); ++i) {
			new_points[i][0] = points[i][0];
			new_points[i][1] = points[i][1];
			new_points[i][2] = 0.0;
		}
		// Set sorting to off since contributions are sorted by third dimension
		return hv3d(false).contributions(new_points, new_r);
	}


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
	* @throws value_error when trying to compute the hypervolume for the dimension other than 3 or non-maximal reference point
	*/
	void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const
	{
		if (r_point.size() != 2) {
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

	
	/// Comparison function for sorting of pairs (point, index)
	/**
	* Required for hv2d::extreme_contributor method for keeping track of the original indices when sorting.
	*/
	static bool point_pairs_cmp(const std::pair<vector_double, unsigned int> &a, const std::pair<vector_double, unsigned int> &b)
	{
		return a.first[1] > b.first[1];
	}
	
	
	/// Comparison function for arrays of double.
	/**
	* Required by the hv2d::compute method for the sorting of arrays of double*.
	*/
	static bool cmp_double_2d(double* a, double* b)
	{
		return a[1] < b[1];
	}
};

} // namespace pagmo

#endif
