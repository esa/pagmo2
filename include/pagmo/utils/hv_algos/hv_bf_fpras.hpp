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

#ifndef PAGMO_UTIL_bf_fpras_H
#define PAGMO_UTIL_bf_fpras_H

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
#include <pagmo/utils/hypervolume.hpp>

namespace pagmo
{

/// Bringmann-Friedrich approximation method
/**
 * This class contains the implementation of the Bringmann-Friedrich approximation scheme (FPRAS),
 * reduced to a special case of approximating the hypervolume indicator.
 * @see "Approximating the volume of unions and intersections of high-dimensional geometric objects", Karl Bringmann,
 * Tobias Friedrich.
 *
 */

class bf_fpras final : public hv_algorithm
{
public:
    /// Constructor
    /**
     * Constructs an instance of the algorithm
     *
     * @param eps accuracy of the approximation
     * @param delta confidence of the approximation
     * @param seed seeding for the pseudo-random number generator
     */
    bf_fpras(double eps = 1e-2, double delta = 1e-2, unsigned seed = pagmo::random_device::next())
        : m_eps(eps), m_delta(delta), m_e(seed)
    {
        if (eps <= 0 || eps > 1) {
            pagmo_throw(std::invalid_argument, "Epsilon needs to be a probability greater then zero");
        }
        if (delta <= 0 || delta > 1) {
            pagmo_throw(std::invalid_argument, "Delta needs to be a probability greater than zero");
        }
    }

    /// Verify before compute
    /**
     * Verifies whether given algorithm suits the requested data.
     *
     * @param points vector of points containing the d dimensional points for which we compute the hypervolume
     * @param r_point reference point for the vector of points
     *
     * @throws value_error when trying to compute the hypervolume for the non-maximal reference point
     */
    void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const override
    {
        hv_algorithm::assert_minimisation(points, r_point);
    }

    /// Compute method
    /**
     * Compute the hypervolume using FPRAS.
     *
     * @see "Approximating the volume of unions and intersections of high-dimensional geometric objects", Karl
     * Bringmann, Tobias Friedrich.
     *
     * @param points vector of fitness_vectors for which the hypervolume is computed
     * @param r_point distinguished "reference point".
     *
     * @return approximated hypervolume
     */
    double compute(std::vector<vector_double> &points, const vector_double &r_point) const override
    {
        auto n = points.size();
        auto dim = r_point.size();

        auto T = std::floor(12. * std::log(1. / m_delta) / std::log(2.) * static_cast<double>(n) / m_eps / m_eps);

        // Partial sums of consecutive boxes
        vector_double sums(n, 0.0);

        // points iterator
        std::vector<vector_double>::iterator it_p;

        // volume iterator - used for finding the contributor using std::lower_bound
        vector_double::iterator it_sums;

        unsigned i = 0u;

        // Total sum of every box
        double V = 0.0;
        for (it_p = points.begin(); it_p != points.end(); ++it_p) {
            V = (sums[i++] = V + hv_algorithm::volume_between(*it_p, r_point));
        }

        double M = 0.;     // Round counter
        double M_sum = 0.; // Total number of samples over every round so far

        vector_double rnd_point(dim, 0.0); // Container for the random point
        auto unireal_dist = std::uniform_real_distribution<double>(0.0, 1.0);

        while (true) {
            // Get the random volume in-between [0, V] range, in order to choose the box with probability sums[i] / V

            auto V_dist = std::uniform_real_distribution<double>(0.0, V);
            auto r = V_dist(m_e);

            // Find the contributor using binary search
            it_sums = std::lower_bound(sums.begin(), sums.end(), r);
            i = static_cast<unsigned>(std::distance(sums.begin(), it_sums));

            // Sample a point inside the 'box' (r_point, points[i])
            for (decltype(dim) d_idx = 0u; d_idx < dim; ++d_idx) {
                rnd_point[d_idx] = (points[i][d_idx] + unireal_dist(m_e) * (r_point[d_idx] - points[i][d_idx]));
            }

            unsigned j = 0u;
            do {
                if (M_sum >= T) {
                    return (T * V) / (static_cast<double>(n) * M);
                }
                j = static_cast<unsigned>(static_cast<double>(n) * unireal_dist(m_e));
                ++M_sum;
            } while (!(hv_algorithm::dom_cmp(rnd_point, points[j], 0) == hv_algorithm::DOM_CMP_B_DOMINATES_A));
            ++M;
        }
    }

    /// Exclusive method
    /**
     * This algorithm does not support this method.
     * @return Nothing as it throws before
     */
    double exclusive(unsigned, std::vector<vector_double> &, const vector_double &) const override
    {
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Least contributor method
    /**
     * This algorithm does not support this method.
     *
     * @return Nothing as it throws before
     */
    unsigned long long least_contributor(std::vector<vector_double> &, const vector_double &) const override
    {
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Greatest contributor method
    /**
     * This algorithm does not support this method.
     * @return Nothing as it throws before
     */
    unsigned long long greatest_contributor(std::vector<vector_double> &, const vector_double &) const override
    {
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Contributions method
    /**
     * As of yet, this algorithm does not support this method, even in its naive form, due to a poor handling of the
     * dominated points.
     * @return Nothing as it throws before
     */
    vector_double contributions(std::vector<vector_double> &, const vector_double &) const override
    {
        pagmo_throw(std::invalid_argument, "This method is not supported by the bf_fpras algorithm");
    }

    /// Clone method.
    /**
     * @return a pointer to a new object cloning this
     */
    std::shared_ptr<hv_algorithm> clone() const override
    {
        return std::shared_ptr<hv_algorithm>(new bf_fpras(*this));
    }

    /// Algorithm name
    /**
     * @return The name of this particular algorithm
     */
    std::string get_name() const override
    {
        return "bf_fpras algorithm";
    }

private:
    // error of the approximation
    const double m_eps;
    // probabiltiy of error
    const double m_delta;

    mutable detail::random_engine_type m_e;
};
} // namespace pagmo

#endif
