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

#ifndef PAGMO_UTIL_hvwfg_H
#define PAGMO_UTIL_hvwfg_H

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

// WFG hypervolume algorithm
/**
 * This is the class containing the implementation of the WFG algorithm for the computation of hypervolume indicator.
 *
 * @see "While, Lyndon, Lucas Bradstreet, and Luigi Barone. "A fast way of calculating exact hypervolumes." Evolutionary
 * Computation, IEEE Transactions on 16.1 (2012): 86-95."
 * @see "Lyndon While and Lucas Bradstreet. Applying the WFG Algorithm To Calculate Incremental Hypervolumes. 2012 IEEE
 * Congress on Evolutionary Computation. CEC 2012, pages 489-496. IEEE, June 2012."
 */
class hvwfg final : public hv_algorithm
{
public:
    /// Constructor
    /**
     * @param stop_dimension The stop dimension
     */
    hvwfg(unsigned int stop_dimension = 2u) : hv_algorithm(), m_current_slice(0), m_stop_dimension(stop_dimension)
    {
        if (stop_dimension < 2u) {
            pagmo_throw(std::invalid_argument, "Stop dimension for WFG must be greater than or equal to 2");
        }
    }

    /// Compute hypervolume
    /**
     * Computes the hypervolume using the WFG algorithm.
     *
     * @param points vector of points containing the D-dimensional points for which we compute the hypervolume
     * @param r_point reference point for the points
     *
     * @return hypervolume.
     */
    double compute(std::vector<vector_double> &points, const vector_double &r_point) const override
    {
        allocate_wfg_members(points, r_point);
        double hv = compute_hv(1);
        free_wfg_members();
        return hv;
    }

    /// Contributions method
    /**
     * This method employs a slightly modified version of the original WFG algorithm to suit the computation of the
     * exclusive contributions.
     * It differs from the IWFG algorithm (referenced below), as we do not use the priority-queueing mechanism, but
     * compute every exclusive contribution instead.
     * This may suggest that the algorithm for the extreme contributor itself reduces to the 'naive' approach. It is not
     * the case however,
     * as we utilize the benefits of the 'limitset', before we begin the recursion.
     * This simplifies the sub problems for each exclusive computation right away, which makes the whole algorithm much
     * faster, and in many cases only slower than regular WFG algorithm by a constant factor.
     *
     * @see "Lyndon While and Lucas Bradstreet. Applying the WFG Algorithm To Calculate Incremental Hypervolumes. 2012
     * IEEE Congress on Evolutionary Computation. CEC 2012, pages 489-496. IEEE, June 2012."
     *
     * @param points vector of points containing the D-dimensional points for which we compute the hypervolume
     * @param r_point reference point for the points
     *
     * @return the single contributions
     */
    std::vector<double> contributions(std::vector<vector_double> &points, const vector_double &r_point) const override
    {
        std::vector<double> c;
        c.reserve(points.size());

        // Allocate the same members as for 'compute' method
        allocate_wfg_members(points, r_point);

        // Prepare the memory for first front
        double **fr = new double *[m_max_points];
        for (unsigned int i = 0; i < m_max_points; ++i) {
            fr[i] = new double[m_current_slice];
        }
        m_frames[m_n_frames] = fr;
        m_frames_size[m_n_frames] = 0;
        ++m_n_frames;

        for (unsigned int p_idx = 0u; p_idx < m_max_points; ++p_idx) {
            limitset(0, p_idx, 1);
            c.push_back(exclusive_hv(p_idx, 1));
        }

        // Free the contributions and the remaining WFG members
        free_wfg_members();

        return c;
    }

    /// Verify before compute method
    /**
     * Verifies whether given algorithm suits the requested data.
     *
     * @param points vector of points containing the D-dimensional points for which we compute the hypervolume
     * @param r_point reference point for the vector of points
     *
     * @throws value_error when trying to compute the hypervolume for the non-maximal reference point
     */
    void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const override
    {
        hv_algorithm::assert_minimisation(points, r_point);
    }

    /// Clone method.
    /**
     * @return a pointer to a new object cloning this
     */
    std::shared_ptr<hv_algorithm> clone() const override
    {
        return std::shared_ptr<hv_algorithm>(new hvwfg(*this));
    }

    /// Algorithm name
    /**
     * @return The name of this particular algorithm
     */
    std::string get_name() const override
    {
        return "WFG algorithm";
    }

private:
    /// Limit the set of points to point at p_idx
    void limitset(unsigned int begin_idx, unsigned int p_idx, unsigned int rec_level) const
    {
        double **points = m_frames[rec_level - 1];
        auto n_points = m_frames_size[rec_level - 1];

        vector_double::size_type no_points = 0u;

        double *p = points[p_idx];
        double **frame = m_frames[rec_level];

        for (auto idx = begin_idx; idx < n_points; ++idx) {
            if (idx == p_idx) {
                continue;
            }

            for (decltype(m_current_slice) f_idx = 0u; f_idx < m_current_slice; ++f_idx) {
                frame[no_points][f_idx] = std::max(points[idx][f_idx], p[f_idx]);
            }

            std::vector<int> cmp_results;
            cmp_results.resize(no_points);
            double *s = frame[no_points];

            bool keep_s = true;

            // Check whether any point is dominating the point 's'.
            for (decltype(no_points) q_idx = 0u; q_idx < no_points; ++q_idx) {
                cmp_results[q_idx] = hv_algorithm::dom_cmp(s, frame[q_idx], m_current_slice);
                if (cmp_results[q_idx] == hv_algorithm::DOM_CMP_B_DOMINATES_A) {
                    keep_s = false;
                    break;
                }
            }

            // If neither is, remove points dominated by 's' (we store that during the first loop).
            if (keep_s) {
                vector_double::size_type prev = 0u;
                vector_double::size_type next = 0u;
                while (next < no_points) {
                    if (cmp_results[next] != hv_algorithm::DOM_CMP_A_DOMINATES_B
                        && cmp_results[next] != hv_algorithm::DOM_CMP_A_B_EQUAL) {
                        if (prev < next) {
                            for (decltype(m_current_slice) d_idx = 0u; d_idx < m_current_slice; ++d_idx) {
                                frame[prev][d_idx] = frame[next][d_idx];
                            }
                        }
                        ++prev;
                    }
                    ++next;
                }
                // Append 's' at the end, if prev==next it's not necessary as it's already there.
                if (prev < next) {
                    for (decltype(m_current_slice) d_idx = 0u; d_idx < m_current_slice; ++d_idx) {
                        frame[prev][d_idx] = s[d_idx];
                    }
                }
                no_points = prev + 1u;
            }
        }

        m_frames_size[rec_level] = no_points;
    }

    /// Compute the exclusive hypervolume of point at p_idx
    double exclusive_hv(unsigned int p_idx, unsigned int rec_level) const
    {
        // double H = hv_algorithm::volume_between(points[p_idx], m_refpoint, m_current_slice);
        double H = hv_algorithm::volume_between(m_frames[rec_level - 1][p_idx], m_refpoint, m_current_slice);

        if (m_frames_size[rec_level] == 1) {
            H -= hv_algorithm::volume_between(m_frames[rec_level][0], m_refpoint, m_current_slice);
        } else if (m_frames_size[rec_level] > 1) {
            H -= compute_hv(rec_level + 1);
        }

        return H;
    }

    /// Compute the hypervolume recursively
    double compute_hv(unsigned int rec_level) const
    {
        double **points = m_frames[rec_level - 1];
        auto n_points = m_frames_size[rec_level - 1];

        // Simple inclusion-exclusion for one and two points
        if (n_points == 1u) {
            return hv_algorithm::volume_between(points[0], m_refpoint, m_current_slice);
        } else if (n_points == 2u) {
            double hv = hv_algorithm::volume_between(points[0], m_refpoint, m_current_slice)
                        + hv_algorithm::volume_between(points[1], m_refpoint, m_current_slice);
            double isect = 1.0;
            for (decltype(m_current_slice) i = 0u; i < m_current_slice; ++i) {
                isect *= (m_refpoint[i] - std::max(points[0][i], points[1][i]));
            }
            return hv - isect;
        }

        // If already sliced to dimension at which we use another algorithm.
        if (m_current_slice == m_stop_dimension) {

            if (m_stop_dimension == 2u) {
                // Use a very efficient version of hv2d
                return hv2d().compute(points, n_points, m_refpoint);
            } else {
                // Let hypervolume object pick the best method otherwise.
                std::vector<vector_double> points_cpy;
                points_cpy.reserve(n_points);
                for (decltype(n_points) i = 0u; i < n_points; ++i) {
                    points_cpy.push_back(vector_double(points[i], points[i] + m_current_slice));
                }
                vector_double r_cpy(m_refpoint, m_refpoint + m_current_slice);

                hypervolume hv = hypervolume(points_cpy, false);
                hv.set_copy_points(false);
                return hv.compute(r_cpy);
            }
        } else {
            // Otherwise, sort the points in preparation for the next recursive step
            // Bind the object under "this" pointer to the cmp_points method so it can be used as a valid comparator
            // function for std::sort
            // We need that in order for the cmp_points to have acces to the m_current_slice member variable.
            std::sort(points, points + n_points, [this](double *a, double *b) { return this->cmp_points(a, b); });
        }

        double H = 0.0;
        --m_current_slice;

        if (rec_level >= m_n_frames) {
            double **fr = new double *[m_max_points];
            for (decltype(m_max_points) i = 0u; i < m_max_points; ++i) {
                fr[i] = new double[m_current_slice];
            }
            m_frames[m_n_frames] = fr;
            m_frames_size[m_n_frames] = 0u;
            ++m_n_frames;
        }

        for (unsigned int p_idx = 0u; p_idx < n_points; ++p_idx) {
            limitset(p_idx + 1u, p_idx, rec_level);

            H += std::abs((points[p_idx][m_current_slice] - m_refpoint[m_current_slice])
                          * exclusive_hv(p_idx, rec_level));
        }
        ++m_current_slice;
        return H;
    }

    /// Comparator function for sorting
    /**
     * Comparison function for WFG. Can't be static in order to have access to member variable m_current_slice.
     */
    bool cmp_points(double *a, double *b) const
    {
        for (auto i = m_current_slice; i > 0u; --i) {
            if (a[i - 1] > b[i - 1]) {
                return true;
            } else if (a[i - 1] < b[i - 1]) {
                return false;
            }
        }
        return false;
    }

    /// Allocate the memory for the 'compute' method
    void allocate_wfg_members(std::vector<vector_double> &points, const vector_double &r_point) const
    {
        m_max_points = points.size();
        m_max_dim = r_point.size();

        m_refpoint = new double[m_max_dim];
        for (decltype(m_max_dim) d_idx = 0u; d_idx < m_max_dim; ++d_idx) {
            m_refpoint[d_idx] = r_point[d_idx];
        }

        // Reserve the space beforehand for each level or recursion.
        // WFG with slicing feature will not go recursively deeper than the dimension size.
        m_frames = new double **[m_max_dim];
        m_frames_size = new vector_double::size_type[m_max_dim];

        // Copy the initial set into the frame at index 0.
        double **fr = new double *[m_max_points];
        for (decltype(m_max_points) p_idx = 0; p_idx < m_max_points; ++p_idx) {
            fr[p_idx] = new double[m_max_dim];
            for (decltype(m_max_dim) d_idx = 0u; d_idx < m_max_dim; ++d_idx) {
                fr[p_idx][d_idx] = points[p_idx][d_idx];
            }
        }
        m_frames[0] = fr;
        m_frames_size[0] = m_max_points;
        m_n_frames = 1u;

        // Variable holding the current "depth" of dimension slicing. We progress by slicing dimensions from the end.
        m_current_slice = m_max_dim;
    }

    /// Free the previously allocated memory
    void free_wfg_members() const
    {
        // Free the memory.
        delete[] m_refpoint;

        for (decltype(m_n_frames) fr_idx = 0u; fr_idx < m_n_frames; ++fr_idx) {
            for (decltype(m_max_points) p_idx = 0u; p_idx < m_max_points; ++p_idx) {
                delete[] m_frames[fr_idx][p_idx];
            }
            delete[] m_frames[fr_idx];
        }
        delete[] m_frames;
        delete[] m_frames_size;
    }

    /**
     * 'compute' and 'extreme_contributor' method variables section.
     *
     * Variables below (especially the pointers m_frames, m_frames_size and m_refpoint) are initialized
     * at the beginning of the 'compute' and 'extreme_contributor' methods, and freed afterwards.
     * The state of the variables is irrelevant outside the scope of the these methods.
     */

    // Current slice depth
    mutable vector_double::size_type m_current_slice;

    // Array of point sets for each recursive level.
    mutable double ***m_frames;

    // Maintains the number of points at given recursion level.
    mutable vector_double::size_type *m_frames_size;

    // Keeps track of currently allocated number of frames.
    mutable unsigned int m_n_frames;

    // Copy of the reference point
    mutable double *m_refpoint;

    // Size of the original front
    mutable vector_double::size_type m_max_points;

    // Size of the dimension
    mutable vector_double::size_type m_max_dim;
    /**
     * End of 'compute' method variables section.
     */

    // Dimension at which WFG stops the slicing
    const unsigned int m_stop_dimension;
};
} // namespace pagmo
#endif
