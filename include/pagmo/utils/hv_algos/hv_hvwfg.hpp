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

#ifndef PAGMO_UTILS_HV_HVWFG_HPP
#define PAGMO_UTILS_HV_HVWFG_HPP

#include <memory>
#include <string>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>

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
class PAGMO_DLL_PUBLIC hvwfg final : public hv_algorithm
{
public:
    /// Constructor
    /**
     * @param stop_dimension The stop dimension
     */
    hvwfg(unsigned stop_dimension = 2u);

    // Compute hypervolume
    double compute(std::vector<vector_double> &, const vector_double &) const override;

    // Contributions method
    std::vector<double> contributions(std::vector<vector_double> &, const vector_double &) const override;

    // Verify before compute method
    void verify_before_compute(const std::vector<vector_double> &, const vector_double &) const override;

    // Clone method.
    std::shared_ptr<hv_algorithm> clone() const override;

    // Algorithm name
    std::string get_name() const override;

private:
    // Limit the set of points to point at p_idx
    PAGMO_DLL_LOCAL void limitset(unsigned, unsigned, unsigned) const;

    // Compute the exclusive hypervolume of point at p_idx
    PAGMO_DLL_LOCAL double exclusive_hv(unsigned, unsigned) const;

    // Compute the hypervolume recursively
    PAGMO_DLL_LOCAL double compute_hv(unsigned) const;

    // Comparator function for sorting
    PAGMO_DLL_LOCAL bool cmp_points(double *, double *) const;

    // Allocate the memory for the 'compute' method
    PAGMO_DLL_LOCAL void allocate_wfg_members(std::vector<vector_double> &, const vector_double &) const;

    // Free the previously allocated memory
    PAGMO_DLL_LOCAL void free_wfg_members() const;

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
    mutable unsigned m_n_frames;

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
    const unsigned m_stop_dimension;
};

} // namespace pagmo

#endif
