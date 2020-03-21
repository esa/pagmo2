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

#ifndef PAGMO_UTILS_HV_HV2D_HPP
#define PAGMO_UTILS_HV_HV2D_HPP

#include <memory>
#include <string>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>

namespace pagmo
{

/// hv2d hypervolume algorithm class
/**
 * This is the class containing the implementation of the hypervolume algorithm for the 2-dimensional fronts.
 * This method achieves the lower bound of n*log(n) time by sorting the initial set of points and then computing the
 * partial areas linearly.
 *
 */
class PAGMO_DLL_PUBLIC hv2d final : public hv_algorithm
{
public:
    /// Constructor
    /**
     * @param initial_sorting Turn initial sorting on-off
     */
    hv2d(const bool initial_sorting = true);

    // Compute hypervolume method.
    double compute(std::vector<vector_double> &, const vector_double &) const override;

    // Compute hypervolume method.
    double compute(double **, vector_double::size_type, double *) const;

    // Contributions method
    std::vector<double> contributions(std::vector<vector_double> &, const vector_double &) const override;

    // Clone method.
    std::shared_ptr<hv_algorithm> clone() const override;

    // Verify input method.
    void verify_before_compute(const std::vector<vector_double> &, const vector_double &) const override;

    // Algorithm name
    std::string get_name() const override;

private:
    // Flag stating whether the points should be sorted in the first step of the algorithm.
    const bool m_initial_sorting;
};
} // namespace pagmo

#endif
