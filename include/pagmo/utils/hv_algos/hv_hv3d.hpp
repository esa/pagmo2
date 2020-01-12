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

#ifndef PAGMO_UTILS_HV_HV3D_HPP
#define PAGMO_UTILS_HV_HV3D_HPP

#include <memory>
#include <string>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>

namespace pagmo
{

/// hv3d hypervolume algorithm class
/**
 * This class contains the implementation of efficient algorithms for the hypervolume computation in 3-dimensions.
 *
 * 'compute' method relies on the efficient algorithm as it was presented by Nicola Beume et al.
 * 'least[greatest]_contributor' methods rely on the HyCon3D algorithm by Emmerich and Fonseca.
 *
 * @see "On the Complexity of Computing the Hypervolume Indicator", Nicola Beume, Carlos M. Fonseca, Manuel
 * Lopez-Ibanez, Luis Paquete, Jan Vahrenhold. IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 13, NO. 5, OCTOBER
 * 2009
 * @see "Computing hypervolume contribution in low dimensions: asymptotically optimal algorithm and complexity results",
 * Michael T. M. Emmerich, Carlos M. Fonseca
 */
class PAGMO_DLL_PUBLIC hv3d final : public hv_algorithm
{
public:
    /// Constructor
    /**
     * Constructor of the algorithm object.
     * In the very first step, algorithm requires the inital set of points to be sorted ASCENDING in the third
     * dimension. If the input is already sorted, user can skip this step using "initial_sorting = false" option, saving
     * some extra time.
     *
     * @param initial_sorting when set to true (default), algorithm will sort the points ascending by third dimension
     */
    hv3d(const bool initial_sorting = true);

    // Compute hypervolume
    double compute(std::vector<vector_double> &, const vector_double &) const override;

    // Contributions method
    std::vector<double> contributions(std::vector<vector_double> &, const vector_double &) const override;

    // Verify before compute
    void verify_before_compute(const std::vector<vector_double> &, const vector_double &) const override;

    // Clone method.
    std::shared_ptr<hv_algorithm> clone() const override;

    // Algorithm name
    std::string get_name() const override;

private:
    // flag stating whether the points should be sorted in the first step of the algorithm
    const bool m_initial_sorting;
};

} // namespace pagmo

#endif
