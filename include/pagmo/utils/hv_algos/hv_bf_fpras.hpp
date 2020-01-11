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

#ifndef PAGMO_UTILS_HV_BF_FPRAS_HPP
#define PAGMO_UTILS_HV_BF_FPRAS_HPP

#include <memory>
#include <string>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>

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
class PAGMO_DLL_PUBLIC bf_fpras final : public hv_algorithm
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
    bf_fpras(double eps = 1e-2, double delta = 1e-2, unsigned seed = pagmo::random_device::next());

    // Verify before compute
    void verify_before_compute(const std::vector<vector_double> &, const vector_double &) const override;

    // Compute method
    double compute(std::vector<vector_double> &, const vector_double &) const override;

    // Exclusive method
    [[noreturn]] double exclusive(unsigned, std::vector<vector_double> &, const vector_double &) const override;

    // Least contributor method
    [[noreturn]] unsigned long long least_contributor(std::vector<vector_double> &,
                                                      const vector_double &) const override;

    // Greatest contributor method
    [[noreturn]] unsigned long long greatest_contributor(std::vector<vector_double> &,
                                                         const vector_double &) const override;

    // Contributions method
    [[noreturn]] vector_double contributions(std::vector<vector_double> &, const vector_double &) const override;

    // Clone method.
    std::shared_ptr<hv_algorithm> clone() const override;

    // Algorithm name
    std::string get_name() const override;

private:
    // error of the approximation
    const double m_eps;
    // probabiltiy of error
    const double m_delta;

    mutable detail::random_engine_type m_e;
};

} // namespace pagmo

#endif
