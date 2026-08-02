/* Copyright 2017-2021 PaGMO development team

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

#ifndef PAGMO_DETAIL_NSGA3_IMPL_HPP
#define PAGMO_DETAIL_NSGA3_IMPL_HPP

#include <optional>
#include <random>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/rng.hpp>         // random_engine_type
#include <pagmo/types.hpp>

// NOTE: this header contains the numerical utilities of the NSGA-III algorithm.
// They are implementation details of pagmo::nsga3.
namespace pagmo
{

namespace detail
{

// Solve the linear system Ax = b by Gaussian elimination with partial pivoting.
// Returns an empty optional when A is (numerically) singular.
PAGMO_DLL_PUBLIC std::optional<vector_double> gaussian_elimination(std::vector<std::vector<double>>,
                                                                   const vector_double &);

// Achievement Scalarization Function
PAGMO_DLL_PUBLIC double achievement(const vector_double &, const vector_double &);

// Perpendicular distance to reference point vectors
PAGMO_DLL_PUBLIC double perpendicular_distance(const std::vector<double> &, const std::vector<double> &);

/* Choose single random element from vector container.
 * The random engine is supplied by the caller so that the choice depends only
 * on the state of that engine, and not on the global pagmo::random_device.
 */
template <class T>
T choose_random_element(const std::vector<T> &container, random_engine_type &random_engine){
    std::uniform_int_distribution<typename std::vector<T>::size_type> dist(0u, container.size() - 1u);
    return container[dist(random_engine)];
}

} // namespace detail

} // namespace pagmo

#endif
