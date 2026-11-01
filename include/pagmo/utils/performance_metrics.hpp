/* Copyright 2017-2026 PaGMO development team

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

#ifndef PAGMO_UTILS_PERFORMANCE_METRICS_HPP
#define PAGMO_UTILS_PERFORMANCE_METRICS_HPP

#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Hypervolume ratio
PAGMO_DLL_PUBLIC double hypervolume_ratio(const std::vector<vector_double> &, const std::vector<vector_double> &,
                                          const vector_double &);

// Modified inverted generational distance (IGD+)
PAGMO_DLL_PUBLIC double igd_plus(const std::vector<vector_double> &, const std::vector<vector_double> &);

// Normalized modified inverted generational distance (NIGD+)
PAGMO_DLL_PUBLIC double normalized_igd_plus(const std::vector<vector_double> &, const std::vector<vector_double> &,
                                            const vector_double &);

} // namespace pagmo

#endif
