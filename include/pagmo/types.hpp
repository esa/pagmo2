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

#ifndef PAGMO_TYPES_HPP
#define PAGMO_TYPES_HPP

#include <utility>
#include <vector>

/// Root PaGMO namespace.
namespace pagmo
{

/// Alias for an <tt>std::vector</tt> of <tt>double</tt>s.
typedef std::vector<double> vector_double;
/// Alias for an <tt>std::vector</tt> of <tt>std::pair</tt>s of the size type of pagmo::vector_double.
typedef std::vector<std::pair<vector_double::size_type, vector_double::size_type>> sparsity_pattern;

} // namespace pagmo

#endif
