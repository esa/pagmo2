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

#ifndef PAGMO_DETAIL_BFE_IMPL_HPP
#define PAGMO_DETAIL_BFE_IMPL_HPP

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

// NOTE: this header contains various bfe-related utilities
// that are used in multiple places.
namespace pagmo
{

namespace detail
{

PAGMO_DLL_PUBLIC void bfe_check_input_dvs(const problem &, const vector_double &);

PAGMO_DLL_PUBLIC void bfe_check_output_fvs(const problem &, const vector_double &, const vector_double &);

} // namespace detail

} // namespace pagmo

#endif
