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

#ifndef PAGMO_DETAIL_GTE_GETTER_HPP
#define PAGMO_DETAIL_GTE_GETTER_HPP

#include <functional>

#include <boost/any.hpp>

#include <pagmo/detail/visibility.hpp>

namespace pagmo
{

namespace detail
{

// NOTE: in some cases, we need to call into Python
// from a thread created within C++ (e.g., the island thread).
// This function will return a RAII-style object that
// can be used to ensure that we can safely call into Python
// from the external thread. When working in C++, this
// functor will return an empty object with no side
// effects.
PAGMO_DLL_PUBLIC extern std::function<boost::any()> gte_getter;

} // namespace detail

} // namespace pagmo

#endif
