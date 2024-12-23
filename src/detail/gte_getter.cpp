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

#include <functional>

#include <boost/any.hpp>

#include <pagmo/detail/gte_getter.hpp>

namespace pagmo
{

namespace detail
{

namespace
{

boost::any default_gte_getter()
{
    return boost::any{};
}

} // namespace

/// @cond
std::function<boost::any()> gte_getter = &default_gte_getter;
/// @endcond

} // namespace detail

} // namespace pagmo
