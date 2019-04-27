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

#ifndef PAGMO_DETAIL_S11N_WRAPPERS_HPP
#define PAGMO_DETAIL_S11N_WRAPPERS_HPP

#include <utility>

namespace pagmo
{

namespace detail
{

// A few helpers to give a cereal-like syntax
// to Boost.serialization.
template <typename Archive>
inline void archive(Archive &)
{
}

template <typename Archive, typename Arg0, typename... Args>
inline void archive(Archive &ar, Arg0 &&arg0, Args &&... args)
{
    ar &std::forward<Arg0>(arg0);
    archive(ar, std::forward<Args>(args)...);
}

template <typename Archive>
inline void to_archive(Archive &)
{
}

template <typename Archive, typename Arg0, typename... Args>
inline void to_archive(Archive &ar, Arg0 &&arg0, Args &&... args)
{
    ar << std::forward<Arg0>(arg0);
    to_archive(ar, std::forward<Args>(args)...);
}

template <typename Archive>
inline void from_archive(Archive &)
{
}

template <typename Archive, typename Arg0, typename... Args>
inline void from_archive(Archive &ar, Arg0 &&arg0, Args &&... args)
{
    ar >> std::forward<Arg0>(arg0);
    from_archive(ar, std::forward<Args>(args)...);
}

} // namespace detail

} // namespace pagmo

#endif
