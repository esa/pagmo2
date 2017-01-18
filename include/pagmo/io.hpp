/* Copyright 2017 PaGMO development team

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

#ifndef PAGMO_IO_HPP
#define PAGMO_IO_HPP

#include <iostream>
#include <utility>
#include <vector>

#define PAGMO_MAX_OUTPUT_LENGTH 5u

namespace pagmo
{

/// Forward declaration
template <typename... Args>
void stream(std::ostream &, const Args &...);

namespace detail
{

template <typename T>
inline void stream_impl(std::ostream &os, const T &x)
{
    os << x;
}

inline void stream_impl(std::ostream &os, const bool &b)
{
    if (b) {
        os << "true";
    } else {
        os << "false";
    }
}

template <typename T>
inline void stream_impl(std::ostream &os, const std::vector<T> &v)
{
    auto len = v.size();
    if (len <= PAGMO_MAX_OUTPUT_LENGTH) {
        os << '[';
        for (decltype(v.size()) i = 0u; i < v.size(); ++i) {
            stream(os, v[i]);
            if (i != v.size() - 1u) {
                os << ", ";
            }
        }
        os << ']';
    } else {
        os << '[';
        for (decltype(v.size()) i = 0u; i < PAGMO_MAX_OUTPUT_LENGTH; ++i) {
            stream(os, v[i], ", ");
        }
        os << "... ]";
    }
}

template <typename T, typename U>
inline void stream_impl(std::ostream &os, const std::pair<T, U> &p)
{
    stream(os, '(', p.first, ',', p.second, ')');
}

template <typename T, typename... Args>
inline void stream_impl(std::ostream &os, const T &x, const Args &... args)
{
    stream_impl(os, x);
    stream_impl(os, args...);
}

} // end of namespace detail

/// The PaGMO streaming function
template <typename... Args>
inline void stream(std::ostream &os, const Args &... args)
{
    detail::stream_impl(os, args...);
}

/// The PaGMO print function
template <typename... Args>
inline void print(const Args &... args)
{
    stream(std::cout, args...);
}

} // end of namespace pagmo

#undef PAGMO_MAX_OUTPUT_LENGTH

#endif
