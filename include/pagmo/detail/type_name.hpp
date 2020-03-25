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

#ifndef PAGMO_DETAIL_TYPE_NAME_HPP
#define PAGMO_DETAIL_TYPE_NAME_HPP

#include <string>
#include <type_traits>
#include <typeinfo>

#include <pagmo/detail/visibility.hpp>

namespace pagmo
{

namespace detail
{

PAGMO_DLL_PUBLIC std::string demangle_from_typeid(const char *);

// Determine the name of the type T at runtime.
template <typename T>
inline std::string type_name()
{
    // Get the demangled name without cvref.
    auto ret
        = demangle_from_typeid(typeid(typename std::remove_cv<typename std::remove_reference<T>::type>::type).name());

    // Redecorate it with cv qualifiers.
    constexpr unsigned flag = unsigned(std::is_const<typename std::remove_reference<T>::type>::value)
                              + (unsigned(std::is_volatile<typename std::remove_reference<T>::type>::value) << 1);
    switch (flag) {
        case 0u:
            // NOTE: handle this explicitly to keep compiler warnings at bay.
            break;
        case 1u:
            ret += " const";
            break;
        case 2u:
            ret += " volatile";
            break;
        case 3u:
            ret += " const volatile";
    }

    // Re-add the reference, if necessary.
    if (std::is_lvalue_reference<T>::value) {
        ret += " &";
    } else if (std::is_rvalue_reference<T>::value) {
        ret += " &&";
    }

    return ret;
}

} // namespace detail

} // namespace pagmo

#endif
