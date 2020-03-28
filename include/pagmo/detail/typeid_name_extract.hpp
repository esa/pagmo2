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

#ifndef PAGMO_DETAIL_TYPEID_NAME_EXTRACT_HPP
#define PAGMO_DETAIL_TYPEID_NAME_EXTRACT_HPP

#include <cstring>
#include <type_traits>
#include <typeinfo>

#include <pagmo/type_traits.hpp>

namespace pagmo
{

namespace detail
{

// This is an implementation of the extract() functionality
// for UDx classes based on the name() of the UDx C++ type,
// as returned by typeid().name(). This is needed
// because the dynamic_cast() used in the
// usual extract() implementations can fail on some
// compiler/platform/stdlib implementations
// when crossing boundaries between dlopened()
// modules. See:
// https://github.com/pybind/pybind11/issues/912#issuecomment-310157016
// https://bugs.llvm.org/show_bug.cgi?id=33542
template <typename T, typename C>
inline typename std::conditional<std::is_const<C>::value, const T *, T *>::type typeid_name_extract(C &class_inst)
{
    // NOTE: typeid() strips away both reference and cv qualifiers. Thus,
    // if T is cv-qualified or a reference type, return nullptr pre-empitvely
    // (in any case, extraction cannot be successful in such cases).
    if (!std::is_same<T, uncvref_t<T>>::value || std::is_reference<T>::value) {
        return nullptr;
    }

    if (std::strcmp(class_inst.get_type_index().name(), typeid(T).name())) {
        // The names differ, return null.
        return nullptr;
    } else {
        // The names match, cast to the correct type and return.
        return static_cast<typename std::conditional<std::is_const<C>::value, const T *, T *>::type>(
            class_inst.get_ptr());
    }
}

} // namespace detail

} // namespace pagmo

#endif
