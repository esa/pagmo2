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

#ifndef PYGMO_NUMPY_HPP
#define PYGMO_NUMPY_HPP

#include <pygmo/python_includes.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include <numpy/arrayobject.h>

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#undef NPY_NO_DEPRECATED_API

// NOTE: if the NO_IMPORT_ARRAY definition is active,
// the import_array() macro is not defined.
#if !defined(NO_IMPORT_ARRAY)

namespace pygmo
{

// This is necessary because the NumPy macro import_array() has different return values
// depending on the Python version.
#if PY_MAJOR_VERSION < 3
inline void numpy_import_array()
{
    import_array();
}
#else
inline void *numpy_import_array()
{
    import_array();
    return nullptr;
}
#endif
} // namespace pygmo

#endif

#endif
