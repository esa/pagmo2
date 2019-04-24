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

#ifndef PAGMO_DETAIL_VISIBILITY_HPP
#define PAGMO_DETAIL_VISIBILITY_HPP

// Convenience macros for visibility attributes. Mostly insipred by:
// https://gcc.gnu.org/wiki/Visibility
// We check first for Windows, where we assume every compiler
// knows dllexport/dllimport. On other platforms, we use the GCC-like
// syntax for GCC, clang and ICC. Otherwise, we leave PAGMO_PUBLIC
// empty.
#if defined(_WIN32) || defined(__CYGWIN__)

#if defined(pagmo_EXPORTS)

#define PAGMO_PUBLIC __declspec(dllexport)

#else

#define PAGMO_PUBLIC __declspec(dllimport)

#endif

#elif defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)

#define PAGMO_PUBLIC __attribute__((visibility("default")))

#else

#define PAGMO_PUBLIC

#endif

#endif
