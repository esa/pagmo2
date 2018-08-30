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

#ifndef PAGMO_DETAIL_EIGEN_HPP
#define PAGMO_DETAIL_EIGEN_HPP

// NOTE: we have experimental evidence that on some platform/compiler combinations
// Eigen is failing to include necessary header files. As a workaround, we use this header
// to wrap any Eigen functionality that might be needed in pagmo, and we pre-emptively
// include the missing headers as necessary.
#if defined(__apple_build_version__)

// NOTE: on OSX and if the _POSIX_C_SOURCE definition is active (or at least for some specific
// values of this definition), Eigen uses the alloca() function without including the header
// that declares it.
#include <alloca.h>

#endif

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#endif
