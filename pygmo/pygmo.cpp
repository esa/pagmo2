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

#if defined(_MSC_VER)

// Disable various warnings from MSVC.
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)
#pragma warning(disable : 4244)

#endif

#include <boost/python/class.hpp>
#include <memory>

#define PAGMO_SKIP_SERIALIZATION

#include <pagmo/algorithm.hpp>
#include <pagmo/island.hpp>
#include <pagmo/problem.hpp>

#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

PYGMO_DLL_PUBLIC std::unique_ptr<bp::class_<pagmo::problem>> problem_ptr{};

PYGMO_DLL_PUBLIC std::unique_ptr<bp::class_<pagmo::algorithm>> algorithm_ptr{};

PYGMO_DLL_PUBLIC std::unique_ptr<bp::class_<pagmo::island>> island_ptr{};
}
