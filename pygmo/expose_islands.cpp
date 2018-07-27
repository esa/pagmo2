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

#if defined(_MSC_VER)

// Disable various warnings from MSVC.
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)
#pragma warning(disable : 4503)
#pragma warning(disable : 4244)

#endif

#include <pygmo/python_includes.hpp>

// See: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// In every cpp file we need to make sure this is included before everything else,
// with the correct #defines.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include <pygmo/numpy.hpp>

#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>

#include <pygmo/docstrings.hpp>
#include <pygmo/pygmo_classes.hpp>

using namespace pagmo;
namespace bp = boost::python;

namespace pygmo
{

// Main island exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename Isl>
static inline bp::class_<Isl> expose_island_pygmo(const char *name, const char *descr)
{
    // We require all islands to be def-ctible at the bare minimum.
    bp::class_<Isl> c(name, descr, bp::init<>());

    // Mark it as a C++ island.
    c.attr("_pygmo_cpp_island") = true;

    // Expose the island constructor from Isl.
    get_island_class().def(bp::init<const Isl &, const pagmo::algorithm &, const pagmo::population &>());

    // Add the island to the islands submodule.
    bp::scope().attr("islands").attr(name) = c;

    return c;
}

void expose_islands()
{
    // Thread island.
    expose_island_pygmo<thread_island>("thread_island", thread_island_docstring().c_str());
}
} // namespace pygmo
