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

#ifndef PYGMO_ISLAND_EXPOSITION_SUITE_HPP
#define PYGMO_ISLAND_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/scope.hpp>
#include <cassert>
#include <pagmo/algorithm.hpp>

#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

// Expose an island ctor from a C++ UDI.
template <typename Isl>
inline void island_expose_init_cpp_udi()
{
    assert(island_ptr.get() != nullptr);
    auto &isl_class = *island_ptr;
    isl_class.def(
        bp::init<const Isl &, const pagmo::problem &, const pagmo::algorithm &, pagmo::population::size_type>());
    isl_class.def(bp::init<const Isl &, const pagmo::problem &, const pagmo::algorithm &, pagmo::population::size_type,
                           unsigned>());
}

// Main island exposition function.
template <typename Isl>
inline bp::class_<Isl> expose_island(const char *name, const char *descr)
{
    // We require all islands to be def-ctible at the bare minimum.
    bp::class_<Isl> c(name, descr, bp::init<>());
    // Mark it as a C++ island.
    c.attr("_pygmo_cpp_island") = true;

    // Expose the island constructor from Isl.
    island_expose_init_cpp_udi<Isl>();

    // Add the island to the islands submodule.
    bp::scope().attr("islands").attr(name) = c;

    return c;
}
}

#endif
