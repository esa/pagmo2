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

#ifndef PYGMO_EXPOSE_ISLANDS_HPP
#define PYGMO_EXPOSE_ISLANDS_HPP

#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

namespace bp = boost::python;

void expose_islands();

// Main island exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename Isl>
inline bp::class_<Isl> expose_island_pygmo(const char *name, const char *descr)
{
    // We require all islands to be def-ctible at the bare minimum.
    bp::class_<Isl> c(name, descr, bp::init<>());

    // Mark it as a C++ island.
    c.attr("_pygmo_cpp_island") = true;

    // Get reference to the island class.
    auto &isl = get_island_class();

    // Expose the island constructor from Isl.
    isl.def(bp::init<const Isl &, const pagmo::algorithm &, const pagmo::population &>());

    // Expose extract.
    isl.def("_cpp_extract", &generic_cpp_extract<pagmo::island, Isl>, bp::return_internal_reference<>());

    // Add the island to the islands submodule.
    bp::scope().attr("islands").attr(name) = c;

    return c;
}

} // namespace pygmo

#endif
