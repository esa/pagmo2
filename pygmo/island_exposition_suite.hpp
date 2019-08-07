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

#ifndef PYGMO_ISLAND_EXPOSITION_SUITE_HPP
#define PYGMO_ISLAND_EXPOSITION_SUITE_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <cstdint>
#include <memory>

#include <pagmo/algorithm.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s_policy.hpp>

#include <pygmo/common_utils.hpp>

namespace pygmo
{

namespace bp = boost::python;

// Main island exposition function for use by APs.
template <typename Isl>
inline bp::class_<Isl> expose_island(const char *name, const char *descr)
{
    // We require all islands to be def-ctible at the bare minimum.
    bp::class_<Isl> c(name, descr, bp::init<>());

    // Mark it as a C++ island.
    c.attr("_pygmo_cpp_island") = true;

    // Get the island class from the pygmo module.
    auto &isl = **reinterpret_cast<std::unique_ptr<bp::class_<pagmo::island>> *>(
        bp::extract<std::uintptr_t>(bp::import("pygmo").attr("core").attr("_island_address"))());

    // Expose the island constructor from Isl.
    isl.def(bp::init<const Isl &, const pagmo::algorithm &, const pagmo::population &, const pagmo::r_policy &,
                     const pagmo::s_policy &>());

    // Expose extract.
    isl.def("_cpp_extract", &generic_cpp_extract<pagmo::island, Isl>, bp::return_internal_reference<>());

    return c;
}
} // namespace pygmo

#endif
