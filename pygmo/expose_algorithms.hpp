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

#ifndef PYGMO_EXPOSE_ALGORITHMS_HPP
#define PYGMO_EXPOSE_ALGORITHMS_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/algorithm.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

// Split algorithm exposition functions.
void expose_algorithms_0();
void expose_algorithms_1();

// A couple of utilities useful in the implementation
// of expose_algorithms_n().
namespace bp = boost::python;

// Main algorithm exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename Algo>
inline bp::class_<Algo> expose_algorithm_pygmo(const char *name, const char *descr)
{
    // We require all algorithms to be def-ctible at the bare minimum.
    bp::class_<Algo> c(name, descr, bp::init<>());

    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Get reference to the algorithm class.
    auto &algo = get_algorithm_class();

    // Expose the algorithm constructor from Algo.
    algo.def(bp::init<const Algo &>((bp::arg("uda"))));

    // Expose extract.
    algo.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>, bp::return_internal_reference<>());

    // Add the algorithm to the algorithms submodule.
    bp::scope().attr("algorithms").attr(name) = c;

    return c;
}
} // namespace pygmo

#endif
