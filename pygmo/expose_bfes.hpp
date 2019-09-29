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

#ifndef PYGMO_EXPOSE_BFES_HPP
#define PYGMO_EXPOSE_BFES_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/bfe.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

// Bfes exposition function.
void expose_bfes();

namespace bp = boost::python;

// Main bfe exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename Bfe>
inline bp::class_<Bfe> expose_bfe_pygmo(const char *name, const char *descr)
{
    // We require all bfes to be def-ctible at the bare minimum.
    bp::class_<Bfe> c(name, descr, bp::init<>());

    // Mark it as a C++ bfe.
    c.attr("_pygmo_cpp_bfe") = true;

    // Get reference to the bfe class.
    auto &b = get_bfe_class();

    // Expose the bfe constructor from Bfe.
    b.def(bp::init<const Bfe &>((bp::arg("udbfe"))));

    // Expose extract.
    b.def("_cpp_extract", &generic_cpp_extract<pagmo::bfe, Bfe>, bp::return_internal_reference<>());

    // Add the bfe to the batch_evaluators submodule.
    bp::scope().attr("batch_evaluators").attr(name) = c;

    return c;
}
} // namespace pygmo

#endif
