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

#ifndef PYGMO_EXPOSE_R_POLICIES_HPP
#define PYGMO_EXPOSE_R_POLICIES_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/r_policy.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

// Replacement policies exposition function.
void expose_r_policies();

namespace bp = boost::python;

// Main r_policy exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename RPol>
inline bp::class_<RPol> expose_r_policy_pygmo(const char *name, const char *descr)
{
    // We require all replacement policies to be def-ctible at the bare minimum.
    bp::class_<RPol> c(name, descr, bp::init<>());

    // Mark it as a C++ replacement policy.
    c.attr("_pygmo_cpp_r_policy") = true;

    // Get reference to the r_policy class.
    auto &t = get_r_policy_class();

    // Expose the r_policy constructor from RPol.
    t.def(bp::init<const RPol &>((bp::arg("udrp"))));

    // Expose extract.
    t.def("_cpp_extract", &generic_cpp_extract<pagmo::r_policy, RPol>, bp::return_internal_reference<>());

    // Add the r_policy to the replacement policies submodule.
    bp::scope().attr("r_policies").attr(name) = c;

    return c;
}
} // namespace pygmo

#endif
