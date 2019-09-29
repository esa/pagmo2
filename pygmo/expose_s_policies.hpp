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

#ifndef PYGMO_EXPOSE_S_POLICIES_HPP
#define PYGMO_EXPOSE_S_POLICIES_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/s_policy.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

// Selection policies exposition function.
void expose_s_policies();

namespace bp = boost::python;

// Main s_policy exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename SPol>
inline bp::class_<SPol> expose_s_policy_pygmo(const char *name, const char *descr)
{
    // We require all selection policies to be def-ctible at the bare minimum.
    bp::class_<SPol> c(name, descr, bp::init<>());

    // Mark it as a C++ selection policy.
    c.attr("_pygmo_cpp_s_policy") = true;

    // Get reference to the s_policy class.
    auto &t = get_s_policy_class();

    // Expose the s_policy constructor from SPol.
    t.def(bp::init<const SPol &>((bp::arg("udsp"))));

    // Expose extract.
    t.def("_cpp_extract", &generic_cpp_extract<pagmo::s_policy, SPol>, bp::return_internal_reference<>());

    // Add the s_policy to the selection policies submodule.
    bp::scope().attr("s_policies").attr(name) = c;

    return c;
}
} // namespace pygmo

#endif
