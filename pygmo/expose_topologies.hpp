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

#ifndef PYGMO_EXPOSE_TOPOLOGIES_HPP
#define PYGMO_EXPOSE_TOPOLOGIES_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>

#include <pagmo/topology.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/pygmo_classes.hpp>

namespace pygmo
{

// Topologies exposition function.
void expose_topologies();

namespace bp = boost::python;

// Main topology exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename Topo>
inline bp::class_<Topo> expose_topology_pygmo(const char *name, const char *descr)
{
    // We require all topologies to be def-ctible at the bare minimum.
    bp::class_<Topo> c(name, descr, bp::init<>());

    // Mark it as a C++ topology.
    c.attr("_pygmo_cpp_topology") = true;

    // Get reference to the topology class.
    auto &t = get_topology_class();

    // Expose the topology constructor from Topo.
    t.def(bp::init<const Topo &>((bp::arg("udt"))));

    // Expose extract.
    t.def("_cpp_extract", &generic_cpp_extract<pagmo::topology, Topo>, bp::return_internal_reference<>());

    // Add the topology to the topologies submodule.
    bp::scope().attr("topologies").attr(name) = c;

    return c;
}
} // namespace pygmo

#endif
