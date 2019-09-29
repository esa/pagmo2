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

#include <cstddef>
#include <utility>
#include <vector>

#include <boost/python/args.hpp>
#include <boost/python/init.hpp>

#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/topologies/ring.hpp>
#include <pagmo/topologies/unconnected.hpp>
#include <pagmo/types.hpp>

#include <pygmo/docstrings.hpp>
#include <pygmo/expose_topologies.hpp>

using namespace pagmo;
namespace bp = boost::python;

namespace pygmo
{

namespace detail
{

namespace
{

// A test topology.
struct test_topology {
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const
    {
        return std::pair<std::vector<std::size_t>, vector_double>{};
    }
    void push_back() {}
    // Set/get an internal value to test extraction semantics.
    void set_n(int n)
    {
        m_n = n;
    }
    int get_n() const
    {
        return m_n;
    }
    int m_n = 1;
};

// Expose the methods from the base BGL topology.
template <typename Topo>
void expose_base_bgl_topo(bp::class_<Topo> &c)
{
    c.def("num_vertices", &Topo::num_vertices, pygmo::base_bgl_num_vertices_docstring().c_str());
    c.def("are_adjacent", &Topo::are_adjacent, pygmo::base_bgl_are_adjacent_docstring().c_str(),
          (bp::arg("i"), bp::arg("j")));
    c.def("add_vertex", &Topo::add_vertex, pygmo::base_bgl_add_vertex_docstring().c_str());
    c.def("add_edge", &Topo::add_edge, pygmo::base_bgl_add_edge_docstring().c_str(),
          (bp::arg("i"), bp::arg("j"), bp::arg("w") = 1.));
    c.def("remove_edge", &Topo::remove_edge, pygmo::base_bgl_remove_edge_docstring().c_str(),
          (bp::arg("i"), bp::arg("j")));
    c.def("set_weight", &Topo::set_weight, pygmo::base_bgl_set_weight_docstring().c_str(),
          (bp::arg("i"), bp::arg("j"), bp::arg("w")));
    c.def("set_all_weights", &Topo::set_all_weights, pygmo::base_bgl_set_all_weights_docstring().c_str(),
          (bp::arg("w")));
}

} // namespace

} // namespace detail

void expose_topologies()
{
    // Test topology.
    auto t_topology = expose_topology_pygmo<detail::test_topology>("_test_topology", "A test topology.");
    t_topology.def("get_n", &detail::test_topology::get_n);
    t_topology.def("set_n", &detail::test_topology::set_n);

    // Unconnected topology.
    expose_topology_pygmo<unconnected>("unconnected", unconnected_docstring().c_str());

    // Ring.
    auto ring_ = expose_topology_pygmo<ring>("ring", ring_docstring().c_str());
    ring_.def(bp::init<std::size_t, double>((bp::arg("n") = std::size_t(0), bp::arg("w") = 1.)))
        .def("get_weight", &ring::get_weight, pygmo::ring_get_weight_docstring().c_str());
    detail::expose_base_bgl_topo(ring_);

    // Fully connected.
    auto fully_connected_
        = expose_topology_pygmo<fully_connected>("fully_connected", fully_connected_docstring().c_str());
    fully_connected_.def(bp::init<std::size_t, double>((bp::arg("n") = std::size_t(0), bp::arg("w") = 1.)))
        .def("get_weight", &fully_connected::get_weight, pygmo::fully_connected_get_weight_docstring().c_str())
        .def("num_vertices", &fully_connected::num_vertices, pygmo::fully_connected_num_vertices_docstring().c_str());
}

} // namespace pygmo
