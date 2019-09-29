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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_io.hpp>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/topologies/base_bgl_topology.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

namespace
{

// Small helpers to reduce typing when converting to/from std::size_t.
template <typename I>
std::size_t scast(I n)
{
    return boost::numeric_cast<std::size_t>(n);
}

bgl_topology_graph_t::vertices_size_type vcast(std::size_t n)
{
    return boost::numeric_cast<bgl_topology_graph_t::vertices_size_type>(n);
}

} // namespace

} // namespace detail

// Small helper function that checks that the input vertices are in the graph.
// It will throw otherwise.
void base_bgl_topology::unsafe_check_vertex_indices() const {}

template <typename... Args>
void base_bgl_topology::unsafe_check_vertex_indices(std::size_t idx, Args... others) const
{
    const auto nv = boost::num_vertices(m_graph);
    if (idx >= nv) {
        pagmo_throw(std::invalid_argument, "invalid vertex index in a BGL topology: the index is " + std::to_string(idx)
                                               + ", but the number of vertices is only " + std::to_string(nv));
    }
    unsafe_check_vertex_indices(others...);
}

base_bgl_topology::graph_t base_bgl_topology::get_graph() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_graph;
}

base_bgl_topology::graph_t base_bgl_topology::move_graph()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return std::move(m_graph);
}

void base_bgl_topology::set_graph(graph_t &&g)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_graph = std::move(g);
}

base_bgl_topology::base_bgl_topology(const base_bgl_topology &other) : m_graph(other.get_graph()) {}

base_bgl_topology::base_bgl_topology(base_bgl_topology &&other) noexcept : m_graph(other.move_graph()) {}

base_bgl_topology &base_bgl_topology::operator=(const base_bgl_topology &other)
{
    if (this != &other) {
        set_graph(other.get_graph());
    }
    return *this;
}

base_bgl_topology &base_bgl_topology::operator=(base_bgl_topology &&other) noexcept
{
    if (this != &other) {
        set_graph(other.move_graph());
    }
    return *this;
}

void base_bgl_topology::add_vertex()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    boost::add_vertex(m_graph);
}

std::size_t base_bgl_topology::num_vertices() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return detail::scast(boost::num_vertices(m_graph));
}

bool base_bgl_topology::unsafe_are_adjacent(std::size_t i, std::size_t j) const
{
    unsafe_check_vertex_indices(i, j);
    const auto a_vertices = boost::adjacent_vertices(boost::vertex(detail::vcast(i), m_graph), m_graph);
    return std::find(a_vertices.first, a_vertices.second, boost::vertex(detail::vcast(j), m_graph))
           != a_vertices.second;
}

bool base_bgl_topology::are_adjacent(std::size_t i, std::size_t j) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return unsafe_are_adjacent(i, j);
}

void base_bgl_topology::add_edge(std::size_t i, std::size_t j, double w)
{
    detail::topology_check_edge_weight(w);

    std::lock_guard<std::mutex> lock(m_mutex);

    if (unsafe_are_adjacent(i, j)) {
        pagmo_throw(std::invalid_argument, "cannot add an edge in a BGL topology: there is already an edge connecting "
                                               + std::to_string(i) + " to " + std::to_string(j));
    }

    const auto result
        = boost::add_edge(boost::vertex(detail::vcast(i), m_graph), boost::vertex(detail::vcast(j), m_graph), m_graph);
    assert(result.second);
    m_graph[result.first] = w;
}

void base_bgl_topology::remove_edge(std::size_t i, std::size_t j)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!unsafe_are_adjacent(i, j)) {
        pagmo_throw(std::invalid_argument, "cannot remove an edge in a BGL topology: there is no edge connecting "
                                               + std::to_string(i) + " to " + std::to_string(j));
    }
    boost::remove_edge(boost::vertex(detail::vcast(i), m_graph), boost::vertex(detail::vcast(j), m_graph), m_graph);
}

void base_bgl_topology::set_all_weights(double w)
{
    detail::topology_check_edge_weight(w);

    std::lock_guard<std::mutex> lock(m_mutex);

    for (auto e_range = boost::edges(m_graph); e_range.first != e_range.second; ++e_range.first) {
        m_graph[*e_range.first] = w;
    }
}

void base_bgl_topology::set_weight(std::size_t i, std::size_t j, double w)
{
    detail::topology_check_edge_weight(w);

    std::lock_guard<std::mutex> lock(m_mutex);

    unsafe_check_vertex_indices(i, j);

    const auto ret
        = boost::edge(boost::vertex(detail::vcast(i), m_graph), boost::vertex(detail::vcast(j), m_graph), m_graph);
    if (ret.second) {
        m_graph[ret.first] = w;
    } else {
        pagmo_throw(std::invalid_argument, "cannot set the weight of an edge in a BGL topology: the vertex "
                                               + std::to_string(i) + " is not connected to vertex "
                                               + std::to_string(j));
    }
}

std::pair<std::vector<std::size_t>, vector_double> base_bgl_topology::get_connections(std::size_t i) const
{
    std::lock_guard<std::mutex> lock(m_mutex);

    unsafe_check_vertex_indices(i);

    std::pair<std::vector<std::size_t>, vector_double> retval;

    const auto vi = boost::vertex(detail::vcast(i), m_graph);
    for (auto iav = boost::inv_adjacent_vertices(vi, m_graph); iav.first != iav.second; ++iav.first) {
        const auto e = boost::edge(boost::vertex(*iav.first, m_graph), vi, m_graph);
        assert(e.second);
        retval.first.emplace_back(detail::scast(*iav.first));
        retval.second.emplace_back(m_graph[e.first]);
    }
    return retval;
}

std::string base_bgl_topology::get_extra_info() const
{
    std::ostringstream oss;

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        oss << "\tNumber of vertices: " << boost::num_vertices(m_graph) << '\n';
        oss << "\tNumber of edges: " << boost::num_edges(m_graph) << '\n';
        oss << "\tAdjacency list:\n\n";

        for (auto vs = boost::vertices(m_graph); vs.first != vs.second; ++vs.first) {
            // Get the list of outgoing edges from the current vertex.
            const auto erange = boost::out_edges(*vs.first, m_graph);

            // Helper to extract the target vertex from an edge descriptor (that is,
            // the vertex a directed edge points to).
            auto target_getter = [this](decltype(*erange.first) ed) { return boost::target(ed, m_graph); };

            // Helper to extract the edge weight from an edge descriptor.
            auto weight_getter = [this](decltype(*erange.first) ed) { return m_graph[ed]; };

            // Make zip iterators for bundling together the target of an edge
            // and its weight.
            auto z_begin = boost::make_zip_iterator(
                boost::make_tuple(boost::make_transform_iterator(erange.first, target_getter),
                                  boost::make_transform_iterator(erange.first, weight_getter)));
            auto z_end = boost::make_zip_iterator(
                boost::make_tuple(boost::make_transform_iterator(erange.second, target_getter),
                                  boost::make_transform_iterator(erange.second, weight_getter)));

            // Print the vertex, and its adjacent vertices together with the edges' weights.
            oss << "\t\t" << *vs.first << ": ";
            detail::stream_range(oss, z_begin, z_end);
            oss << '\n';
        }
    }

    return oss.str();
}

} // namespace pagmo
