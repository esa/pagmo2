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

#ifndef PAGMO_TOPOLOGIES_BASE_BGL_TOPOLOGY_HPP
#define PAGMO_TOPOLOGIES_BASE_BGL_TOPOLOGY_HPP

#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <boost/graph/adjacency_list.hpp>

#if defined(_MSC_VER)

// Disable a warning from MSVC in the graph serialization code.
#pragma warning(push)
#pragma warning(disable : 4267)

#endif

#include <boost/graph/adj_list_serialize.hpp>

#if defined(_MSC_VER)

#pragma warning(pop)

#endif

#include <pagmo/detail/visibility.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

// NOTE: the definition of the graph type is taken from pagmo 1. We might
// want to consider alternative storage classes down the line, as the complexity
// of some graph operations is not that great when using vecs and lists.
using bgl_topology_graph_t
    = boost::adjacency_list<boost::vecS,           // std::vector for list of adjacent vertices (OutEdgeList)
                            boost::vecS,           // std::vector for the list of vertices (VertexList)
                            boost::bidirectionalS, // we require bi-directional edges for topology (Directed)
                            boost::no_property,    // no vertex properties (VertexProperties)
                            double,                // edge property stores migration probability (EdgeProperties)
                            boost::no_property,    // no graph properties (GraphProperties)
                            boost::listS           // std::list for of the graph's edge list (EdgeList)
                            >;

} // namespace detail

// Helper for the implementation of topologies
// based on the Boost Graph library.
class PAGMO_DLL_PUBLIC base_bgl_topology
{
    using graph_t = detail::bgl_topology_graph_t;

    // NOTE: all these methods do *not* lock the mutex,
    // hence they are marked as "unsafe". These should
    // be invoked only if the mutex is already being
    // held by the calling thread.
    //
    // Small helper function that checks that the input vertices are in the graph.
    // It will throw otherwise.
    PAGMO_DLL_LOCAL void unsafe_check_vertex_indices() const;
    template <typename... Args>
    PAGMO_DLL_LOCAL void unsafe_check_vertex_indices(std::size_t, Args...) const;
    // Helper to detect adjacent vertices.
    PAGMO_DLL_LOCAL bool unsafe_are_adjacent(std::size_t, std::size_t) const;

    // A few helpers to set/get the integral graph
    // object. These will lock the mutex, so they
    // are safe for general use.
    graph_t get_graph() const;
    PAGMO_DLL_LOCAL graph_t move_graph();
    PAGMO_DLL_LOCAL void set_graph(graph_t &&);

public:
    base_bgl_topology() = default;
    base_bgl_topology(const base_bgl_topology &);
    base_bgl_topology(base_bgl_topology &&) noexcept;
    base_bgl_topology &operator=(const base_bgl_topology &);
    base_bgl_topology &operator=(base_bgl_topology &&) noexcept;

    std::size_t num_vertices() const;
    bool are_adjacent(std::size_t, std::size_t) const;
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;

    void add_vertex();
    void add_edge(std::size_t, std::size_t, double = 1.);
    void remove_edge(std::size_t, std::size_t);
    void set_weight(std::size_t, std::size_t, double);
    void set_all_weights(double);

    std::string get_extra_info() const;

    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, get_graph());
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        base_bgl_topology tmp;
        // NOTE: no need to protect the
        // access to the graph of the
        // newly-constructed tmp topology.
        detail::from_archive(ar, tmp.m_graph);
        *this = std::move(tmp);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    mutable std::mutex m_mutex;
    graph_t m_graph;
};

} // namespace pagmo

// Disable tracking for the serialisation of base_bgl_topology.
BOOST_CLASS_TRACKING(pagmo::base_bgl_topology, boost::serialization::track_never)

#endif
