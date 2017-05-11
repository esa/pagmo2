/* Copyright 2017 PaGMO development team

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

#ifndef PAGMO_BASE_BGL_TOPOLOGY_HPP
#define PAGMO_BASE_BGL_TOPOLOGY_HPP

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Base BGL topology.
/**
 * \verbatim embed:rst:leading-asterisk
 * .. versionadded:: 2.3
 * \endverbatim
 *
 *
 */
class base_bgl_topology
{
    // NOTE: the definition of the graph type is taken from pagmo 1. We might
    // want to consider alternative storage classes down the line, as the complexity
    // of some graph operations is not that great when using vecs and lists.
    using graph_t
        = boost::adjacency_list<boost::vecS,           // std::vector for list of adjacent vertices (OutEdgeList)
                                boost::vecS,           // std::vector for the list of vertices (VertexList)
                                boost::bidirectionalS, // we require bi-directional edges for topology (Directed)
                                boost::no_property,    // no vertex properties (VertexProperties)
                                double,                // edge property stores migration probability (EdgeProperties)
                                boost::no_property,    // no graph properties (GraphProperties)
                                boost::listS           // std::list for of the graph's edge list (EdgeList)
                                >;
    // Small helper function that checks that the input vertices are in the graph.
    // It will throw otherwise.
    void check_vertex_indices() const
    {
    }
    template <typename... Args>
    void check_vertex_indices(std::size_t idx, Args... others) const
    {
        if (idx >= num_vertices()) {
            pagmo_throw(std::invalid_argument,
                        "invalid vertex index in a BGL topology: the index is " + std::to_string(idx)
                            + ", but the number of vertices is only " + std::to_string(num_vertices()));
        }
        check_vertex_indices(others...);
    }
    // Helper function to check that a weight is a finite number in the [0.,1.] range.
    static void check_weight(double w)
    {
        if (!std::isfinite(w)) {
            pagmo_throw(std::invalid_argument,
                        "invalid weight for the edge of a BGL topology: the value " + std::to_string(w)
                            + " is not finite");
        }
        if (w < 0. || w > 1.) {
            pagmo_throw(std::invalid_argument,
                        "invalid weight for the edge of a BGL topology: the value " + std::to_string(w)
                            + " is not in the [0.,1.] range");
        }
    }
    // Small helpers to reduce typing when converting to/from std::size_t.
    template <typename I>
    static std::size_t scast(I n)
    {
        return boost::numeric_cast<std::size_t>(n);
    }
    static graph_t::vertices_size_type vcast(std::size_t n)
    {
        return boost::numeric_cast<graph_t::vertices_size_type>(n);
    }

public:
    void add_vertex()
    {
        boost::add_vertex(m_graph);
    }
    std::size_t num_vertices() const
    {
        return scast(boost::num_vertices(m_graph));
    }
    bool are_adjacent(std::size_t i, std::size_t j) const
    {
        check_vertex_indices(i, j);
        const auto a_vertices = boost::adjacent_vertices(boost::vertex(vcast(i), m_graph), m_graph);
        return std::find(a_vertices.first, a_vertices.second, boost::vertex(vcast(j), m_graph)) != a_vertices.second;
    }
    void add_edge(std::size_t i, std::size_t j, double w = 1.)
    {
        if (are_adjacent(i, j)) {
            pagmo_throw(std::invalid_argument,
                        "cannot add edge in a BGL topology: there is already an edge connecting " + std::to_string(i)
                            + " to " + std::to_string(j));
        }
        check_weight(w);
        const auto result
            = boost::add_edge(boost::vertex(vcast(i), m_graph), boost::vertex(vcast(j), m_graph), m_graph);
        assert(result.second);
        m_graph[result.first] = w;
    }
    void remove_edge(std::size_t i, std::size_t j)
    {
        if (!are_adjacent(i, j)) {
            pagmo_throw(std::invalid_argument,
                        "cannot remove edge in a BGL topology: there is no edge connecting " + std::to_string(i)
                            + " to " + std::to_string(j));
        }
        boost::remove_edge(boost::vertex(vcast(i), m_graph), boost::vertex(vcast(j), m_graph), m_graph);
    }
    void set_weight(double w)
    {
        check_weight(w);
        for (auto e_range = boost::edges(m_graph); e_range.first != e_range.second; ++e_range.first) {
            m_graph[*e_range.first] = w;
        }
    }
    void set_weight(std::size_t i, std::size_t j, double w)
    {
        check_vertex_indices(i, j);
        check_weight(w);
        const auto ret = boost::edge(boost::vertex(vcast(i), m_graph), boost::vertex(vcast(j), m_graph), m_graph);
        if (ret.second) {
            m_graph[ret.first] = w;
        } else {
            pagmo_throw(std::invalid_argument,
                        "cannot set the weight of an edge in a BGL topology: the vertex " + std::to_string(i)
                            + " is not connected to vertex " + std::to_string(j));
        }
    }
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t i) const
    {
        check_vertex_indices(i);
        std::pair<std::vector<std::size_t>, vector_double> retval;
        const auto vi = boost::vertex(vcast(i), m_graph);
        for (auto iav = boost::inv_adjacent_vertices(vi, m_graph); iav.first != iav.second; ++iav.first) {
            const auto e = boost::edge(boost::vertex(*iav.first, m_graph), vi, m_graph);
            assert(e.second);
            retval.first.emplace_back(scast(*iav.first));
            retval.second.emplace_back(m_graph[e.first]);
        }
        return retval;
    }

private:
    graph_t m_graph;
};
}

#endif
