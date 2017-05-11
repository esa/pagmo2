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
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include <pagmo/exceptions.hpp>

namespace pagmo
{

class base_bgl_topology
{
    using graph_t
        = boost::adjacency_list<boost::vecS,           // std::vector for list of adjacent vertices (OutEdgeList)
                                boost::vecS,           // std::vector for the list of vertices (VertexList)
                                boost::bidirectionalS, // we require bi-directional edges for topology (Directed)
                                boost::no_property,    // no vertex properties (VertexProperties)
                                double,                // edge property stores migration probability (EdgeProperties)
                                boost::no_property,    // no graph properties (GraphProperties)
                                boost::listS           // std::list for of the graph's edge list (EdgeList)
                                >;

public:
    using size_type = graph_t::vertices_size_type;

private:
    void check_vertex_indices() const
    {
    }
    template <typename... Args>
    void check_vertex_indices(size_type idx, Args... others) const
    {
        if (idx >= num_vertices()) {
            pagmo_throw(std::invalid_argument,
                        "invalid vertex index in a BGL topology: the index is " + std::to_string(idx)
                            + ", but the number of vertices is only " + std::to_string(num_vertices()));
        }
        check_vertex_indices(others...);
    }
    static void check_weight(double w)
    {
        if (!std::isfinite(w)) {
            pagmo_throw(std::invalid_argument,
                        "invalid weight for the edge of a BGL topology: the value is not finite");
        }
        if (w < 0. || w > 1.) {
            pagmo_throw(std::invalid_argument,
                        "invalid weight for the edge of a BGL topology: the value " + std::to_string(w)
                            + " is not in the [0.,1.] range");
        }
    }

public:
    void add_vertex()
    {
        boost::add_vertex(m_graph);
    }
    size_type num_vertices() const
    {
        return boost::num_vertices(m_graph);
    }
    bool are_adjacent(size_type i, size_type j) const
    {
        check_vertex_indices(i, j);
        const auto a_vertices = boost::adjacent_vertices(boost::vertex(i, m_graph), m_graph);
        return std::find(a_vertices.first, a_vertices.second, boost::vertex(j, m_graph)) != a_vertices.second;
    }
    void add_edge(size_type i, size_type j, double weight = 1.)
    {
        if (are_adjacent(i, j)) {
            pagmo_throw(std::invalid_argument,
                        "cannot add edge: there is already an edge connecting " + std::to_string(i) + " to "
                            + std::to_string(j));
        }
        check_weight(weight);
        const auto result = boost::add_edge(boost::vertex(i, m_graph), boost::vertex(j, m_graph), m_graph);
        assert(result.second);
        m_graph[result.first] = weight;
    }

private:
    graph_t m_graph;
};
}

#endif
