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

#include <cassert>
#include <cstddef>
#include <string>

#include <pagmo/s11n.hpp>
#include <pagmo/topologies/base_bgl_topology.hpp>
#include <pagmo/topologies/ring.hpp>
#include <pagmo/topology.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

// Default ctor.
ring::ring() : m_weight(1) {}

// Ctor from edge weight.
ring::ring(double w) : m_weight(w)
{
    detail::topology_check_edge_weight(m_weight);
}

// Ctor from number of vertices and edge weight.
ring::ring(std::size_t n, double w) : m_weight(w)
{
    detail::topology_check_edge_weight(m_weight);

    for (std::size_t i = 0; i < n; ++i) {
        push_back();
    }
}

// Serialization.
template <typename Archive>
void ring::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, boost::serialization::base_object<base_bgl_topology>(*this), m_weight);
}

// Add vertex.
void ring::push_back()
{
    // Add the new vertex.
    add_vertex();

    // Connect it.
    const auto size = num_vertices();
    assert(size);

    switch (size) {
        case 1u: {
            // If the topology was empty, no connections need to be established.
            break;
        }
        case 2u: {
            // The two elements connect each other.
            add_edge(0, 1, m_weight);
            add_edge(1, 0, m_weight);
            break;
        }
        case 3u: {
            // A triangle of double links.
            add_edge(1, 2, m_weight);
            add_edge(2, 1, m_weight);
            add_edge(2, 0, m_weight);
            add_edge(0, 2, m_weight);
            break;
        }
        default: {
            // Remove the pair of links that close the current ring.
            remove_edge(size - 2u, 0);
            remove_edge(0, size - 2u);
            // Connect the new "last" vertex to the previous "last" vertex.
            add_edge(size - 2u, size - 1u, m_weight);
            add_edge(size - 1u, size - 2u, m_weight);
            // Connect the new last vertex to the first.
            add_edge(0, size - 1u, m_weight);
            add_edge(size - 1u, 0, m_weight);
        }
    }
}

// Get the edge weight used for construction.
double ring::get_weight() const
{
    return m_weight;
}

// Topology name.
std::string ring::get_name() const
{
    return "Ring";
}

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_IMPLEMENT(pagmo::ring)
