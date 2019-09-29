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

#include <atomic>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/exceptions.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

// Default constructor: weight of 1, zero vertices.
fully_connected::fully_connected() : fully_connected(1.) {}

// Ctor from edge weight, zero vertices.
fully_connected::fully_connected(double w) : fully_connected(0, w) {}

// Ctor from number of vertices and edge weight
fully_connected::fully_connected(std::size_t n, double w) : m_weight(w), m_num_vertices(n)
{
    detail::topology_check_edge_weight(m_weight);
}

// Identical copy/move constructors.
fully_connected::fully_connected(const fully_connected &other)
    : m_weight(other.m_weight), m_num_vertices(other.m_num_vertices.load(std::memory_order_relaxed))
{
}

fully_connected::fully_connected(fully_connected &&other) noexcept
    : fully_connected(static_cast<const fully_connected &>(other))
{
}

// Push back implementation.
void fully_connected::push_back()
{
    m_num_vertices.fetch_add(1u, std::memory_order_relaxed);
}

// Get connections.
std::pair<std::vector<std::size_t>, vector_double> fully_connected::get_connections(std::size_t i) const
{
    // Fetch the number of vertices.
    const auto num_vertices = m_num_vertices.load(std::memory_order_relaxed);

    if (i >= num_vertices) {
        pagmo_throw(std::invalid_argument,
                    "Cannot get the connections to the vertex at index " + std::to_string(i)
                        + " in a fully connected topology: the number of vertices in the topology is only "
                        + std::to_string(num_vertices));
    }

    // Init the retval.
    std::pair<std::vector<std::size_t>, vector_double> retval;

    // Prepare storage for the indices list.
    retval.first.resize(boost::numeric_cast<decltype(retval.first.size())>(num_vertices - 1u));

    // Fill in the indices list.
    for (std::size_t j = 0; j < i; ++j) {
        retval.first[j] = j;
    }
    for (std::size_t j = i + 1u; j < num_vertices; ++j) {
        retval.first[j - 1u] = j;
    }

    // Fill the weights list with m_weight.
    retval.second.resize(boost::numeric_cast<decltype(retval.second.size())>(num_vertices - 1u), m_weight);

    return retval;
}

// Topology name.
std::string fully_connected::get_name() const
{
    return "Fully connected";
}

// Topology extra info.
std::string fully_connected::get_extra_info() const
{
    return "\tNumber of vertices: " + std::to_string(m_num_vertices.load(std::memory_order_relaxed))
           + "\n\tEdges' weight: " + std::to_string(m_weight) + "\n";
}

// Get the edge weight.
double fully_connected::get_weight() const
{
    return m_weight;
}

// Get the number of vertices.
std::size_t fully_connected::num_vertices() const
{
    return m_num_vertices.load(std::memory_order_relaxed);
}

// Serialization.
template <typename Archive>
void fully_connected::save(Archive &ar, unsigned) const
{
    detail::archive(ar, m_weight, m_num_vertices.load(std::memory_order_relaxed));
}

template <typename Archive>
void fully_connected::load(Archive &ar, unsigned)
{
    std::size_t num_vertices;

    ar >> m_weight;
    ar >> num_vertices;

    m_num_vertices.store(num_vertices, std::memory_order_relaxed);
}

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_IMPLEMENT(pagmo::fully_connected)
