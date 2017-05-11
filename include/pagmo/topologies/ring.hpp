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

#ifndef PAGMO_RING_HPP
#define PAGMO_RING_HPP

#include <cassert>
#include <string>

#include <pagmo/topologies/base_bgl_topology.hpp>
#include <pagmo/topology.hpp>

namespace pagmo
{

struct ring : base_bgl_topology {
    void push_back()
    {
        add_vertex();
        const auto size = num_vertices();
        assert(size);
        switch (size) {
            case 1u: {
                // If the topology was empty, no connections need to be established.
                break;
            }
            case 2u: {
                // The two elements connect each other.
                add_edge(0, 1);
                add_edge(1, 0);
                break;
            }
            case 3u: {
                // A triangle of double links.
                add_edge(1, 2);
                add_edge(2, 1);
                add_edge(2, 0);
                add_edge(0, 2);
                break;
            }
            default: {
                // Remove the pair of links that close the current ring.
                remove_edge(size - 2u, 0);
                remove_edge(0, size - 2u);
                // Connect the new "last" vertex to the previous "last" vertex.
                add_edge(size - 2u, size - 1u);
                add_edge(size - 1u, size - 2u);
                // Connect the new last vertex to the first.
                add_edge(0, size - 1u);
                add_edge(size - 1u, 0);
            }
        }
    }
    std::string get_name() const
    {
        return "Ring";
    }
};
}

#endif
