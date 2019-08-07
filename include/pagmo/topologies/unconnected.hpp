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

#ifndef PAGMO_TOPOLOGIES_UNCONNECTED_HPP
#define PAGMO_TOPOLOGIES_UNCONNECTED_HPP

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Unconnected topology.
struct PAGMO_DLL_PUBLIC unconnected {
    // Get the connections.
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;
    // Add the next vertex (no-op).
    void push_back() {}
    // Name.
    std::string get_name() const
    {
        return "Unconnected";
    }
    // Serialization.
    template <typename Archive>
    void serialize(Archive &, unsigned);
};

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_EXPORT_KEY(pagmo::unconnected)

#endif
