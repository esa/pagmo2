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

#ifndef PAGMO_TOPOLOGIES_FULLY_CONNECTED_HPP
#define PAGMO_TOPOLOGIES_FULLY_CONNECTED_HPP

#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Fully connected topology.
class PAGMO_DLL_PUBLIC fully_connected
{
public:
    fully_connected();
    explicit fully_connected(double);
    explicit fully_connected(std::size_t, double);
    fully_connected(const fully_connected &);
    fully_connected(fully_connected &&) noexcept;

    void push_back();
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;

    std::string get_name() const;
    std::string get_extra_info() const;

    double get_weight() const;
    std::size_t num_vertices() const;

    template <typename Archive>
    void save(Archive &, unsigned) const;
    template <typename Archive>
    void load(Archive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    double m_weight;
    std::atomic<std::size_t> m_num_vertices;
};

} // namespace pagmo

PAGMO_S11N_TOPOLOGY_EXPORT_KEY(pagmo::fully_connected)

#endif
