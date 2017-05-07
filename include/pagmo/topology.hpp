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

#ifndef PAGMO_TOPOLOGY_HPP
#define PAGMO_TOPOLOGY_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/serialization.hpp>

/// Macro for the registration of the serialization functionality for user-defined topologies.
/**
 * This macro should always be invoked after the declaration of a user-defined topology: it will register
 * the topology with pagmo's serialization machinery. The macro should be called in the root namespace
 * and using the fully qualified name of the topology to be registered. For example:
 * @code{.unparsed}
 * namespace my_namespace
 * {
 *
 * class my_topology
 * {
 *    // ...
 * };
 *
 * }
 *
 * PAGMO_REGISTER_TOPOLOGY(my_namespace::my_topology)
 * @endcode
 */
#define PAGMO_REGISTER_TOPOLOGY(topo) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::topo_inner<topo>, "udt " #topo)

namespace pagmo
{

namespace detail
{

struct topo_inner_base {
    virtual ~topo_inner_base()
    {
    }
    virtual std::unique_ptr<topo_inner_base> clone() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual std::size_t get_nvertices() const = 0;
    virtual std::size_t get_nedges() const = 0;
    virtual void push_back() = 0;
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

template <typename T>
struct topo_inner final : topo_inner_base {
    // We just need the def ctor, delete everything else.
    topo_inner() = default;
    topo_inner(const topo_inner &) = delete;
    topo_inner(topo_inner &&) = delete;
    topo_inner &operator=(const topo_inner &) = delete;
    topo_inner &operator=(topo_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit topo_inner(const T &x) : m_value(x)
    {
    }
    explicit topo_inner(T &&x) : m_value(std::move(x))
    {
    }
    // The clone method, used in the copy constructor of topology.
    virtual std::unique_ptr<topo_inner_base> clone() const override final
    {
        return make_unique<topo_inner>(m_value);
    }
    // Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<topo_inner_base>(this), m_value);
    }
    T m_value;
};
}
}

#endif
