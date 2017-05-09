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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/type_traits.hpp>

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

/// Detect \p get_inv_adjacent_vertices() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * std::vector<std::size_t> get_inv_adjacent_vertices(std::size_t) const;
 * @endcode
 * The \p get_inv_adjacent_vertices() method is part of the interface for the definition of a topology
 * (see pagmo::topology).
 */
template <typename T>
class has_get_inv_adjacent_vertices
{
    template <typename U>
    using get_inv_adjacent_vertices_t = decltype(std::declval<const U &>().get_inv_adjacent_vertices(std::size_t(0)));
    static const bool implementation_defined
        = std::is_same<std::vector<std::size_t>, detected_t<get_inv_adjacent_vertices_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_inv_adjacent_vertices<T>::value;

/// Detect \p push_back() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * void push_back();
 * @endcode
 * The \p push_back() method is part of the interface for the definition of a topology
 * (see pagmo::topology).
 */
template <typename T>
class has_push_back
{
    template <typename U>
    using push_back_t = decltype(std::declval<U &>().push_back());
    static const bool implementation_defined = std::is_same<void, detected_t<push_back_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_push_back<T>::value;

/// Detect \p get_weight() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * double get_weight(std::size_t, std::size_t) const;
 * @endcode
 * The \p get_weight() method is part of the interface for the definition of a topology
 * (see pagmo::topology).
 */
template <typename T>
class has_get_weight
{
    template <typename U>
    using get_weight_t = decltype(std::declval<const U &>().get_weight(std::size_t(0), std::size_t(0)));
    static const bool implementation_defined = std::is_same<double, detected_t<get_weight_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_weight<T>::value;

namespace detail
{

struct topo_inner_base {
    virtual ~topo_inner_base()
    {
    }
    virtual std::unique_ptr<topo_inner_base> clone() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual std::vector<std::size_t> get_inv_adjacent_vertices(std::size_t) const = 0;
    virtual void push_back() = 0;
    virtual double get_weight(std::size_t, std::size_t) const = 0;
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

/// Unconnected topology.
/**
 * This user-defined topology (UDT) represents an unconnected graph.
 */
struct unconnected {
    /// Get the list of inversely adjacent vertices.
    /**
     * @return an empty vector.
     */
    std::vector<std::size_t> get_inv_adjacent_vertices(std::size_t) const
    {
        return std::vector<std::size_t>{};
    }
    /// Add the next vertex.
    void push_back()
    {
    }
    /// Get the weight of an edge.
    /**
     * @return nothing, this method never returns.
     *
     * @throws std::invalid_argument if invoked.
     */
    double get_weight(std::size_t, std::size_t) const
    {
        pagmo_throw(std::invalid_argument, "An unconnected topology has no connections and thus no weights.");
    }
};

/// Topology.
/**
 * \image html migration_no_text.png
 *
 * In the jargon of pagmo, a topology is an object that represents connections among \link pagmo::island islands\endlink
 * in an \link pagmo::archipelago archipelago\endlink. In essence, a topology is a *weighted directed graph* in which
 *
 * * the *vertices* (or *nodes*) are islands,
 * * the *edges* (or *arcs*) are directed connections between islands across which information flows during the
 *   optimisation process (via the migration of individuals between islands),
 * * the *weights* of the edges (whose numerical values are the \f$ [0.,1.] \f$ range) represent the migration
 *   probability.
 *
 * Following the same schema adopted for pagmo::problem, pagmo::algorithm, etc., pagmo::topology exposes a generic
 * interface to **user-defined topologies** (or UDT for short). UDTs are classes (or struct) exposing a certain set
 * of methods that describe the properties of a topology (e.g., the number of vertices, the list of edges, etc.). Once
 * defined and instantiated, a UDT can then be used to construct an instance of this class, pagmo::topology, which
 * provides a generic interface to topologies for use by pagmo::archipelago and pagmo::island.
 */
class topology
{
};
}

#endif
