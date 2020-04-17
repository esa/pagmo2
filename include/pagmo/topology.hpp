/* Copyright 2017-2020 PaGMO development team

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

#include <cassert>
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>

#include <pagmo/config.hpp>
#include <pagmo/detail/support_xeus_cling.hpp>
#include <pagmo/detail/type_name.hpp>
#include <pagmo/detail/typeid_name_extract.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#define PAGMO_S11N_TOPOLOGY_EXPORT_KEY(topo)                                                                           \
    BOOST_CLASS_EXPORT_KEY2(pagmo::detail::topo_inner<topo>, "udt " #topo)                                             \
    BOOST_CLASS_TRACKING(pagmo::detail::topo_inner<topo>, boost::serialization::track_never)

#define PAGMO_S11N_TOPOLOGY_IMPLEMENT(topo) BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::detail::topo_inner<topo>)

#define PAGMO_S11N_TOPOLOGY_EXPORT(topo)                                                                               \
    PAGMO_S11N_TOPOLOGY_EXPORT_KEY(topo)                                                                               \
    PAGMO_S11N_TOPOLOGY_IMPLEMENT(topo)

namespace pagmo
{

// Detect the get_connections() method.
template <typename T>
class has_get_connections
{
    template <typename U>
    using get_connections_t = decltype(std::declval<const U &>().get_connections(std::size_t(0)));
    static const bool implementation_defined
        = std::is_same<std::pair<std::vector<std::size_t>, vector_double>, detected_t<get_connections_t, T>>::value;

public:
    // Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_connections<T>::value;

// Detect the push_back() method.
template <typename T>
class has_push_back
{
    template <typename U>
    using push_back_t = decltype(std::declval<U &>().push_back());
    static const bool implementation_defined = std::is_same<void, detected_t<push_back_t, T>>::value;

public:
    // Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_push_back<T>::value;

#if !defined(PAGMO_DOXYGEN_INVOKED)

// A Boost graph type which is used as an export format for topologies
// and also as the underyling graph type for base_bgl_topology.
// NOTE: the definition of the graph type is taken from pagmo 1. We might
// want to consider alternative storage classes down the line, as the complexity
// of some graph operations is not that great when using vecs and lists.
using bgl_graph_t
    = boost::adjacency_list<boost::vecS,           // std::vector for list of adjacent vertices (OutEdgeList)
                            boost::vecS,           // std::vector for the list of vertices (VertexList)
                            boost::bidirectionalS, // we require bi-directional edges for topology (Directed)
                            boost::no_property,    // no vertex properties (VertexProperties)
                            double,                // edge property stores migration probability (EdgeProperties)
                            boost::no_property,    // no graph properties (GraphProperties)
                            boost::listS           // std::list for of the graph's edge list (EdgeList)
                            >;

#endif

// Detect the to_bgl() method.
template <typename T>
class has_to_bgl
{
    template <typename U>
    using to_bgl_t = decltype(std::declval<const U &>().to_bgl());
    static const bool implementation_defined = std::is_same<bgl_graph_t, detected_t<to_bgl_t, T>>::value;

public:
    // Value of the type trait.
    static const bool value = implementation_defined;
};

namespace detail
{

// Specialise this to true in order to disable all the UDT checks and mark a type
// as a UDT regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python topos.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_udt_checks : std::false_type {
};

} // namespace detail

// Detect user-defined topologies (UDT).
template <typename T>
class is_udt
{
    static const bool implementation_defined
        = detail::disjunction<detail::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                                  std::is_copy_constructible<T>, std::is_move_constructible<T>,
                                                  std::is_destructible<T>, has_get_connections<T>, has_push_back<T>>,
                              detail::disable_udt_checks<T>>::value;

public:
    // Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udt<T>::value;

namespace detail
{

struct PAGMO_DLL_PUBLIC_INLINE_CLASS topo_inner_base {
    virtual ~topo_inner_base() {}
    virtual std::unique_ptr<topo_inner_base> clone() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const = 0;
    virtual void push_back() = 0;
    virtual bgl_graph_t to_bgl() const = 0;
    virtual std::type_index get_type_index() const = 0;
    virtual const void *get_ptr() const = 0;
    virtual void *get_ptr() = 0;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
struct PAGMO_DLL_PUBLIC_INLINE_CLASS topo_inner final : topo_inner_base {
    // We just need the def ctor, delete everything else.
    topo_inner() = default;
    topo_inner(const topo_inner &) = delete;
    topo_inner(topo_inner &&) = delete;
    topo_inner &operator=(const topo_inner &) = delete;
    topo_inner &operator=(topo_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit topo_inner(const T &x) : m_value(x) {}
    explicit topo_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of topology.
    std::unique_ptr<topo_inner_base> clone() const final
    {
        return std::make_unique<topo_inner>(m_value);
    }
    // The mandatory methods.
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t n) const final
    {
        return m_value.get_connections(n);
    }
    void push_back() final
    {
        m_value.push_back();
    }
    // Optional methods.
    bgl_graph_t to_bgl() const final
    {
        return to_bgl_impl(m_value);
    }
    std::string get_name() const final
    {
        return get_name_impl(m_value);
    }
    std::string get_extra_info() const final
    {
        return get_extra_info_impl(m_value);
    }
    // Implementation of the optional methods.
    template <typename U, enable_if_t<has_to_bgl<U>::value, int> = 0>
    static bgl_graph_t to_bgl_impl(const U &value)
    {
        return value.to_bgl();
    }
    template <typename U, enable_if_t<!has_to_bgl<U>::value, int> = 0>
    [[noreturn]] static bgl_graph_t to_bgl_impl(const U &value)
    {
        pagmo_throw(not_implemented_error,
                    "The to_bgl() method has been invoked, but it is not implemented in a UDT of type '"
                        + get_name_impl(value) + "'");
    }
    template <typename U, enable_if_t<has_name<U>::value, int> = 0>
    static std::string get_name_impl(const U &value)
    {
        return value.get_name();
    }
    template <typename U, enable_if_t<!has_name<U>::value, int> = 0>
    static std::string get_name_impl(const U &)
    {
        return detail::type_name<U>();
    }
    template <typename U, enable_if_t<has_extra_info<U>::value, int> = 0>
    static std::string get_extra_info_impl(const U &value)
    {
        return value.get_extra_info();
    }
    template <typename U, enable_if_t<!has_extra_info<U>::value, int> = 0>
    static std::string get_extra_info_impl(const U &)
    {
        return "";
    }
    // Get the type at runtime.
    std::type_index get_type_index() const final
    {
        return std::type_index(typeid(T));
    }
    // Raw getters for the internal instance.
    const void *get_ptr() const final
    {
        return &m_value;
    }
    void *get_ptr() final
    {
        return &m_value;
    }
    // Serialization
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, boost::serialization::base_object<topo_inner_base>(*this), m_value);
    }
    T m_value;
};

} // namespace detail

} // namespace pagmo

namespace boost
{

template <typename T>
struct is_virtual_base_of<pagmo::detail::topo_inner_base, pagmo::detail::topo_inner<T>> : false_type {
};

} // namespace boost

namespace pagmo
{

// Topology class.
class PAGMO_DLL_PUBLIC topology
{
    // Enable the generic ctor only if T is not a topology (after removing
    // const/reference qualifiers), and if T is a udt.
    template <typename T>
    using generic_ctor_enabler = enable_if_t<
        detail::conjunction<detail::negation<std::is_same<topology, uncvref_t<T>>>, is_udt<uncvref_t<T>>>::value, int>;

public:
    // Default constructor.
    topology();

private:
    void generic_ctor_impl();

public:
    // Generic constructor.
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit topology(T &&x) : m_ptr(std::make_unique<detail::topo_inner<uncvref_t<T>>>(std::forward<T>(x)))
    {
        generic_ctor_impl();
    }
    // Copy ctor.
    topology(const topology &);
    // Move ctor.
    topology(topology &&) noexcept;
    // Move assignment.
    topology &operator=(topology &&) noexcept;
    // Copy assignment.
    topology &operator=(const topology &);
    // Generic assignment.
    template <typename T, generic_ctor_enabler<T> = 0>
    topology &operator=(T &&x)
    {
        return (*this) = topology(std::forward<T>(x));
    }

    // Extract.
    template <typename T>
    const T *extract() const noexcept
    {
#if defined(PAGMO_PREFER_TYPEID_NAME_EXTRACT)
        return detail::typeid_name_extract<T>(*this);
#else
        auto p = dynamic_cast<const detail::topo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
#endif
    }
    template <typename T>
    T *extract() noexcept
    {
#if defined(PAGMO_PREFER_TYPEID_NAME_EXTRACT)
        return detail::typeid_name_extract<T>(*this);
#else
        auto p = dynamic_cast<detail::topo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
#endif
    }
    template <typename T>
    bool is() const noexcept
    {
        return extract<T>() != nullptr;
    }

    // Name.
    std::string get_name() const
    {
        return m_name;
    }

    // Extra info.
    std::string get_extra_info() const;

    // Check if the topology is valid.
    bool is_valid() const;

    // Get the connections to a vertex.
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;

    // Add a vertex.
    void push_back();
    // Add multiple vertices.
    void push_back(unsigned);

    // Convert to BGL.
    bgl_graph_t to_bgl() const;

    // Get the type at runtime.
    std::type_index get_type_index() const;

    // Get a const pointer to the UDT.
    const void *get_ptr() const;

    // Get a mutable pointer to the UDT.
    void *get_ptr();

    // Serialization.
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_ptr, m_name);
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        topology tmp;
        detail::from_archive(ar, tmp.m_ptr, tmp.m_name);
        *this = std::move(tmp);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    // Two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::topo_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::topo_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }

private:
    std::unique_ptr<detail::topo_inner_base> m_ptr;
    // Various topology properties determined at construction time
    // from the concrete topology. These will be constant for the lifetime
    // of topology, but we cannot mark them as such because of serialization.
    std::string m_name;
};

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Streaming operator for topology.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const topology &);

#endif

namespace detail
{

// A small helper for checking the weight of an edge in a topology.
PAGMO_DLL_PUBLIC void topology_check_edge_weight(double);

} // namespace detail

} // namespace pagmo

// Add some repr support for CLING
PAGMO_IMPLEMENT_XEUS_CLING_REPR(topology)

// Disable tracking for the serialisation of topology.
BOOST_CLASS_TRACKING(pagmo::topology, boost::serialization::track_never)

#endif
