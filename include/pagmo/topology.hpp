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

#ifndef PAGMO_TOPOLOGY_HPP
#define PAGMO_TOPOLOGY_HPP

#include <cassert>
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/visibility.hpp>
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
    virtual std::unique_ptr<topo_inner_base> clone() const override final
    {
        return detail::make_unique<topo_inner>(m_value);
    }
    // The mandatory methods.
    virtual std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t n) const override final
    {
        return m_value.get_connections(n);
    }
    virtual void push_back() override final
    {
        m_value.push_back();
    }
    // Optional methods.
    virtual std::string get_name() const override final
    {
        return get_name_impl(m_value);
    }
    virtual std::string get_extra_info() const override final
    {
        return get_extra_info_impl(m_value);
    }
    // Implementation of the optional methods.
    template <typename U, enable_if_t<has_name<U>::value, int> = 0>
    static std::string get_name_impl(const U &value)
    {
        return value.get_name();
    }
    template <typename U, enable_if_t<!has_name<U>::value, int> = 0>
    static std::string get_name_impl(const U &)
    {
        return typeid(U).name();
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
    explicit topology(T &&x) : m_ptr(detail::make_unique<detail::topo_inner<uncvref_t<T>>>(std::forward<T>(x)))
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
        auto p = dynamic_cast<const detail::topo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::topo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
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

// Disable tracking for the serialisation of topology.
BOOST_CLASS_TRACKING(pagmo::topology, boost::serialization::track_never)

#endif
