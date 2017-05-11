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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

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

/// Detect \p get_connections() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * std::pair<std::vector<std::size_t>,vector_double> get_connections(std::size_t) const;
 * @endcode
 * The \p get_connections() method is part of the interface for the definition of a topology
 * (see pagmo::topology).
 */
template <typename T>
class has_get_connections
{
    template <typename U>
    using get_connections_t = decltype(std::declval<const U &>().get_connections(std::size_t(0)));
    static const bool implementation_defined
        = std::is_same<std::pair<std::vector<std::size_t>, vector_double>, detected_t<get_connections_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_connections<T>::value;

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

namespace detail
{

// Specialise this to true in order to disable all the UDT checks and mark a type
// as a UDT regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python topos.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_udt_checks : std::false_type {
};
}

/// Detect user-defined topologies (UDT).
/**
 * This type trait will be \p true if \p T is not cv/reference qualified, it is destructible, default, copy and move
 * constructible, and if it satisfies the pagmo::has_get_connections and pagmo::has_push_back type traits.
 *
 * Types satisfying this type trait can be used as user-defined topologies (UDT) in pagmo::topology.
 */
template <typename T>
class is_udt
{
    static const bool implementation_defined
        = (std::is_same<T, uncvref_t<T>>::value && std::is_default_constructible<T>::value
           && std::is_copy_constructible<T>::value && std::is_move_constructible<T>::value
           && std::is_destructible<T>::value && has_get_connections<T>::value && has_push_back<T>::value)
          || detail::disable_udt_checks<T>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udt<T>::value;

namespace detail
{

struct topo_inner_base {
    virtual ~topo_inner_base()
    {
    }
    virtual std::unique_ptr<topo_inner_base> clone() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const = 0;
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
    /// Get the list of connections.
    /**
     * In an unconnected topology there are no connections for any vertex.
     *
     * @return a pair of empty vectors.
     */
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const
    {
        return std::make_pair(std::vector<std::size_t>{}, vector_double{});
    }
    /// Add the next vertex.
    /**
     * This method is a no-op.
     */
    void push_back()
    {
    }
    /// Get the name of the topology.
    /**
     * @return ``"Unconnected"``.
     */
    std::string get_name() const
    {
        return "Unconnected";
    }
    // Serialization.
    /**
     * This class is stateless, no data will be loaded or saved during serialization.
     */
    template <typename Archive>
    void serialize(Archive &)
    {
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
 *   optimisation process (via the migration of individuals),
 * * the *weights* of the edges (whose numerical values are the \f$ [0.,1.] \f$ range) represent the migration
 *   probability.
 *
 * Following the same schema adopted for pagmo::problem, pagmo::algorithm, etc., pagmo::topology exposes a generic
 * interface to **user-defined topologies** (or UDT for short). UDTs are classes (or struct) exposing a certain set
 * of methods that describe the properties of (and allow to interact with) a topology. Once
 * defined and instantiated, a UDT can then be used to construct an instance of this class, pagmo::topology, which
 * provides a generic interface to topologies for use by pagmo::archipelago.
 *
 * In pagmo::topology, vertices in the graph are identified by a zero-based unique integral index (represented by
 * an \p std::size_t). This integral index corresponds to the index of an
 * \link pagmo::island island\endlink in an \link pagmo::archipelago archipelago\endlink.
 * Every UDT must implement at least the following methods:
 * @code{.unparsed}
 * std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;
 * void push_back();
 * @endcode
 *
 * The <tt>%get_connections()</tt> method takes as input a vertex index \p n, and it is expected to return
 * a pair of vectors containing respectively:
 * * the indices of the vertices which are connecting to \p n (that is, the list of vertices for which a directed edge
 *   towards \p n exists),
 * * the weights (i.e., the migration probabilities) of the edges linking the connecting vertices to \p n.
 *
 * The <tt>%push_back()</tt> method is expected to add a new vertex to the topology, assigning it the next
 * available index and establishing connections to other vertices. The <tt>%push_back()</tt> method is invoked
 * by pagmo::archipelago::push_back() upon the insertion of a new island into an archipelago, and it is meant
 * to allow the incremental construction of a topology. That is, after ``N`` calls to <tt>%push_back()</tt>
 * on an initially-empty topology, the topology should contain ``N`` vertices and any number of edges (depending
 * on the specifics of the topology).
 *
 * In addition to providing the above methods, a UDT must also be default, copy and move constructible.
 *
 * Additional optional methods can be implemented in a UDT:
 * @code{.unparsed}
 * std::string get_name() const;
 * std::string get_extra_info() const;
 * @endcode
 *
 * See the documentation of the corresponding methods in this class for details on how the optional
 * methods in the UDT are used by pagmo::topology.
 *
 * Topologies are often used in asynchronous operations involving migration in archipelagos. pagmo
 * guarantees that only a single thread at a time is interacting with any topology, so there is no
 * need to protect UDTs against concurrent access. Topologies however are **required** to offer at
 * least the thread_safety::basic guarantee, in order to make it possible to use different
 * topologies from different threads.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    A moved-from :cpp:class:`pagmo::topology` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * \endverbatim
 */
class topology
{
    // Enable the generic ctor only if T is not a topo (after removing
    // const/reference qualifiers), and if T is a udt.
    template <typename T>
    using generic_ctor_enabler
        = enable_if_t<!std::is_same<topology, uncvref_t<T>>::value && is_udt<uncvref_t<T>>::value, int>;

public:
    /// Default constructor.
    /**
     * The default constructor will initialize a pagmo::topology containing a pagmo::unconnected.
     *
     * @throws unspecified any exception thrown by the constructor from UDT.
     */
    topology() : topology(unconnected{})
    {
    }
    /// Constructor from a user-defined topology of type \p T
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is not enabled if, after the removal of cv and reference qualifiers,
     *    ``T`` is of type :cpp:class:`pagmo::topology` (that is, this constructor does not compete with the copy/move
     *    constructors of :cpp:class:`pagmo::topology`), or if ``T`` does not satisfy :cpp:class:`pagmo::is_udt`.
     *
     * \endverbatim
     *
     * This constructor will construct a pagmo::topology from the UDT (user-defined topology) \p x of type \p T. In
     * order for the construction to be successful, the UDT must implement a minimal set of methods,
     * as described in the documentation of pagmo::topology. The constructor will examine the properties of \p x and
     * store them as data members of \p this.
     *
     * @param x the UDT.
     *
     * @throws unspecified any exception thrown by methods of the UDT invoked during construction or by memory errors
     * in strings and standard containers.
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit topology(T &&x) : m_ptr(detail::make_unique<detail::topo_inner<uncvref_t<T>>>(std::forward<T>(x)))
    {
        // We store at construction the value returned from the user implemented get_name
        m_name = ptr()->get_name();
    }
    /// Copy constructor
    /**
     * The copy constructor will deep copy the input topology \p other.
     *
     * @param other the topology to be copied.
     *
     * @throws unspecified any exception thrown by:
     * - memory allocation errors in standard containers,
     * - the copying of the internal UDT.
     */
    topology(const topology &other) : m_ptr(other.m_ptr->clone()), m_name(other.m_name)
    {
    }
    /// Move constructor
    /**
     * @param other the topology from which \p this will be move-constructed.
     */
    topology(topology &&other) noexcept : m_ptr(std::move(other.m_ptr)), m_name(std::move(other.m_name))
    {
    }
    /// Move assignment operator
    /**
     * @param other the assignment target.
     *
     * @return a reference to \p this.
     */
    topology &operator=(topology &&other) noexcept
    {
        if (this != &other) {
            m_ptr = std::move(other.m_ptr);
            m_name = std::move(other.m_name);
        }
        return *this;
    }
    /// Copy assignment operator
    /**
     * Copy assignment is implemented as a copy constructor followed by a move assignment.
     *
     * @param other the assignment target.
     *
     * @return a reference to \p this.
     *
     * @throws unspecified any exception thrown by the copy constructor.
     */
    topology &operator=(const topology &other)
    {
        // Copy ctor + move assignment.
        return *this = topology(other);
    }

    /// Extract a const pointer to the UDT used for construction.
    /**
     * This method will extract a const pointer to the internal instance of the UDT. If \p T is not the same type
     * as the UDT used during construction (after removal of cv and reference qualifiers), this method will
     * return \p nullptr.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
     *    of ``this``, and ``delete`` must never be called on the pointer.
     *
     * \endverbatim
     *
     * @return a const pointer to the internal UDT, or \p nullptr
     * if \p T does not correspond exactly to the original UDT type used
     * in the constructor.
     */
    template <typename T>
    const T *extract() const
    {
        auto p = dynamic_cast<const detail::topo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    /// Extract a pointer to the UDT used for construction.
    /**
     * This method will extract a pointer to the internal instance of the UDT. If \p T is not the same type
     * as the UDT used during construction (after removal of cv and reference qualifiers), this method will
     * return \p nullptr.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
     *    of ``this``, and ``delete`` must never be called on the pointer.
     *
     * .. note::
     *
     *    The ability to extract a mutable pointer is provided only in order to allow to call non-const
     *    methods on the internal UDT instance. Assigning a new UDT via this pointer is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a pointer to the internal UDT, or \p nullptr
     * if \p T does not correspond exactly to the original UDT type used
     * in the constructor.
     */
    template <typename T>
    T *extract()
    {
        auto p = dynamic_cast<detail::topo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    /// Check if the UDT used for construction is of type \p T.
    /**
     * @return \p true if the UDT used in construction is of type \p T, \p false otherwise.
     */
    template <typename T>
    bool is() const
    {
        return extract<T>() != nullptr;
    }

    /// Topology's name.
    /**
     * If the UDT satisfies pagmo::has_name, then this method will return the output of its <tt>%get_name()</tt> method.
     * Otherwise, an implementation-defined name based on the type of the UDT will be returned.
     *
     * @return the topology's name.
     *
     * @throws unspecified any exception thrown by copying an \p std::string object.
     */
    std::string get_name() const
    {
        return m_name;
    }

    /// Topology's extra info.
    /**
     * If the UDT satisfies pagmo::has_extra_info, then this method will return the output of its
     * <tt>%get_extra_info()</tt> method. Otherwise, an empty string will be returned.
     *
     * @return extra info about the UDT.
     *
     * @throws unspecified any exception thrown by the <tt>%get_extra_info()</tt> method of the UDT.
     */
    std::string get_extra_info() const
    {
        return ptr()->get_extra_info();
    }

    /// Get the connections to a vertex.
    /**
     * This method will invoke the <tt>%get_connections()</tt> method of the UDT, which is expected to return
     * a pair of vectors containing respectively:
     * * the indices of the vertices which are connecting to \p n (that is, the list of vertices for which a directed
     *   edge towards \p n exists),
     * * the weights (i.e., the migration probabilities) of the edges linking the connecting vertices to \p n.
     *
     * This method will also run sanity checks on the output of the <tt>%get_connections()</tt> method of the UDT.
     *
     * @param n the index of the vertex whose incoming connections' details will be returned.
     *
     * @return a pair of vectors describing <tt>n</tt>'s incoming connections.
     *
     * @throws std::invalid_argument if the sizes of the returned vectors differ, or if any element of the second
     * vector is not in the \f$ [0.,1.] \f$ range.
     * @throws unspecified any exception thrown by the <tt>%get_connections()</tt> method of the UDT.
     */
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t n) const
    {
        auto retval = ptr()->get_connections(n);
        // Check the returned value.
        if (retval.first.size() != retval.second.size()) {
            pagmo_throw(std::invalid_argument,
                        "An invalid pair of vectors was returned by the 'get_connections()' method of the '"
                            + get_name() + "' topology: the vector of connecting islands has a size of "
                            + std::to_string(retval.first.size())
                            + ", while the vector of migration probabilities has a size of "
                            + std::to_string(retval.second.size()) + " (the two sizes must be equal)");
        }
        for (const auto &p : retval.second) {
            if (!std::isfinite(p)) {
                pagmo_throw(
                    std::invalid_argument,
                    "An invalid non-finite migration probability of " + std::to_string(p)
                        + " was detected in the vector of migration probabilities returned by the 'get_connections()' "
                          "method of the '"
                        + get_name() + "' topology");
            }
            if (p < 0. || p > 1.) {
                pagmo_throw(
                    std::invalid_argument,
                    "An invalid migration probability of " + std::to_string(p)
                        + " was detected in the vector of migration probabilities returned by the 'get_connections()' "
                          "method of the '"
                        + get_name() + "' topology: the value must be in the [0.,1.] range");
            }
        }
        return retval;
    }
    /// Add a vertex.
    /**
     * This method will invoke the <tt>%push_back()</tt> method of the UDT, which is expected to add a new vertex to the
     * topology, assigning it the next available index and establishing connections to other vertices.
     *
     * @throws unspecified any exception thrown by the <tt>%push_back()</tt> method of the UDT.
     */
    void push_back()
    {
        ptr()->push_back();
    }

    /// Streaming operator
    /**
     * This function will stream to \p os a human-readable representation of the input
     * topology \p t.
     *
     * @param os input <tt>std::ostream</tt>.
     * @param t pagmo::topology object to be streamed.
     *
     * @return a reference to \p os.
     *
     * @throws unspecified any exception thrown by querying various topology properties and streaming them into \p os.
     */
    friend std::ostream &operator<<(std::ostream &os, const topology &t)
    {
        os << "Topology name: " << t.get_name();
        const auto extra_str = t.get_extra_info();
        if (!extra_str.empty()) {
            stream(os, "\nExtra info:\n", extra_str);
        }
        return os;
    }

    /// Save to archive.
    /**
     * This method will save \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDT and of primitive types.
     */
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(m_ptr, m_name);
    }
    /// Load from archive.
    /**
     * This method will load a pagmo::topology from \p ar into \p this.
     *
     * @param ar source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of the UDT and of primitive types.
     */
    template <typename Archive>
    void load(Archive &ar)
    {
        topology tmp;
        ar(tmp.m_ptr, tmp.m_name);
        *this = std::move(tmp);
    }

private:
    // Just two small helpers to make sure that whenever we require
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
    // Pointer to the inner base topo.
    std::unique_ptr<detail::topo_inner_base> m_ptr;
    // Name of the topo.
    std::string m_name;
};
}

PAGMO_REGISTER_TOPOLOGY(pagmo::unconnected)

#endif
