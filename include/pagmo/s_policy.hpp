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

#ifndef PAGMO_S_POLICY_HPP
#define PAGMO_S_POLICY_HPP

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#define PAGMO_S11N_S_POLICY_EXPORT_KEY(s)                                                                              \
    BOOST_CLASS_EXPORT_KEY2(pagmo::detail::s_pol_inner<s>, "udsp " #s)                                                 \
    BOOST_CLASS_TRACKING(pagmo::detail::s_pol_inner<s>, boost::serialization::track_never)

#define PAGMO_S11N_S_POLICY_IMPLEMENT(s) BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::detail::s_pol_inner<s>)

#define PAGMO_S11N_S_POLICY_EXPORT(s)                                                                                  \
    PAGMO_S11N_S_POLICY_EXPORT_KEY(s)                                                                                  \
    PAGMO_S11N_S_POLICY_IMPLEMENT(s)

namespace pagmo
{

// Check if T has a select() member function conforming to the UDSP requirements.
template <typename T>
class has_select
{
    template <typename U>
    using select_t = decltype(std::declval<const U &>().select(
        std::declval<const individuals_group_t &>(), std::declval<const vector_double::size_type &>(),
        std::declval<const vector_double::size_type &>(), std::declval<const vector_double::size_type &>(),
        std::declval<const vector_double::size_type &>(), std::declval<const vector_double::size_type &>(),
        std::declval<const vector_double &>()));
    static const bool implementation_defined = std::is_same<detected_t<select_t, T>, individuals_group_t>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_select<T>::value;

namespace detail
{

// Specialise this to true in order to disable all the UDSP checks and mark a type
// as a UDSP regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python s_policies.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_udsp_checks : std::false_type {
};

} // namespace detail

// Detect UDSPs
template <typename T>
class is_udsp
{
    static const bool implementation_defined
        = detail::disjunction<detail::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                                  std::is_copy_constructible<T>, std::is_move_constructible<T>,
                                                  std::is_destructible<T>, has_select<T>>,
                              detail::disable_udsp_checks<T>>::value;

public:
    // Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udsp<T>::value;

namespace detail
{

struct PAGMO_DLL_PUBLIC_INLINE_CLASS s_pol_inner_base {
    virtual ~s_pol_inner_base() {}
    virtual std::unique_ptr<s_pol_inner_base> clone() const = 0;
    virtual individuals_group_t select(const individuals_group_t &, const vector_double::size_type &,
                                       const vector_double::size_type &, const vector_double::size_type &,
                                       const vector_double::size_type &, const vector_double::size_type &,
                                       const vector_double &) const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
struct PAGMO_DLL_PUBLIC_INLINE_CLASS s_pol_inner final : s_pol_inner_base {
    // We just need the def ctor, delete everything else.
    s_pol_inner() = default;
    s_pol_inner(const s_pol_inner &) = delete;
    s_pol_inner(s_pol_inner &&) = delete;
    s_pol_inner &operator=(const s_pol_inner &) = delete;
    s_pol_inner &operator=(s_pol_inner &&) = delete;
    // Constructors from T.
    explicit s_pol_inner(const T &x) : m_value(x) {}
    explicit s_pol_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of s_policy.
    virtual std::unique_ptr<s_pol_inner_base> clone() const override final
    {
        return detail::make_unique<s_pol_inner>(m_value);
    }
    // The mandatory select() method.
    virtual individuals_group_t select(const individuals_group_t &inds, const vector_double::size_type &nx,
                                       const vector_double::size_type &nix, const vector_double::size_type &nobj,
                                       const vector_double::size_type &nec, const vector_double::size_type &nic,
                                       const vector_double &tol) const override final
    {
        return m_value.select(inds, nx, nix, nobj, nec, nic, tol);
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
        detail::archive(ar, boost::serialization::base_object<s_pol_inner_base>(*this), m_value);
    }
    T m_value;
};

} // namespace detail

} // namespace pagmo

namespace boost
{

template <typename T>
struct is_virtual_base_of<pagmo::detail::s_pol_inner_base, pagmo::detail::s_pol_inner<T>> : false_type {
};

} // namespace boost

namespace pagmo
{

// Selection policy.
class PAGMO_DLL_PUBLIC s_policy
{
    // Enable the generic ctor only if T is not an s_policy (after removing
    // const/reference qualifiers), and if T is a udsp.
    template <typename T>
    using generic_ctor_enabler = enable_if_t<
        detail::conjunction<detail::negation<std::is_same<s_policy, uncvref_t<T>>>, is_udsp<uncvref_t<T>>>::value, int>;
    // Implementation of the generic ctor.
    void generic_ctor_impl();

public:
    // Default constructor.
    s_policy();
    // Constructor from a UDSP.
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit s_policy(T &&x) : m_ptr(detail::make_unique<detail::s_pol_inner<uncvref_t<T>>>(std::forward<T>(x)))
    {
        generic_ctor_impl();
    }
    // Copy constructor.
    s_policy(const s_policy &);
    // Move constructor.
    s_policy(s_policy &&) noexcept;
    // Move assignment operator
    s_policy &operator=(s_policy &&) noexcept;
    // Copy assignment operator
    s_policy &operator=(const s_policy &);
    // Assignment from a UDSP.
    template <typename T, generic_ctor_enabler<T> = 0>
    s_policy &operator=(T &&x)
    {
        return (*this) = s_policy(std::forward<T>(x));
    }

    // Extraction and related.
    template <typename T>
    const T *extract() const noexcept
    {
        auto p = dynamic_cast<const detail::s_pol_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::s_pol_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    bool is() const noexcept
    {
        return extract<T>() != nullptr;
    }

    // Select.
    individuals_group_t select(const individuals_group_t &, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double &) const;

    // Name.
    std::string get_name() const
    {
        return m_name;
    }
    // Extra info.
    std::string get_extra_info() const;

    // Check if the s_policy is valid.
    bool is_valid() const;

    // Serialisation support.
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_ptr, m_name);
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        // Deserialize in a separate object and move it in later, for exception safety.
        s_policy tmp_s_pol;
        detail::from_archive(ar, tmp_s_pol.m_ptr, tmp_s_pol.m_name);
        *this = std::move(tmp_s_pol);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::s_pol_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::s_pol_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    // Helper to check the inputs and outputs of the select() function.
    PAGMO_DLL_LOCAL void verify_select_input(const individuals_group_t &, const vector_double::size_type &,
                                             const vector_double::size_type &, const vector_double::size_type &,
                                             const vector_double::size_type &, const vector_double::size_type &,
                                             const vector_double &) const;
    PAGMO_DLL_LOCAL void verify_select_output(const individuals_group_t &, vector_double::size_type,
                                              vector_double::size_type) const;

private:
    // Pointer to the inner base s_pol.
    std::unique_ptr<detail::s_pol_inner_base> m_ptr;
    // Various properties determined at construction time
    // from the udsp. These will be constant for the lifetime
    // of s_policy, but we cannot mark them as such because we want to be
    // able to assign and deserialise s_policies.
    std::string m_name;
};

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Stream operator.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const s_policy &);

#endif

} // namespace pagmo

// Disable tracking for the serialisation of s_policy.
BOOST_CLASS_TRACKING(pagmo::s_policy, boost::serialization::track_never)

#endif
