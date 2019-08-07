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

#ifndef PAGMO_R_POLICY_HPP
#define PAGMO_R_POLICY_HPP

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

#define PAGMO_S11N_R_POLICY_EXPORT_KEY(r)                                                                              \
    BOOST_CLASS_EXPORT_KEY2(pagmo::detail::r_pol_inner<r>, "udrp " #r)                                                 \
    BOOST_CLASS_TRACKING(pagmo::detail::r_pol_inner<r>, boost::serialization::track_never)

#define PAGMO_S11N_R_POLICY_IMPLEMENT(r) BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::detail::r_pol_inner<r>)

#define PAGMO_S11N_R_POLICY_EXPORT(r)                                                                                  \
    PAGMO_S11N_R_POLICY_EXPORT_KEY(r)                                                                                  \
    PAGMO_S11N_R_POLICY_IMPLEMENT(r)

namespace pagmo
{

// Check if T has a replace() member function conforming to the UDRP requirements.
template <typename T>
class has_replace
{
    template <typename U>
    using replace_t = decltype(std::declval<const U &>().replace(
        std::declval<const individuals_group_t &>(), std::declval<const vector_double::size_type &>(),
        std::declval<const vector_double::size_type &>(), std::declval<const vector_double::size_type &>(),
        std::declval<const vector_double::size_type &>(), std::declval<const vector_double::size_type &>(),
        std::declval<const vector_double &>(), std::declval<const individuals_group_t &>()));
    static const bool implementation_defined = std::is_same<detected_t<replace_t, T>, individuals_group_t>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_replace<T>::value;

namespace detail
{

// Specialise this to true in order to disable all the UDRP checks and mark a type
// as a UDRP regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python r_policies.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_udrp_checks : std::false_type {
};

} // namespace detail

// Detect UDRPs
template <typename T>
class is_udrp
{
    static const bool implementation_defined
        = detail::disjunction<detail::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                                  std::is_copy_constructible<T>, std::is_move_constructible<T>,
                                                  std::is_destructible<T>, has_replace<T>>,
                              detail::disable_udrp_checks<T>>::value;

public:
    // Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udrp<T>::value;

namespace detail
{

struct PAGMO_DLL_PUBLIC_INLINE_CLASS r_pol_inner_base {
    virtual ~r_pol_inner_base() {}
    virtual std::unique_ptr<r_pol_inner_base> clone() const = 0;
    virtual individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                        const vector_double::size_type &, const vector_double::size_type &,
                                        const vector_double::size_type &, const vector_double::size_type &,
                                        const vector_double &, const individuals_group_t &) const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
struct PAGMO_DLL_PUBLIC_INLINE_CLASS r_pol_inner final : r_pol_inner_base {
    // We just need the def ctor, delete everything else.
    r_pol_inner() = default;
    r_pol_inner(const r_pol_inner &) = delete;
    r_pol_inner(r_pol_inner &&) = delete;
    r_pol_inner &operator=(const r_pol_inner &) = delete;
    r_pol_inner &operator=(r_pol_inner &&) = delete;
    // Constructors from T.
    explicit r_pol_inner(const T &x) : m_value(x) {}
    explicit r_pol_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of r_policy.
    virtual std::unique_ptr<r_pol_inner_base> clone() const override final
    {
        return detail::make_unique<r_pol_inner>(m_value);
    }
    // The mandatory replace() method.
    virtual individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &nx,
                                        const vector_double::size_type &nix, const vector_double::size_type &nobj,
                                        const vector_double::size_type &nec, const vector_double::size_type &nic,
                                        const vector_double &tol, const individuals_group_t &mig) const override final
    {
        return m_value.replace(inds, nx, nix, nobj, nec, nic, tol, mig);
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
        detail::archive(ar, boost::serialization::base_object<r_pol_inner_base>(*this), m_value);
    }
    T m_value;
};

} // namespace detail

} // namespace pagmo

namespace boost
{

template <typename T>
struct is_virtual_base_of<pagmo::detail::r_pol_inner_base, pagmo::detail::r_pol_inner<T>> : false_type {
};

} // namespace boost

namespace pagmo
{

// Replacement policy.
class PAGMO_DLL_PUBLIC r_policy
{
    // Enable the generic ctor only if T is not an r_policy (after removing
    // const/reference qualifiers), and if T is a udrp.
    template <typename T>
    using generic_ctor_enabler = enable_if_t<
        detail::conjunction<detail::negation<std::is_same<r_policy, uncvref_t<T>>>, is_udrp<uncvref_t<T>>>::value, int>;
    // Implementation of the generic ctor.
    void generic_ctor_impl();

public:
    // Default constructor.
    r_policy();
    // Constructor from a UDRP.
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit r_policy(T &&x) : m_ptr(detail::make_unique<detail::r_pol_inner<uncvref_t<T>>>(std::forward<T>(x)))
    {
        generic_ctor_impl();
    }
    // Copy constructor.
    r_policy(const r_policy &);
    // Move constructor.
    r_policy(r_policy &&) noexcept;
    // Move assignment operator
    r_policy &operator=(r_policy &&) noexcept;
    // Copy assignment operator
    r_policy &operator=(const r_policy &);
    // Assignment from a UDRP.
    template <typename T, generic_ctor_enabler<T> = 0>
    r_policy &operator=(T &&x)
    {
        return (*this) = r_policy(std::forward<T>(x));
    }

    // Extraction and related.
    template <typename T>
    const T *extract() const noexcept
    {
        auto p = dynamic_cast<const detail::r_pol_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::r_pol_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    bool is() const noexcept
    {
        return extract<T>() != nullptr;
    }

    // Replace.
    individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double &, const individuals_group_t &) const;

    // Name.
    std::string get_name() const
    {
        return m_name;
    }
    // Extra info.
    std::string get_extra_info() const;

    // Check if the r_policy is valid.
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
        r_policy tmp_r_pol;
        detail::from_archive(ar, tmp_r_pol.m_ptr, tmp_r_pol.m_name);
        *this = std::move(tmp_r_pol);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::r_pol_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::r_pol_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    // Helper to check the inputs and outputs of the replace() function.
    PAGMO_DLL_LOCAL void verify_replace_input(const individuals_group_t &, const vector_double::size_type &,
                                              const vector_double::size_type &, const vector_double::size_type &,
                                              const vector_double::size_type &, const vector_double::size_type &,
                                              const vector_double &, const individuals_group_t &) const;
    PAGMO_DLL_LOCAL void verify_replace_output(const individuals_group_t &, vector_double::size_type,
                                               vector_double::size_type) const;

private:
    // Pointer to the inner base r_pol.
    std::unique_ptr<detail::r_pol_inner_base> m_ptr;
    // Various properties determined at construction time
    // from the udrp. These will be constant for the lifetime
    // of r_policy, but we cannot mark them as such because we want to be
    // able to assign and deserialise r_policies.
    std::string m_name;
};

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Stream operator.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const r_policy &);

#endif

} // namespace pagmo

// Disable tracking for the serialisation of r_policy.
BOOST_CLASS_TRACKING(pagmo::r_policy, boost::serialization::track_never)

#endif
