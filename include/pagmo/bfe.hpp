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

#ifndef PAGMO_BFE_HPP
#define PAGMO_BFE_HPP

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
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#define PAGMO_S11N_BFE_EXPORT_KEY(b)                                                                                   \
    BOOST_CLASS_EXPORT_KEY2(pagmo::detail::bfe_inner<b>, "udbfe " #b)                                                  \
    BOOST_CLASS_TRACKING(pagmo::detail::bfe_inner<b>, boost::serialization::track_never)

#define PAGMO_S11N_BFE_IMPLEMENT(b) BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::detail::bfe_inner<b>)

#define PAGMO_S11N_BFE_EXPORT(b)                                                                                       \
    PAGMO_S11N_BFE_EXPORT_KEY(b)                                                                                       \
    PAGMO_S11N_BFE_IMPLEMENT(b)

namespace pagmo
{

// Check if T has a call operator conforming to the UDBFE requirements.
template <typename T>
class has_bfe_call_operator
{
    template <typename U>
    using call_t
        = decltype(std::declval<const U &>()(std::declval<const problem &>(), std::declval<const vector_double &>()));
    static const bool implementation_defined = std::is_same<detected_t<call_t, T>, vector_double>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_bfe_call_operator<T>::value;

namespace detail
{

// Specialise this to true in order to disable all the UDBFE checks and mark a type
// as a UDBFE regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python batch evaluators.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_udbfe_checks : std::false_type {
};

} // namespace detail

// Check if T is a UDBFE.
template <typename T>
class is_udbfe
{
    static const bool implementation_defined
        = detail::disjunction<detail::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                                  std::is_copy_constructible<T>, std::is_move_constructible<T>,
                                                  std::is_destructible<T>, has_bfe_call_operator<T>>,
                              detail::disable_udbfe_checks<T>>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udbfe<T>::value;

namespace detail
{

struct PAGMO_DLL_PUBLIC_INLINE_CLASS bfe_inner_base {
    virtual ~bfe_inner_base() {}
    virtual std::unique_ptr<bfe_inner_base> clone() const = 0;
    virtual vector_double operator()(const problem &, const vector_double &) const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual thread_safety get_thread_safety() const = 0;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
struct PAGMO_DLL_PUBLIC_INLINE_CLASS bfe_inner final : bfe_inner_base {
    // We just need the def ctor, delete everything else.
    bfe_inner() = default;
    bfe_inner(const bfe_inner &) = delete;
    bfe_inner(bfe_inner &&) = delete;
    bfe_inner &operator=(const bfe_inner &) = delete;
    bfe_inner &operator=(bfe_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit bfe_inner(const T &x) : m_value(x) {}
    explicit bfe_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of bfe.
    virtual std::unique_ptr<bfe_inner_base> clone() const override final
    {
        return detail::make_unique<bfe_inner>(m_value);
    }
    // Mandatory methods.
    virtual vector_double operator()(const problem &p, const vector_double &dvs) const override final
    {
        return m_value(p, dvs);
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
    virtual thread_safety get_thread_safety() const override final
    {
        return get_thread_safety_impl(m_value);
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
    template <typename U, enable_if_t<has_get_thread_safety<U>::value, int> = 0>
    static thread_safety get_thread_safety_impl(const U &value)
    {
        return value.get_thread_safety();
    }
    template <typename U, enable_if_t<!has_get_thread_safety<U>::value, int> = 0>
    static thread_safety get_thread_safety_impl(const U &)
    {
        return thread_safety::basic;
    }
    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, boost::serialization::base_object<bfe_inner_base>(*this), m_value);
    }
    T m_value;
};

} // namespace detail

} // namespace pagmo

namespace boost
{

template <typename T>
struct is_virtual_base_of<pagmo::detail::bfe_inner_base, pagmo::detail::bfe_inner<T>> : false_type {
};

} // namespace boost

namespace pagmo
{

class PAGMO_DLL_PUBLIC bfe
{
    // Enable the generic ctor only if T is not a bfe (after removing
    // const/reference qualifiers), and if T is a udbfe. Additionally,
    // enable the ctor also if T is a function type (in that case, we
    // will convert the function type to a function pointer in
    // the machinery below).
    template <typename T>
    using generic_ctor_enabler = enable_if_t<
        detail::disjunction<
            detail::conjunction<detail::negation<std::is_same<bfe, uncvref_t<T>>>, is_udbfe<uncvref_t<T>>>,
            std::is_same<vector_double(const problem &, const vector_double &), uncvref_t<T>>>::value,
        int>;
    // Dispatching for the generic ctor. We have a special case if T is
    // a function type, in which case we will manually do the conversion to
    // function pointer and delegate to the other overload.
    template <typename T>
    explicit bfe(T &&x, std::true_type)
        : bfe(static_cast<vector_double (*)(const problem &, const vector_double &)>(std::forward<T>(x)),
              std::false_type{})
    {
    }
    template <typename T>
    explicit bfe(T &&x, std::false_type)
        : m_ptr(detail::make_unique<detail::bfe_inner<uncvref_t<T>>>(std::forward<T>(x)))
    {
    }
    // Implementation of the generic ctor.
    void generic_ctor_impl();

public:
    // Default ctor.
    bfe();
    // Constructor from a UDBFE.
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit bfe(T &&x) : bfe(std::forward<T>(x), std::is_function<uncvref_t<T>>{})
    {
        generic_ctor_impl();
    }
    // Copy constructor.
    bfe(const bfe &);
    // Move constructor.
    bfe(bfe &&) noexcept;
    // Move assignment operator
    bfe &operator=(bfe &&) noexcept;
    // Copy assignment operator
    bfe &operator=(const bfe &);
    // Assignment from a UDBFE.
    template <typename T, generic_ctor_enabler<T> = 0>
    bfe &operator=(T &&x)
    {
        return (*this) = bfe(std::forward<T>(x));
    }

    // Extraction and related.
    template <typename T>
    const T *extract() const noexcept
    {
        auto p = dynamic_cast<const detail::bfe_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::bfe_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    bool is() const noexcept
    {
        return extract<T>() != nullptr;
    }

    // Call operator.
    vector_double operator()(const problem &, const vector_double &) const;

    // Name.
    std::string get_name() const
    {
        return m_name;
    }
    // Extra info.
    std::string get_extra_info() const;

    // Thread safety level.
    thread_safety get_thread_safety() const
    {
        return m_thread_safety;
    }

    // Check if the bfe is valid.
    bool is_valid() const;

    // Serialisation support.
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_ptr, m_name, m_thread_safety);
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        // Deserialize in a separate object and move it in later, for exception safety.
        bfe tmp_bfe;
        detail::from_archive(ar, tmp_bfe.m_ptr, tmp_bfe.m_name, tmp_bfe.m_thread_safety);
        *this = std::move(tmp_bfe);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::bfe_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::bfe_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }

private:
    // Pointer to the inner base bfe
    std::unique_ptr<detail::bfe_inner_base> m_ptr;
    // Various properties determined at construction time
    // from the udbfe. These will be constant for the lifetime
    // of bfe, but we cannot mark them as such because we want to be
    // able to assign and deserialise bfes.
    std::string m_name;
    // Thread safety.
    thread_safety m_thread_safety;
};

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Stream operator.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const bfe &);

#endif

} // namespace pagmo

// Disable tracking for the serialisation of bfe.
BOOST_CLASS_TRACKING(pagmo::bfe, boost::serialization::track_never)

#endif
