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

#ifndef PAGMO_TYPE_TRAITS_HPP
#define PAGMO_TYPE_TRAITS_HPP

#include <cstddef>
#include <initializer_list>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <pagmo/threading.hpp>

namespace pagmo
{

namespace detail
{

// http://en.cppreference.com/w/cpp/types/void_t
template <typename... Ts>
struct make_void {
    typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

// http://en.cppreference.com/w/cpp/experimental/is_detected
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

// http://en.cppreference.com/w/cpp/experimental/nonesuch
struct nonesuch {
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const &) = delete;
    void operator=(nonesuch const &) = delete;
};

// http://en.cppreference.com/w/cpp/types/conjunction
template <class...>
struct conjunction : std::true_type {
};

template <class B1>
struct conjunction<B1> : B1 {
};

template <class B1, class... Bn>
struct conjunction<B1, Bn...> : std::conditional<B1::value != false, conjunction<Bn...>, B1>::type {
};

// http://en.cppreference.com/w/cpp/types/disjunction
template <class...>
struct disjunction : std::false_type {
};

template <class B1>
struct disjunction<B1> : B1 {
};

template <class B1, class... Bn>
struct disjunction<B1, Bn...> : std::conditional<B1::value != false, B1, disjunction<Bn...>>::type {
};

// http://en.cppreference.com/w/cpp/types/negation
template <class B>
struct negation : std::integral_constant<bool, !B::value> {
};

// std::index_sequence and std::make_index_sequence implementation for C++11. These are available
// in the std library in C++14. Implementation taken from:
// http://stackoverflow.com/questions/17424477/implementation-c14-make-integer-sequence
template <std::size_t... Ints>
struct index_sequence {
    using type = index_sequence;
    using value_type = std::size_t;
    static constexpr std::size_t size() noexcept
    {
        return sizeof...(Ints);
    }
};

template <class Sequence1, class Sequence2>
struct merge_and_renumber;

template <std::size_t... I1, std::size_t... I2>
struct merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
    : index_sequence<I1..., (sizeof...(I1) + I2)...> {
};

template <std::size_t N>
struct make_index_sequence
    : merge_and_renumber<typename make_index_sequence<N / 2>::type, typename make_index_sequence<N - N / 2>::type> {
};

template <>
struct make_index_sequence<0> : index_sequence<> {
};

template <>
struct make_index_sequence<1> : index_sequence<0> {
};

template <typename T, typename F, std::size_t... Is>
void apply_to_each_item(T &&t, const F &f, index_sequence<Is...>)
{
    (void)std::initializer_list<int>{0, (void(f(std::get<Is>(std::forward<T>(t)))), 0)...};
}

// Tuple for_each(). Execute the functor f on each element of the input Tuple.
// https://isocpp.org/blog/2015/01/for-each-arg-eric-niebler
// https://www.reddit.com/r/cpp/comments/2tffv3/for_each_argumentsean_parent/
// https://www.reddit.com/r/cpp/comments/33b06v/for_each_in_tuple/
template <class Tuple, class F>
void tuple_for_each(Tuple &&t, const F &f)
{
    apply_to_each_item(std::forward<Tuple>(t), f,
                       make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
}

} // namespace detail

/// Implementation of \p std::is_detected.
/**
 * Implementation of \p std::is_detected, from C++17. See: http://en.cppreference.com/w/cpp/experimental/is_detected.
 */
template <template <class...> class Op, class... Args>
using is_detected =
#if defined(PAGMO_DOXYGEN_INVOKED)
    implementation_defined;
#else
    typename detail::detector<detail::nonesuch, void, Op, Args...>::value_t;
#endif

/// Implementation of \p std::detected_t.
/**
 * Implementation of \p std::detected_t, from C++17. See: http://en.cppreference.com/w/cpp/experimental/is_detected.
 */
template <template <class...> class Op, class... Args>
using detected_t =
#if defined(PAGMO_DOXYGEN_INVOKED)
    implementation_defined;
#else
    typename detail::detector<detail::nonesuch, void, Op, Args...>::type;
#endif

/// Implementation of \p std::decay_t.
/**
 * Implementation of \p std::decay_t, from C++14. See: http://en.cppreference.com/w/cpp/types/decay.
 */
template <typename T>
using decay_t = typename std::decay<T>::type;

/// Implementation of \p std::enable_if_t.
/**
 * Implementation of \p std::enable_if_t, from C++14. See: http://en.cppreference.com/w/cpp/types/enable_if.
 */
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

namespace detail
{
/// SFINAE enabler for floating point types
/**
 * A templated method or function with enable_if_is_floating_point<T> = 0 will only be available for
 * floating point types
 */
template <typename T>
using enable_if_is_floating_point = enable_if_t<std::is_floating_point<T>::value, int>;
/// SFINAE enabler for floating point types
/**
 * A templated method or function with enable_if_is_not_floating_point<T> = 0 will only be available for
 * non floating point types
 */
template <typename T>
using enable_if_is_not_floating_point = enable_if_t<!std::is_floating_point<T>::value, int>;
} // namespace detail

/// Remove reference and cv qualifiers from type \p T.
template <typename T>
using uncvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

/// Detect \p set_seed() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * void set_seed(unsigned);
 * @endcode
 * The \p set_seed() method is part of the interface for the definition of problems and algorithms
 * (see pagmo::problem and pagmo::algorithm).
 */
template <typename T>
class has_set_seed
{
    template <typename U>
    using set_seed_t = decltype(std::declval<U &>().set_seed(1u));
    static const bool implementation_defined = std::is_same<void, detected_t<set_seed_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_set_seed<T>::value;

/// Detect \p has_set_seed() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * bool has_set_seed() const;
 * @endcode
 * The \p has_set_seed() method is part of the interface for the definition of problems and algorithms
 * (see pagmo::problem and pagmo::algorithm).
 */
template <typename T>
class override_has_set_seed
{
    template <typename U>
    using has_set_seed_t = decltype(std::declval<const U &>().has_set_seed());
    static const bool implementation_defined = std::is_same<bool, detected_t<has_set_seed_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_set_seed<T>::value;

/// Detect \p get_name() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * std::string get_name() const;
 * @endcode
 * The \p get_name() method is part of the interface for the definition of problems and algorithms
 * (see pagmo::problem and pagmo::algorithm).
 */
template <typename T>
class has_name
{
    template <typename U>
    using get_name_t = decltype(std::declval<const U &>().get_name());
    static const bool implementation_defined = std::is_same<std::string, detected_t<get_name_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_name<T>::value;

/// Detect \p get_extra_info() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * std::string get_extra_info() const;
 * @endcode
 * The \p get_extra_info() method is part of the interface for the definition of problems and algorithms
 * (see pagmo::problem and pagmo::algorithm).
 */
template <typename T>
class has_extra_info
{
    template <typename U>
    using get_extra_info_t = decltype(std::declval<const U &>().get_extra_info());
    static const bool implementation_defined = std::is_same<std::string, detected_t<get_extra_info_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_extra_info<T>::value;

/// Detect \p get_thread_safety() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * pagmo::thread_safety get_thread_safety() const;
 * @endcode
 * The \p get_thread_safety() method is part of the interface for the definition of problems and algorithms
 * (see pagmo::problem and pagmo::algorithm).
 */
template <typename T>
class has_get_thread_safety
{
    template <typename U>
    using get_thread_safety_t = decltype(std::declval<const U &>().get_thread_safety());
    static const bool implementation_defined = std::is_same<thread_safety, detected_t<get_thread_safety_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_thread_safety<T>::value;

} // namespace pagmo

#endif
