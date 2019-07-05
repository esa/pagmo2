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

#ifndef PAGMO_PROBLEM_HPP
#define PAGMO_PROBLEM_HPP

#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

// NOTE: we disable address tracking for all user-defined classes. The reason is that even if the final
// classes (e.g., problem) use value semantics, the internal implementation details use old-style
// OO construct (i.e., base classes, pointers, etc.). By default, Boost serialization wants to track
// the addresses of these internal implementation-detail classes, and this has some undesirable consequences
// (for instance, when deserializing a problem object in a variable and then moving it into another
// one, which is a pattern we sometimes use in order to provide increased exception safety).
//
// See also:
// https://www.boost.org/doc/libs/1_70_0/libs/serialization/doc/special.html#objecttracking
// https://www.boost.org/doc/libs/1_70_0/libs/serialization/doc/traits.html#level
#define PAGMO_S11N_PROBLEM_EXPORT_KEY(prob)                                                                            \
    BOOST_CLASS_EXPORT_KEY2(pagmo::detail::prob_inner<prob>, "udp " #prob)                                             \
    BOOST_CLASS_TRACKING(pagmo::detail::prob_inner<prob>, boost::serialization::track_never)

#define PAGMO_S11N_PROBLEM_IMPLEMENT(prob) BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::detail::prob_inner<prob>)

#define PAGMO_S11N_PROBLEM_EXPORT(prob)                                                                                \
    PAGMO_S11N_PROBLEM_EXPORT_KEY(prob)                                                                                \
    PAGMO_S11N_PROBLEM_IMPLEMENT(prob)

namespace pagmo
{

/// Detect \p fitness() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * vector_double fitness(const vector_double &) const;
 * @endcode
 * The \p fitness() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_fitness
{
    template <typename U>
    using fitness_t = decltype(std::declval<const U &>().fitness(std::declval<const vector_double &>()));
    static const bool implementation_defined = std::is_same<detected_t<fitness_t, T>, vector_double>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_fitness<T>::value;

/// Detect \p get_nobj() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * vector_double::size_type get_nobj() const;
 * @endcode
 * The \p get_nobj() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_get_nobj
{
    template <typename U>
    using get_nobj_t = decltype(std::declval<const U &>().get_nobj());
    static const bool implementation_defined = std::is_same<detected_t<get_nobj_t, T>, vector_double::size_type>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_nobj<T>::value;

/// Detect \p get_bounds() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * std::pair<vector_double, vector_double> get_bounds() const;
 * @endcode
 * The \p get_bounds() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_bounds
{
    template <typename U>
    using get_bounds_t = decltype(std::declval<const U &>().get_bounds());
    static const bool implementation_defined
        = std::is_same<std::pair<vector_double, vector_double>, detected_t<get_bounds_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_bounds<T>::value;

/// Detect \p get_nec() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * vector_double::size_type get_nec() const;
 * @endcode
 * The \p get_nec() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_e_constraints
{
    template <typename U>
    using get_nec_t = decltype(std::declval<const U &>().get_nec());
    static const bool implementation_defined = std::is_same<vector_double::size_type, detected_t<get_nec_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_e_constraints<T>::value;

/// Detect \p get_nic() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * vector_double::size_type get_nic() const;
 * @endcode
 * The \p get_nic() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_i_constraints
{
    template <typename U>
    using get_nic_t = decltype(std::declval<const U &>().get_nic());
    static const bool implementation_defined = std::is_same<vector_double::size_type, detected_t<get_nic_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_i_constraints<T>::value;

/// Detect \p get_nix() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * vector_double::size_type get_nix() const;
 * @endcode
 * The \p get_nix() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_integer_part
{
    template <typename U>
    using get_nix_t = decltype(std::declval<const U &>().get_nix());
    static const bool implementation_defined = std::is_same<vector_double::size_type, detected_t<get_nix_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_integer_part<T>::value;

/// Detect \p gradient() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * vector_double gradient(const vector_double &) const;
 * @endcode
 * The \p gradient() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_gradient
{
    template <typename U>
    using gradient_t = decltype(std::declval<const U &>().gradient(std::declval<const vector_double &>()));
    static const bool implementation_defined = std::is_same<vector_double, detected_t<gradient_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_gradient<T>::value;

/// Detect \p has_gradient() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * bool has_gradient() const;
 * @endcode
 * The \p has_gradient() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class override_has_gradient
{
    template <typename U>
    using has_gradient_t = decltype(std::declval<const U &>().has_gradient());
    static const bool implementation_defined = std::is_same<bool, detected_t<has_gradient_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_gradient<T>::value;

/// Detect \p gradient_sparsity() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * sparsity_pattern gradient_sparsity() const;
 * @endcode
 * The \p gradient_sparsity() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_gradient_sparsity
{
    template <typename U>
    using gradient_sparsity_t = decltype(std::declval<const U &>().gradient_sparsity());
    static const bool implementation_defined
        = std::is_same<sparsity_pattern, detected_t<gradient_sparsity_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_gradient_sparsity<T>::value;

/// Detect \p hessians() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * std::vector<vector_double> hessians(const vector_double &) const;
 * @endcode
 * The \p hessians() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_hessians
{
    template <typename U>
    using hessians_t = decltype(std::declval<const U &>().hessians(std::declval<const vector_double &>()));
    static const bool implementation_defined
        = std::is_same<std::vector<vector_double>, detected_t<hessians_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_hessians<T>::value;

/// Detect \p has_hessians() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * bool has_hessians() const;
 * @endcode
 * The \p has_hessians() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class override_has_hessians
{
    template <typename U>
    using has_hessians_t = decltype(std::declval<const U &>().has_hessians());
    static const bool implementation_defined = std::is_same<bool, detected_t<has_hessians_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_hessians<T>::value;

/// Detect \p hessians_sparsity() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * std::vector<sparsity_pattern> hessians_sparsity() const;
 * @endcode
 * The \p hessians_sparsity() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class has_hessians_sparsity
{
    template <typename U>
    using hessians_sparsity_t = decltype(std::declval<const U &>().hessians_sparsity());
    static const bool implementation_defined
        = std::is_same<std::vector<sparsity_pattern>, detected_t<hessians_sparsity_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_hessians_sparsity<T>::value;

/// Detect \p has_gradient_sparsity() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * bool has_gradient_sparsity() const;
 * @endcode
 * The \p has_gradient_sparsity() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class override_has_gradient_sparsity
{
    template <typename U>
    using has_gradient_sparsity_t = decltype(std::declval<const U &>().has_gradient_sparsity());
    static const bool implementation_defined = std::is_same<bool, detected_t<has_gradient_sparsity_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_gradient_sparsity<T>::value;

/// Detect \p has_hessians_sparsity() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * bool has_hessians_sparsity() const;
 * @endcode
 * The \p has_hessians_sparsity() method is part of the interface for the definition of a problem
 * (see pagmo::problem).
 */
template <typename T>
class override_has_hessians_sparsity
{
    template <typename U>
    using has_hessians_sparsity_t = decltype(std::declval<const U &>().has_hessians_sparsity());
    static const bool implementation_defined = std::is_same<bool, detected_t<has_hessians_sparsity_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_hessians_sparsity<T>::value;

// Detect the batch_fitness() member function.
template <typename T>
class has_batch_fitness
{
    template <typename U>
    using batch_fitness_t = decltype(std::declval<const U &>().batch_fitness(std::declval<const vector_double &>()));
    static const bool implementation_defined = std::is_same<vector_double, detected_t<batch_fitness_t, T>>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_batch_fitness<T>::value;

// Detect the has_batch_fitness() member function.
template <typename T>
class override_has_batch_fitness
{
    template <typename U>
    using has_batch_fitness_t = decltype(std::declval<const U &>().has_batch_fitness());
    static const bool implementation_defined = std::is_same<bool, detected_t<has_batch_fitness_t, T>>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_batch_fitness<T>::value;

namespace detail
{

// Specialise this to true in order to disable all the UDP checks and mark a type
// as a UDP regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python problems.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_udp_checks : std::false_type {
};
} // namespace detail

/// Detect user-defined problems (UDP).
/**
 * This type trait will be \p true if \p T is not cv/reference qualified, it is destructible, default, copy and move
 * constructible, and if it satisfies the pagmo::has_fitness and pagmo::has_bounds type traits.
 *
 * Types satisfying this type trait can be used as user-defined problems (UDP) in pagmo::problem.
 */
template <typename T>
class is_udp
{
    static const bool implementation_defined
        = detail::disjunction<detail::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                                  std::is_copy_constructible<T>, std::is_move_constructible<T>,
                                                  std::is_destructible<T>, has_fitness<T>, has_bounds<T>>,
                              detail::disable_udp_checks<T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udp<T>::value;

namespace detail
{

// Helper to check that the problem bounds are valid. This will throw if the bounds
// are invalid because of:
// - the bounds size is zero,
// - inconsistent lengths of the vectors,
// - nans in the bounds,
// - lower bounds greater than upper bounds.
// - integer part larger than bounds size
// - integer bounds not integers
PAGMO_DLL_PUBLIC void check_problem_bounds(const std::pair<vector_double, vector_double> &bounds,
                                           vector_double::size_type nix = 0u);

PAGMO_DLL_PUBLIC sparsity_pattern dense_hessian(vector_double::size_type);

PAGMO_DLL_PUBLIC std::vector<sparsity_pattern> dense_hessians(vector_double::size_type, vector_double::size_type);

PAGMO_DLL_PUBLIC sparsity_pattern dense_gradient(vector_double::size_type, vector_double::size_type);

struct PAGMO_DLL_PUBLIC_INLINE_CLASS prob_inner_base {
    virtual ~prob_inner_base() {}
    virtual std::unique_ptr<prob_inner_base> clone() const = 0;
    virtual vector_double fitness(const vector_double &) const = 0;
    virtual vector_double batch_fitness(const vector_double &) const = 0;
    virtual bool has_batch_fitness() const = 0;
    virtual vector_double gradient(const vector_double &) const = 0;
    virtual bool has_gradient() const = 0;
    virtual sparsity_pattern gradient_sparsity() const = 0;
    virtual bool has_gradient_sparsity() const = 0;
    virtual std::vector<vector_double> hessians(const vector_double &) const = 0;
    virtual bool has_hessians() const = 0;
    virtual std::vector<sparsity_pattern> hessians_sparsity() const = 0;
    virtual bool has_hessians_sparsity() const = 0;
    virtual vector_double::size_type get_nobj() const = 0;
    virtual std::pair<vector_double, vector_double> get_bounds() const = 0;
    virtual vector_double::size_type get_nec() const = 0;
    virtual vector_double::size_type get_nic() const = 0;
    virtual vector_double::size_type get_nix() const = 0;
    virtual void set_seed(unsigned) = 0;
    virtual bool has_set_seed() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual thread_safety get_thread_safety() const = 0;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
struct PAGMO_DLL_PUBLIC_INLINE_CLASS prob_inner final : prob_inner_base {
    // We just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit prob_inner(const T &x) : m_value(x) {}
    explicit prob_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of problem.
    virtual std::unique_ptr<prob_inner_base> clone() const override final
    {
        return detail::make_unique<prob_inner>(m_value);
    }
    // Mandatory methods.
    virtual vector_double fitness(const vector_double &dv) const override final
    {
        return m_value.fitness(dv);
    }
    virtual std::pair<vector_double, vector_double> get_bounds() const override final
    {
        return m_value.get_bounds();
    }
    // optional methods
    virtual vector_double batch_fitness(const vector_double &dv) const override final
    {
        return batch_fitness_impl(m_value, dv);
    }
    virtual bool has_batch_fitness() const override final
    {
        return has_batch_fitness_impl(m_value);
    }
    virtual vector_double::size_type get_nobj() const override final
    {
        return get_nobj_impl(m_value);
    }
    virtual vector_double gradient(const vector_double &dv) const override final
    {
        return gradient_impl(m_value, dv);
    }
    virtual bool has_gradient() const override final
    {
        return has_gradient_impl(m_value);
    }
    virtual sparsity_pattern gradient_sparsity() const override final
    {
        return gradient_sparsity_impl(m_value);
    }
    virtual bool has_gradient_sparsity() const override final
    {
        return has_gradient_sparsity_impl(m_value);
    }
    virtual std::vector<vector_double> hessians(const vector_double &dv) const override final
    {
        return hessians_impl(m_value, dv);
    }
    virtual bool has_hessians() const override final
    {
        return has_hessians_impl(m_value);
    }
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final
    {
        return hessians_sparsity_impl(m_value);
    }
    virtual bool has_hessians_sparsity() const override final
    {
        return has_hessians_sparsity_impl(m_value);
    }
    virtual vector_double::size_type get_nec() const override final
    {
        return get_nec_impl(m_value);
    }
    virtual vector_double::size_type get_nic() const override final
    {
        return get_nic_impl(m_value);
    }
    virtual vector_double::size_type get_nix() const override final
    {
        return get_nix_impl(m_value);
    }
    virtual void set_seed(unsigned seed) override final
    {
        set_seed_impl(m_value, seed);
    }
    virtual bool has_set_seed() const override final
    {
        return has_set_seed_impl(m_value);
    }
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
    template <typename U, enable_if_t<pagmo::has_batch_fitness<U>::value, int> = 0>
    static vector_double batch_fitness_impl(const U &value, const vector_double &dv)
    {
        return value.batch_fitness(dv);
    }
    template <typename U, enable_if_t<!pagmo::has_batch_fitness<U>::value, int> = 0>
    [[noreturn]] static vector_double batch_fitness_impl(const U &value, const vector_double &)
    {
        pagmo_throw(not_implemented_error,
                    "The batch_fitness() method has been invoked, but it is not implemented in a UDP of type '"
                        + get_name_impl(value) + "'");
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_batch_fitness<U>, pagmo::override_has_batch_fitness<U>>::value,
                          int> = 0>
    static bool has_batch_fitness_impl(const U &p)
    {
        return p.has_batch_fitness();
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_batch_fitness<U>,
                                              detail::negation<pagmo::override_has_batch_fitness<U>>>::value,
                          int> = 0>
    static bool has_batch_fitness_impl(const U &)
    {
        return true;
    }
    template <typename U, enable_if_t<!pagmo::has_batch_fitness<U>::value, int> = 0>
    static bool has_batch_fitness_impl(const U &)
    {
        return false;
    }
    template <typename U, enable_if_t<has_get_nobj<U>::value, int> = 0>
    static vector_double::size_type get_nobj_impl(const U &value)
    {
        return value.get_nobj();
    }
    template <typename U, enable_if_t<!has_get_nobj<U>::value, int> = 0>
    static vector_double::size_type get_nobj_impl(const U &)
    {
        return 1u;
    }
    template <typename U, enable_if_t<pagmo::has_gradient<U>::value, int> = 0>
    static vector_double gradient_impl(const U &value, const vector_double &dv)
    {
        return value.gradient(dv);
    }
    template <typename U, enable_if_t<!pagmo::has_gradient<U>::value, int> = 0>
    [[noreturn]] static vector_double gradient_impl(const U &value, const vector_double &)
    {
        pagmo_throw(not_implemented_error,
                    "The gradient has been requested, but it is not implemented in a UDP of type '"
                        + get_name_impl(value) + "'");
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_gradient<U>, pagmo::override_has_gradient<U>>::value, int> = 0>
    static bool has_gradient_impl(const U &p)
    {
        return p.has_gradient();
    }
    template <typename U, enable_if_t<detail::conjunction<pagmo::has_gradient<U>,
                                                          detail::negation<pagmo::override_has_gradient<U>>>::value,
                                      int> = 0>
    static bool has_gradient_impl(const U &)
    {
        return true;
    }
    template <typename U, enable_if_t<!pagmo::has_gradient<U>::value, int> = 0>
    static bool has_gradient_impl(const U &)
    {
        return false;
    }
    template <typename U, enable_if_t<pagmo::has_gradient_sparsity<U>::value, int> = 0>
    static sparsity_pattern gradient_sparsity_impl(const U &p)
    {
        return p.gradient_sparsity();
    }
    template <typename U, enable_if_t<!pagmo::has_gradient_sparsity<U>::value, int> = 0>
    [[noreturn]] static sparsity_pattern gradient_sparsity_impl(const U &) // LCOV_EXCL_LINE
    {
        // NOTE: we should never end up here. gradient_sparsity() is called only if m_has_gradient_sparsity
        // in the problem is set to true, and m_has_gradient_sparsity is unconditionally false if the UDP
        // does not implement gradient_sparsity() (see implementation of the three overloads below).
        assert(false); // LCOV_EXCL_LINE
        throw;
    }
    template <typename U, enable_if_t<detail::conjunction<pagmo::has_gradient_sparsity<U>,
                                                          pagmo::override_has_gradient_sparsity<U>>::value,
                                      int> = 0>
    static bool has_gradient_sparsity_impl(const U &p)
    {
        return p.has_gradient_sparsity();
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_gradient_sparsity<U>,
                                              detail::negation<pagmo::override_has_gradient_sparsity<U>>>::value,
                          int> = 0>
    static bool has_gradient_sparsity_impl(const U &)
    {
        return true;
    }
    template <typename U, enable_if_t<!pagmo::has_gradient_sparsity<U>::value, int> = 0>
    static bool has_gradient_sparsity_impl(const U &)
    {
        return false;
    }
    template <typename U, enable_if_t<pagmo::has_hessians<U>::value, int> = 0>
    static std::vector<vector_double> hessians_impl(const U &value, const vector_double &dv)
    {
        return value.hessians(dv);
    }
    template <typename U, enable_if_t<!pagmo::has_hessians<U>::value, int> = 0>
    [[noreturn]] static std::vector<vector_double> hessians_impl(const U &value, const vector_double &)
    {
        pagmo_throw(not_implemented_error,
                    "The hessians have been requested, but they are not implemented in a UDP of type '"
                        + get_name_impl(value) + "'");
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_hessians<U>, pagmo::override_has_hessians<U>>::value, int> = 0>
    static bool has_hessians_impl(const U &p)
    {
        return p.has_hessians();
    }
    template <typename U, enable_if_t<detail::conjunction<pagmo::has_hessians<U>,
                                                          detail::negation<pagmo::override_has_hessians<U>>>::value,
                                      int> = 0>
    static bool has_hessians_impl(const U &)
    {
        return true;
    }
    template <typename U, enable_if_t<!pagmo::has_hessians<U>::value, int> = 0>
    static bool has_hessians_impl(const U &)
    {
        return false;
    }
    template <typename U, enable_if_t<pagmo::has_hessians_sparsity<U>::value, int> = 0>
    static std::vector<sparsity_pattern> hessians_sparsity_impl(const U &value)
    {
        return value.hessians_sparsity();
    }
    template <typename U, enable_if_t<!pagmo::has_hessians_sparsity<U>::value, int> = 0>
    [[noreturn]] static std::vector<sparsity_pattern> hessians_sparsity_impl(const U &) // LCOV_EXCL_LINE
    {
        // NOTE: we should never end up here. hessians_sparsity() is called only if m_has_hessians_sparsity
        // in the problem is set to true, and m_has_hessians_sparsity is unconditionally false if the UDP
        // does not implement hessians_sparsity() (see implementation of the three overloads below).
        assert(false); // LCOV_EXCL_LINE
        throw;
    }
    template <typename U, enable_if_t<detail::conjunction<pagmo::has_hessians_sparsity<U>,
                                                          pagmo::override_has_hessians_sparsity<U>>::value,
                                      int> = 0>
    static bool has_hessians_sparsity_impl(const U &p)
    {
        return p.has_hessians_sparsity();
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_hessians_sparsity<U>,
                                              detail::negation<pagmo::override_has_hessians_sparsity<U>>>::value,
                          int> = 0>
    static bool has_hessians_sparsity_impl(const U &)
    {
        return true;
    }
    template <typename U, enable_if_t<!pagmo::has_hessians_sparsity<U>::value, int> = 0>
    static bool has_hessians_sparsity_impl(const U &)
    {
        return false;
    }
    template <typename U, enable_if_t<has_e_constraints<U>::value, int> = 0>
    static vector_double::size_type get_nec_impl(const U &value)
    {
        return value.get_nec();
    }
    template <typename U, enable_if_t<!has_e_constraints<U>::value, int> = 0>
    static vector_double::size_type get_nec_impl(const U &)
    {
        return 0u;
    }
    template <typename U, enable_if_t<has_i_constraints<U>::value, int> = 0>
    static vector_double::size_type get_nic_impl(const U &value)
    {
        return value.get_nic();
    }
    template <typename U, enable_if_t<!has_i_constraints<U>::value, int> = 0>
    static vector_double::size_type get_nic_impl(const U &)
    {
        return 0u;
    }
    template <typename U, enable_if_t<has_integer_part<U>::value, int> = 0>
    static vector_double::size_type get_nix_impl(const U &value)
    {
        return value.get_nix();
    }
    template <typename U, enable_if_t<!has_integer_part<U>::value, int> = 0>
    static vector_double::size_type get_nix_impl(const U &)
    {
        return 0u;
    }
    template <typename U, enable_if_t<pagmo::has_set_seed<U>::value, int> = 0>
    static void set_seed_impl(U &value, unsigned seed)
    {
        value.set_seed(seed);
    }
    template <typename U, enable_if_t<!pagmo::has_set_seed<U>::value, int> = 0>
    [[noreturn]] static void set_seed_impl(U &value, unsigned)
    {
        pagmo_throw(not_implemented_error,
                    "The set_seed() method has been invoked, but it is not implemented in a UDP of type '"
                        + get_name_impl(value) + "'");
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_set_seed<U>, override_has_set_seed<U>>::value, int> = 0>
    static bool has_set_seed_impl(const U &p)
    {
        return p.has_set_seed();
    }
    template <
        typename U,
        enable_if_t<detail::conjunction<pagmo::has_set_seed<U>, detail::negation<override_has_set_seed<U>>>::value,
                    int> = 0>
    static bool has_set_seed_impl(const U &)
    {
        return true;
    }
    template <typename U, enable_if_t<!pagmo::has_set_seed<U>::value, int> = 0>
    static bool has_set_seed_impl(const U &)
    {
        return false;
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
        detail::archive(ar, boost::serialization::base_object<prob_inner_base>(*this), m_value);
    }
    T m_value;
};

} // namespace detail

} // namespace pagmo

namespace boost
{

// NOTE: in some earlier versions of Boost (i.e., at least up to 1.67)
// the is_virtual_base_of type trait, used by the Boost serialization library, fails
// with a compile time error
// if a class is declared final. Thus, we provide a specialised implementation of
// this type trait to work around the issue. See:
// https://www.boost.org/doc/libs/1_52_0/libs/type_traits/doc/html/boost_typetraits/reference/is_virtual_base_of.html
// https://stackoverflow.com/questions/18982064/boost-serialization-of-base-class-of-final-subclass-error
// We never use virtual inheritance, thus the specialisation is always false.
template <typename T>
struct is_virtual_base_of<pagmo::detail::prob_inner_base, pagmo::detail::prob_inner<T>> : false_type {
};

} // namespace boost

namespace pagmo
{

// Fwd declare for the declarations below.
class PAGMO_DLL_PUBLIC problem;

// Streaming operator
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const problem &);

namespace detail
{

// These are internal private helpers which are used both in problem
// and elsewhere. Hence, decouple them from the problem class and provide
// them as free functions.
PAGMO_DLL_PUBLIC void prob_check_dv(const problem &, const double *, vector_double::size_type);
PAGMO_DLL_PUBLIC void prob_check_fv(const problem &, const double *, vector_double::size_type);
PAGMO_DLL_PUBLIC vector_double prob_invoke_mem_batch_fitness(const problem &, const vector_double &);

} // namespace detail

/// Problem class.
/**
 * \image html prob_no_text.png
 *
 * This class represents a generic *mathematical programming* or *evolutionary optimization* problem in the form:
 * \f[
 * \begin{array}{rl}
 * \mbox{find:}      & \mathbf {lb} \le \mathbf x \le \mathbf{ub}\\
 * \mbox{to minimize: } & \mathbf f(\mathbf x, s) \in \mathbb R^{n_{obj}}\\
 * \mbox{subject to:} & \mathbf {c}_e(\mathbf x, s) = 0 \\
 *                    & \mathbf {c}_i(\mathbf x, s) \le 0
 * \end{array}
 * \f]
 *
 * where \f$\mathbf x \in \mathbb R^{n_{cx}} \times  \mathbb Z^{n_{ix}}\f$ is called *decision vector* or
 * *chromosome*, and is made of \f$n_{cx}\f$ real numbers and \f$n_{ix}\f$ integers (all represented as doubles). The
 * total problem dimension is then indicated with \f$n_x = n_{cx} + n_{ix}\f$. \f$\mathbf{lb}, \mathbf{ub} \in
 * \mathbb R^{n_{cx}} \times  \mathbb Z^{n_{ix}}\f$ are the *box-bounds*, \f$ \mathbf f: \mathbb R^{n_{cx}} \times
 * \mathbb Z^{n_{ix}} \rightarrow \mathbb R^{n_{obj}}\f$ define the *objectives*, \f$ \mathbf c_e:  \mathbb R^{n_{cx}}
 * \times  \mathbb Z^{n_{ix}} \rightarrow \mathbb R^{n_{ec}}\f$ are non linear *equality constraints*, and \f$ \mathbf
 * c_i:  \mathbb R^{n_{cx}} \times  \mathbb Z^{n_{ix}} \rightarrow \mathbb R^{n_{ic}}\f$ are non linear *inequality
 * constraints*. Note that the objectives and constraints may also depend from an added value \f$s\f$ seeding the
 * values of any number of stochastic variables. This allows also for stochastic programming tasks to be represented by
 * this class. A tolerance is also considered for all constraints and set, by default, to zero. It can be modified
 * via the problem::set_c_tol() method.
 *
 * In order to define an optimizaztion problem in pagmo, the user must first define a class
 * (or a struct) whose methods describe the properties of the problem and allow to compute
 * the objective function, the gradient, the constraints, etc. In pagmo, we refer to such
 * a class as a **user-defined problem**, or UDP for short. Once defined and instantiated,
 * a UDP can then be used to construct an instance of this class, pagmo::problem, which
 * provides a generic interface to optimization problems.
 *
 * Every UDP must implement at least the following two methods:
 * @code{.unparsed}
 * vector_double fitness(const vector_double &) const;
 * std::pair<vector_double, vector_double> get_bounds() const;
 * @endcode
 *
 * The <tt>%fitness()</tt> method is expected to return the fitness of the input decision vector (
 * concatenating the objectives, the equality and the inequality constraints), while
 * <tt>%get_bounds()</tt> is expected to return the box bounds of the problem,
 * \f$(\mathbf{lb}, \mathbf{ub})\f$, which also implicitly define the dimension of the problem.
 * The <tt>%fitness()</tt> and <tt>%get_bounds()</tt> methods of the UDP are accessible from the corresponding
 * problem::fitness() and problem::get_bounds() methods (see their documentation for details).
 * In addition to providing the above methods, a UDP must also be default, copy and move constructible.
 *
 * The two mandatory methods above allow to define a continuous, single objective, deterministic, derivative-free,
 * unconstrained optimization problem. In order to consider more complex cases, the UDP may implement one or more of the
 * following methods:
 * @code{.unparsed}
 * vector_double::size_type get_nobj() const;
 * vector_double::size_type get_nec() const;
 * vector_double::size_type get_nic() const;
 * vector_double::size_type get_nix() const;
 * vector_double batch_fitness(const vector_double &) const;
 * bool has_batch_fitness() const;
 * bool has_gradient() const;
 * vector_double gradient(const vector_double &) const;
 * bool has_gradient_sparsity() const;
 * sparsity_pattern gradient_sparsity() const;
 * bool has_hessians() const;
 * std::vector<vector_double> hessians(const vector_double &) const;
 * bool has_hessians_sparsity() const;
 * std::vector<sparsity_pattern> hessians_sparsity() const;
 * bool has_set_seed() const;
 * void set_seed(unsigned);
 * std::string get_name() const;
 * std::string get_extra_info() const;
 * thread_safety get_thread_safety() const;
 * @endcode
 *
 * See the documentation of the corresponding methods in this class for details on how the optional
 * methods in the UDP are used by pagmo::problem.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    The only operations allowed on a moved-from :cpp:class:`pagmo::problem` are destruction,
 *    assignment, and the invocation of the :cpp:func:`~pagmo::problem::is_valid()` member function.
 *    Any other operation will result in undefined behaviour.
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC problem
{
    // Make friends with the streaming operator, which needs access
    // to the internals.
    friend PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const problem &);

    // Enable the generic ctor only if T is not a problem (after removing
    // const/reference qualifiers), and if T is a udp.
    template <typename T>
    using generic_ctor_enabler = enable_if_t<
        detail::conjunction<detail::negation<std::is_same<problem, uncvref_t<T>>>, is_udp<uncvref_t<T>>>::value, int>;

public:
    // Default constructor.
    problem();

private:
    void generic_ctor_impl();

public:
    /// Constructor from a user-defined problem of type \p T
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is not enabled if, after the removal of cv and reference qualifiers,
     *    ``T`` is of type :cpp:class:`pagmo::problem` (that is, this constructor does not compete with the copy/move
     *    constructors of :cpp:class:`pagmo::problem`), or if ``T`` does not satisfy :cpp:class:`pagmo::is_udp`.
     *
     * \endverbatim
     *
     * This constructor will construct a pagmo::problem from the UDP (user-defined problem) \p x of type \p T. In order
     * for the construction to be successful, the UDP must implement a minimal set of methods,
     * as described in the documentation of pagmo::problem. The constructor will examine the properties of \p x and
     * store them as data members of \p this.
     *
     * @param x the UDP.
     *
     * @throws std::invalid_argument in the following cases:
     * - the number of objectives of the UDP is zero,
     * - the number of objectives, equality or inequality constraints is larger than an implementation-defined value,
     * - the problem bounds are invalid (e.g., they contain NaNs, the dimensionality of the lower bounds is
     *   different from the dimensionality of the upper bounds, the bounds relative to the integer part are not
     *   integers, etc. - note that infinite bounds are allowed),
     * - the <tt>%gradient_sparsity()</tt> and <tt>%hessians_sparsity()</tt> methods of the UDP fail basic sanity checks
     *   (e.g., they return vectors with repeated indices, they contain indices exceeding the problem's dimensions,
     *   etc.).
     * - the integer part of the problem is larger than the problem size.
     * @throws unspecified any exception thrown by methods of the UDP invoked during construction or by memory errors
     * in strings and standard containers.
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit problem(T &&x)
        : m_ptr(detail::make_unique<detail::prob_inner<uncvref_t<T>>>(std::forward<T>(x))), m_fevals(0u), m_gevals(0u),
          m_hevals(0u)
    {
        generic_ctor_impl();
    }

    // Copy constructor.
    problem(const problem &);
    // Move constructor.
    problem(problem &&) noexcept;
    // Move assignment operator
    problem &operator=(problem &&) noexcept;
    // Copy assignment operator
    problem &operator=(const problem &);
    /// Assignment from a user-defined problem of type \p T
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This operator is not enabled if, after the removal of cv and reference qualifiers,
     *    ``T`` is of type :cpp:class:`pagmo::problem` (that is, this operator does not compete with the copy/move
     *    assignment operators of :cpp:class:`pagmo::problem`), or if ``T`` does not satisfy :cpp:class:`pagmo::is_udp`.
     *
     * \endverbatim
     *
     * This operator will set the internal UDP to ``x`` by constructing a pagmo::problem from ``x``, and then
     * move-assigning the result to ``this``.
     *
     * @param x the UDP.
     *
     * @return a reference to ``this``.
     *
     * @throws unspecified any exception thrown by the constructor from UDP.
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    problem &operator=(T &&x)
    {
        return (*this) = problem(std::forward<T>(x));
    }

    /// Extract a const pointer to the UDP used for construction.
    /**
     * This method will extract a const pointer to the internal instance of the UDP. If \p T is not the same type
     * as the UDP used during construction (after removal of cv and reference qualifiers), this method will
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
     * @return a const pointer to the internal UDP, or \p nullptr
     * if \p T does not correspond exactly to the original UDP type used
     * in the constructor.
     */
    template <typename T>
    const T *extract() const noexcept
    {
        auto p = dynamic_cast<const detail::prob_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    /// Extract a pointer to the UDP used for construction.
    /**
     * This method will extract a pointer to the internal instance of the UDP. If \p T is not the same type
     * as the UDP used during construction (after removal of cv and reference qualifiers), this method will
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
     *    methods on the internal UDP instance. Assigning a new UDP via this pointer is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a pointer to the internal UDP, or \p nullptr
     * if \p T does not correspond exactly to the original UDP type used
     * in the constructor.
     */
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::prob_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    /// Check if the UDP used for construction is of type \p T.
    /**
     * @return \p true if the UDP used for construction is of type \p T, \p false otherwise.
     */
    template <typename T>
    bool is() const noexcept
    {
        return extract<T>() != nullptr;
    }

    // Fitness.
    vector_double fitness(const vector_double &) const;

private:
#if !defined(PAGMO_DOXYGEN_INVOKED)
    // Make friends with the batch_fitness() invocation helper.
    friend PAGMO_DLL_PUBLIC vector_double detail::prob_invoke_mem_batch_fitness(const problem &, const vector_double &);
#endif

public:
    // Batch fitness.
    vector_double batch_fitness(const vector_double &) const;

    /// Check if the UDP is capable of fitness evaluation in batch mode.
    /**
     * This method will return \p true if the UDP is capable of fitness evaluation in batch mode, \p false otherwise.
     *
     * \verbatim embed:rst:leading-asterisk
     * The batch fitness evaluation capability of the UDP is determined as follows:
     *
     * * if the UDP does not satisfy :cpp:class:`pagmo::has_batch_fitness`, then this method will always return
     *   ``false``;
     * * if the UDP satisfies :cpp:class:`pagmo::has_batch_fitness` but it does not satisfy
     *   :cpp:class:`pagmo::override_has_batch_fitness`, then this method will always return ``true``;
     * * if the UDP satisfies both :cpp:class:`pagmo::has_batch_fitness` and
     *   :cpp:class:`pagmo::override_has_batch_fitness`, then this method will return the output of the
     *   ``has_batch_fitness()`` method of the UDP.
     *
     * \endverbatim
     *
     * @return a flag signalling the availability of fitness evaluation in batch mode in the UDP.
     */
    bool has_batch_fitness() const
    {
        return m_has_batch_fitness;
    }

    // Gradient.
    vector_double gradient(const vector_double &) const;

    /// Check if the gradient is available in the UDP.
    /**
     * This method will return \p true if the gradient is available in the UDP, \p false otherwise.
     *
     * The availability of the gradient is determined as follows:
     * - if the UDP does not satisfy pagmo::has_gradient, then this method will always return \p false;
     * - if the UDP satisfies pagmo::has_gradient but it does not satisfy pagmo::override_has_gradient,
     *   then this method will always return \p true;
     * - if the UDP satisfies both pagmo::has_gradient and pagmo::override_has_gradient,
     *   then this method will return the output of the <tt>%has_gradient()</tt> method of the UDP.
     *
     * @return a flag signalling the availability of the gradient in the UDP.
     */
    bool has_gradient() const
    {
        return m_has_gradient;
    }

    // Gradient sparsity pattern.
    sparsity_pattern gradient_sparsity() const;

    /// Check if the gradient sparsity is available in the UDP.
    /**
     * This method will return \p true if the gradient sparsity is available in the UDP, \p false otherwise.
     *
     * The availability of the gradient sparsity is determined as follows:
     * - if the UDP does not satisfy pagmo::has_gradient_sparsity, then this method will always return \p false;
     * - if the UDP satisfies pagmo::has_gradient_sparsity but it does not satisfy
     *   pagmo::override_has_gradient_sparsity, then this method will always return \p true;
     * - if the UDP satisfies both pagmo::has_gradient_sparsity and pagmo::override_has_gradient_sparsity,
     *   then this method will return the output of the <tt>%has_gradient_sparsity()</tt> method of the UDP.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    Regardless of what this method returns, the :cpp:func:`problem::gradient_sparsity()` method will always return
     *    a sparsity pattern: if the UDP does not provide the gradient sparsity, PaGMO will assume that the sparsity
     *    pattern of the gradient is dense. See :cpp:func:`problem::gradient_sparsity()` for more details.
     *
     * \endverbatim
     *
     * @return a flag signalling the availability of the gradient sparsity in the UDP.
     */
    bool has_gradient_sparsity() const
    {
        return m_has_gradient_sparsity;
    }

    // Hessians.
    std::vector<vector_double> hessians(const vector_double &) const;

    /// Check if the hessians are available in the UDP.
    /**
     * This method will return \p true if the hessians are available in the UDP, \p false otherwise.
     *
     * The availability of the hessians is determined as follows:
     * - if the UDP does not satisfy pagmo::has_hessians, then this method will always return \p false;
     * - if the UDP satisfies pagmo::has_hessians but it does not satisfy pagmo::override_has_hessians,
     *   then this method will always return \p true;
     * - if the UDP satisfies both pagmo::has_hessians and pagmo::override_has_hessians,
     *   then this method will return the output of the <tt>%has_hessians()</tt> method of the UDP.
     *
     * @return a flag signalling the availability of the hessians in the UDP.
     */
    bool has_hessians() const
    {
        return m_has_hessians;
    }

    // Hessians sparsity pattern.
    std::vector<sparsity_pattern> hessians_sparsity() const;

    /// Check if the hessians sparsity is available in the UDP.
    /**
     * This method will return \p true if the hessians sparsity is available in the UDP, \p false otherwise.
     *
     * The availability of the hessians sparsity is determined as follows:
     * - if the UDP does not satisfy pagmo::has_hessians_sparsity, then this method will always return \p false;
     * - if the UDP satisfies pagmo::has_hessians_sparsity but it does not satisfy
     *   pagmo::override_has_hessians_sparsity, then this method will always return \p true;
     * - if the UDP satisfies both pagmo::has_hessians_sparsity and pagmo::override_has_hessians_sparsity,
     *   then this method will return the output of the <tt>%has_hessians_sparsity()</tt> method of the UDP.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    Regardless of what this method returns, the :cpp:func:`problem::hessians_sparsity()` method will always return
     *    a vector of sparsity patterns: if the UDP does not provide the hessians sparsity, PaGMO will assume that the
     *    sparsity pattern of the hessians is dense. See :cpp:func:`problem::hessians_sparsity()` for more details.
     *
     * \endverbatim
     *
     * @return a flag signalling the availability of the hessians sparsity in the UDP.
     */
    bool has_hessians_sparsity() const
    {
        return m_has_hessians_sparsity;
    }

    /// Number of objectives.
    /**
     * This method will return \f$ n_{obj}\f$, the number of objectives of the optimization
     * problem. If the UDP satisfies the pagmo::has_get_nobj type traits, then the output of
     * its <tt>%get_nobj()</tt> method will be returned. Otherwise, this method will return 1.
     *
     * @return the number of objectives of the problem.
     */
    vector_double::size_type get_nobj() const
    {
        return m_nobj;
    }

    /// Dimension.
    /**
     * @return \f$ n_{x}\f$, the dimension of the problem as established
     * by the length of the bounds returned by problem::get_bounds().
     */
    vector_double::size_type get_nx() const
    {
        return m_lb.size();
    }

    /// Integer Dimension.
    /**
     * This method will return \f$ n_{ix} \f$, the dimension of the integer part of the problem.
     * If the UDP satisfies pagmo::has_integer_part, then the output of
     * its <tt>%get_nix()</tt> method will be returned. Otherwise, this method will return 0.
     *
     * @return \f$ n_{ix}\f$, the integer dimension of the problem.
     */
    vector_double::size_type get_nix() const
    {
        return m_nix;
    }

    /// Continuous Dimension.
    /**
     * @return \f$ n_{cx}\f$, the continuous dimension of the problem as established
     * by the relation \f$n_{cx} = n_{x} - n_{ix} \f$.
     *
     * @return \f$ n_{cx}\f$, the continuous dimension of the problem.
     */
    vector_double::size_type get_ncx() const
    {
        return get_nx() - m_nix;
    }

    /// Fitness dimension.
    /**
     * @return \f$ n_{f}\f$, the dimension of the fitness, which is the
     * sum of \f$n_{obj}\f$, \f$n_{ec}\f$ and \f$n_{ic}\f$
     */
    vector_double::size_type get_nf() const
    {
        return m_nobj + m_nic + m_nec;
    }

    // Box-bounds.
    std::pair<vector_double, vector_double> get_bounds() const;

    /// Lower bounds.
    /**
     * @return a const reference to the vector of lower box bounds for this problem.
     */
    const vector_double &get_lb() const
    {
        return m_lb;
    }

    /// Upper bounds.
    /**
     * @return a const reference to the vector of upper box bounds for this problem.
     */
    const vector_double &get_ub() const
    {
        return m_ub;
    }

    /// Number of equality constraints.
    /**
     * This method will return \f$ n_{ec} \f$, the number of equality constraints of the problem.
     * If the UDP satisfies pagmo::has_e_constraints, then the output of
     * its <tt>%get_nec()</tt> method will be returned. Otherwise, this method will return 0.
     *
     * @return the number of equality constraints of the problem.
     */
    vector_double::size_type get_nec() const
    {
        return m_nec;
    }

    /// Number of inequality constraints.
    /**
     * This method will return \f$ n_{ic} \f$, the number of inequality constraints of the problem.
     * If the UDP satisfies pagmo::has_i_constraints, then the output of
     * its <tt>%get_nic()</tt> method will be returned. Otherwise, this method will return 0.
     *
     * @return the number of inequality constraints of the problem.
     */
    vector_double::size_type get_nic() const
    {
        return m_nic;
    }

    // Set the constraint tolerance (from a vector of doubles).
    void set_c_tol(const vector_double &);
    // Set the constraint tolerance (from a single double value).
    void set_c_tol(double);
    /// Get the constraint tolerance.
    /**
     * This method will return a vector of dimension \f$n_{ec} + n_{ic}\f$ containing tolerances to
     * be used when checking constraint feasibility. The constraint tolerance is zero-filled upon problem
     * construction, and it can be set via problem::set_c_tol().
     *
     * @return a pagmo::vector_double containing the tolerances to use when
     * checking for constraint feasibility.
     */
    vector_double get_c_tol() const
    {
        return m_c_tol;
    }

    /// Total number of constraints
    /**
     * @return the sum of the output of get_nic() and get_nec() (i.e., the total number of constraints).
     */
    vector_double::size_type get_nc() const
    {
        return m_nec + m_nic;
    }

    /// Number of fitness evaluations.
    /**
     * Each time a call to problem::fitness() successfully completes, an internal counter is increased by one.
     * The counter is initialised to zero upon problem construction and it is never reset. Copy and move operations
     * copy the counter as well.
     *
     * @return the number of times problem::fitness() was successfully called.
     */
    unsigned long long get_fevals() const
    {
        return m_fevals.load(std::memory_order_relaxed);
    }

    /// Increment the number of fitness evaluations.
    /**
     * This method will increase the internal counter of fitness evaluations by \p n.
     *
     * @param n the amount by which the internal counter of fitness evaluations will be increased.
     */
    void increment_fevals(unsigned long long n) const
    {
        m_fevals.fetch_add(n, std::memory_order_relaxed);
    }

    /// Number of gradient evaluations.
    /**
     * Each time a call to problem::gradient() successfully completes, an internal counter is increased by one.
     * The counter is initialised to zero upon problem construction and it is never reset. Copy and move operations
     * copy the counter as well.
     *
     * @return the number of times problem::gradient() was successfully called.
     */
    unsigned long long get_gevals() const
    {
        return m_gevals.load(std::memory_order_relaxed);
    }

    /// Number of hessians evaluations.
    /**
     * Each time a call to problem::hessians() successfully completes, an internal counter is increased by one.
     * The counter is initialised to zero upon problem construction and it is never reset. Copy and move operations
     * copy the counter as well.
     *
     * @return the number of times problem::hessians() was successfully called.
     */
    unsigned long long get_hevals() const
    {
        return m_hevals.load(std::memory_order_relaxed);
    }

    // Set the seed for the stochastic variables.
    void set_seed(unsigned);

    // Feasibility of a decision vector.
    bool feasibility_x(const vector_double &) const;
    // Feasibility of a fitness vector.
    bool feasibility_f(const vector_double &) const;

    /// Check if a <tt>%set_seed()</tt> method is available in the UDP.
    /**
     * This method will return \p true if a <tt>%set_seed()</tt> method is available in the UDP, \p false otherwise.
     *
     * The availability of the a <tt>%set_seed()</tt> method is determined as follows:
     * - if the UDP does not satisfy pagmo::has_set_seed, then this method will always return \p false;
     * - if the UDP satisfies pagmo::has_set_seed but it does not satisfy pagmo::override_has_set_seed,
     *   then this method will always return \p true;
     * - if the UDP satisfies both pagmo::has_set_seed and pagmo::override_has_set_seed,
     *   then this method will return the output of the <tt>%has_set_seed()</tt> method of the UDP.
     *
     * @return a flag signalling the availability of the <tt>%set_seed()</tt> method in the UDP.
     */
    bool has_set_seed() const
    {
        return m_has_set_seed;
    }

    /// Alias for problem::has_set_seed().
    /**
     * @return the output of problem::has_set_seed().
     */
    bool is_stochastic() const
    {
        return has_set_seed();
    }

    /// Problem's name.
    /**
     * If the UDP satisfies pagmo::has_name, then this method will return the output of its <tt>%get_name()</tt> method.
     * Otherwise, an implementation-defined name based on the type of the UDP will be returned.
     *
     * @return the problem's name.
     *
     * @throws unspecified any exception thrown by copying an \p std::string object.
     */
    std::string get_name() const
    {
        return m_name;
    }

    // Problem's extra info.
    std::string get_extra_info() const;

    /// Problem's thread safety level.
    /**
     * If the UDP satisfies pagmo::has_get_thread_safety, then this method will return the output of its
     * <tt>%get_thread_safety()</tt> method. Otherwise, thread_safety::basic will be returned.
     * That is, pagmo assumes by default that is it safe to operate concurrently on distinct UDP instances.
     *
     * @return the thread safety level of the UDP.
     */
    thread_safety get_thread_safety() const
    {
        return m_thread_safety;
    }

    // Check if the problem is in a valid state.
    bool is_valid() const;

    /// Save to archive.
    /**
     * This method will save \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_ptr, m_fevals.load(std::memory_order_relaxed),
                           m_gevals.load(std::memory_order_relaxed), m_hevals.load(std::memory_order_relaxed), m_lb,
                           m_ub, m_nobj, m_nec, m_nic, m_nix, m_c_tol, m_has_batch_fitness, m_has_gradient,
                           m_has_gradient_sparsity, m_has_hessians, m_has_hessians_sparsity, m_has_set_seed, m_name,
                           m_gs_dim, m_hs_dim, m_thread_safety);
    }

    /// Load from archive.
    /**
     * This method will deserialize into \p this the content of \p ar.
     *
     * @param ar source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        // Deserialize in a separate object and move it in later, for exception safety.
        problem tmp_prob;
        unsigned long long fevals, gevals, hevals;
        detail::from_archive(ar, tmp_prob.m_ptr, fevals, gevals, hevals, tmp_prob.m_lb, tmp_prob.m_ub, tmp_prob.m_nobj,
                             tmp_prob.m_nec, tmp_prob.m_nic, tmp_prob.m_nix, tmp_prob.m_c_tol,
                             tmp_prob.m_has_batch_fitness, tmp_prob.m_has_gradient, tmp_prob.m_has_gradient_sparsity,
                             tmp_prob.m_has_hessians, tmp_prob.m_has_hessians_sparsity, tmp_prob.m_has_set_seed,
                             tmp_prob.m_name, tmp_prob.m_gs_dim, tmp_prob.m_hs_dim, tmp_prob.m_thread_safety);
        tmp_prob.m_fevals.store(fevals, std::memory_order_relaxed);
        tmp_prob.m_gevals.store(gevals, std::memory_order_relaxed);
        tmp_prob.m_hevals.store(hevals, std::memory_order_relaxed);
        *this = std::move(tmp_prob);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::prob_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::prob_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }

    void check_gradient_sparsity(const sparsity_pattern &) const;
    void check_hessians_sparsity(const std::vector<sparsity_pattern> &) const;
    void check_hessian_sparsity(const sparsity_pattern &) const;
    void check_gradient_vector(const vector_double &) const;
    void check_hessians_vector(const std::vector<vector_double> &) const;

private:
    // Pointer to the inner base problem
    std::unique_ptr<detail::prob_inner_base> m_ptr;
    // Counter for calls to the fitness
    mutable std::atomic<unsigned long long> m_fevals;
    // Counter for calls to the gradient
    mutable std::atomic<unsigned long long> m_gevals;
    // Counter for calls to the hessians
    mutable std::atomic<unsigned long long> m_hevals;
    // Various problem properties determined at construction time
    // from the concrete problem. These will be constant for the lifetime
    // of problem, but we cannot mark them as such because we want to be
    // able to assign and deserialise problems.
    vector_double m_lb;
    vector_double m_ub;
    vector_double::size_type m_nobj;
    vector_double::size_type m_nec;
    vector_double::size_type m_nic;
    vector_double::size_type m_nix;
    vector_double m_c_tol;
    bool m_has_batch_fitness;
    bool m_has_gradient;
    bool m_has_gradient_sparsity;
    bool m_has_hessians;
    bool m_has_hessians_sparsity;
    bool m_has_set_seed;
    std::string m_name;
    // These are the dimensions of the sparsity objects, cached
    // here upon construction in order to provide fast checking
    // on the returned gradient and hessians.
    vector_double::size_type m_gs_dim;
    std::vector<vector_double::size_type> m_hs_dim;
    // Thread safety.
    thread_safety m_thread_safety;
};

} // namespace pagmo

// Disable tracking for the serialisation of problem.
BOOST_CLASS_TRACKING(pagmo::problem, boost::serialization::track_never)

#endif
