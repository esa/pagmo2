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

#ifndef PAGMO_PROBLEM_HPP
#define PAGMO_PROBLEM_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "detail/custom_comparisons.hpp"
#include "exceptions.hpp"
#include "io.hpp"
#include "serialization.hpp"
#include "type_traits.hpp"
#include "types.hpp"
#include "utils/constrained.hpp"

/// Macro for the registration of the serialization functionality for problems.
/**
 * This macro should always be invoked after the declaration of a user-defined problem: it will register
 * the problem with PaGMO's serialization machinery. The macro should be called in the root namespace
 * and using the fully qualified name of the problem to be registered. For example:
 * @code{.unparsed}
 * namespace my_namespace
 * {
 *
 * class my_problem
 * {
 *    // ...
 * };
 *
 * }
 *
 * PAGMO_REGISTER_PROBLEM(my_namespace::my_problem)
 * @endcode
 */
#define PAGMO_REGISTER_PROBLEM(prob) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::prob_inner<prob>, #prob)

namespace pagmo
{

/// Null problem
/**
 * This problem is used to implement the default constructors of meta-problems.
 */
struct null_problem {
    /// Fitness.
    /**
     * @return the vector <tt>[0.]</tt>.
     */
    vector_double fitness(const vector_double &) const
    {
        return {0.};
    }
    /// Problem bounds.
    /**
     * @return the pair <tt>([0.],[1.])</tt>.
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};
}

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

namespace detail
{

// Helper to check that the problem bounds are valid. This will throw if the bounds
// are invalid because of:
// - the bounds size is zero,
// - inconsistent lengths of the vectors,
// - nans in the bounds,
// - lower bounds greater than upper bounds.
inline void check_problem_bounds(const std::pair<vector_double, vector_double> &bounds)
{
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    // 0 - Check that the size is at least 1.
    if (lb.size() == 0u) {
        pagmo_throw(std::invalid_argument, "The bounds dimension cannot be zero");
    }
    // 1 - check bounds have equal length
    if (lb.size() != ub.size()) {
        pagmo_throw(std::invalid_argument, "The length of the lower bounds vector is " + std::to_string(lb.size())
                                               + ", the length of the upper bounds vector is "
                                               + std::to_string(ub.size()));
    }
    // 2 - checks lower < upper for all values in lb, ub, and check for nans.
    for (decltype(lb.size()) i = 0u; i < lb.size(); ++i) {
        if (std::isnan(lb[i]) || std::isnan(ub[i])) {
            pagmo_throw(std::invalid_argument,
                        "A NaN value was encountered in the problem bounds, index: " + std::to_string(i));
        }
        if (lb[i] > ub[i]) {
            pagmo_throw(std::invalid_argument,
                        "The lower bound at position " + std::to_string(i) + " is " + std::to_string(lb[i])
                            + " while the upper bound has the smaller value " + std::to_string(ub[i]));
        }
    }
}

// Two helper functions to compute sparsity patterns in the dense case.
inline std::vector<sparsity_pattern> dense_hessians(vector_double::size_type f_dim, vector_double::size_type dim)
{
    std::vector<sparsity_pattern> retval(f_dim);
    for (auto &Hs : retval) {
        for (decltype(dim) j = 0u; j < dim; ++j) {
            for (decltype(dim) i = 0u; i <= j; ++i) {
                Hs.emplace_back(j, i);
            }
        }
    }
    return retval;
}

inline sparsity_pattern dense_gradient(vector_double::size_type f_dim, vector_double::size_type dim)
{
    sparsity_pattern retval;
    for (decltype(f_dim) j = 0u; j < f_dim; ++j) {
        for (decltype(dim) i = 0u; i < dim; ++i) {
            retval.emplace_back(j, i);
        }
    }
    return retval;
}

struct prob_inner_base {
    virtual ~prob_inner_base()
    {
    }
    virtual prob_inner_base *clone() const = 0;
    virtual vector_double fitness(const vector_double &) const = 0;
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
    virtual void set_seed(unsigned int) = 0;
    virtual bool has_set_seed() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

template <typename T>
struct prob_inner final : prob_inner_base {
    // Static checks.
    static_assert(std::is_default_constructible<T>::value && std::is_copy_constructible<T>::value
                      && std::is_move_constructible<T>::value && std::is_destructible<T>::value,
                  "A problem must be default-constructible, copy-constructible, move-constructible and destructible.");
    static_assert(has_fitness<T>::value, "A problem must provide a fitness function: the fitness function was either "
                                         "not provided or not implemented correctly.");
    static_assert(has_bounds<T>::value, "A problem must provide a getter for its box bounds: the getter was either not "
                                        "provided or not implemented correctly.");
    // We just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit prob_inner(const T &x) : m_value(x)
    {
    }
    explicit prob_inner(T &&x) : m_value(std::move(x))
    {
    }
    // The clone method, used in the copy constructor of problem.
    virtual prob_inner_base *clone() const override final
    {
        return ::new prob_inner(m_value);
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
    virtual void set_seed(unsigned int seed) override final
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
    // Implementation of the optional methods.
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
    static vector_double gradient_impl(const U &, const vector_double &)
    {
        pagmo_throw(not_implemented_error, "The gradient has been requested but it is not implemented in the UDP");
    }
    template <typename U, enable_if_t<pagmo::has_gradient<U>::value && pagmo::override_has_gradient<U>::value, int> = 0>
    static bool has_gradient_impl(const U &p)
    {
        return p.has_gradient();
    }
    template <typename U,
              enable_if_t<pagmo::has_gradient<U>::value && !pagmo::override_has_gradient<U>::value, int> = 0>
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
    [[noreturn]] static sparsity_pattern gradient_sparsity_impl(const U &)
    {
        // NOTE: we should never end up here. gradient_sparsity() is called only if m_has_gradient_sparsity
        // in the problem is set to true, and m_has_gradient_sparsity is unconditionally false if the UDP
        // does not implement gradient_sparsity() (see implementation of the three overloads below).
        assert(false);
        throw;
    }
    template <
        typename U,
        enable_if_t<pagmo::has_gradient_sparsity<U>::value && pagmo::override_has_gradient_sparsity<U>::value, int> = 0>
    static bool has_gradient_sparsity_impl(const U &p)
    {
        return p.has_gradient_sparsity();
    }
    template <typename U,
              enable_if_t<pagmo::has_gradient_sparsity<U>::value && !pagmo::override_has_gradient_sparsity<U>::value,
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
    static std::vector<vector_double> hessians_impl(const U &, const vector_double &)
    {
        pagmo_throw(not_implemented_error, "The hessians have been requested but they are not implemented in the UDP");
    }
    template <typename U, enable_if_t<pagmo::has_hessians<U>::value && pagmo::override_has_hessians<U>::value, int> = 0>
    static bool has_hessians_impl(const U &p)
    {
        return p.has_hessians();
    }
    template <typename U,
              enable_if_t<pagmo::has_hessians<U>::value && !pagmo::override_has_hessians<U>::value, int> = 0>
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
    [[noreturn]] static std::vector<sparsity_pattern> hessians_sparsity_impl(const U &)
    {
        // NOTE: we should never end up here. hessians_sparsity() is called only if m_has_hessians_sparsity
        // in the problem is set to true, and m_has_hessians_sparsity is unconditionally false if the UDP
        // does not implement hessians_sparsity() (see implementation of the three overloads below).
        assert(false);
        throw;
    }
    template <
        typename U,
        enable_if_t<pagmo::has_hessians_sparsity<U>::value && pagmo::override_has_hessians_sparsity<U>::value, int> = 0>
    static bool has_hessians_sparsity_impl(const U &p)
    {
        return p.has_hessians_sparsity();
    }
    template <typename U,
              enable_if_t<pagmo::has_hessians_sparsity<U>::value && !pagmo::override_has_hessians_sparsity<U>::value,
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
    template <typename U, typename std::enable_if<pagmo::has_set_seed<U>::value, int>::type = 0>
    static void set_seed_impl(U &value, unsigned int seed)
    {
        value.set_seed(seed);
    }
    template <typename U, enable_if_t<!pagmo::has_set_seed<U>::value, int> = 0>
    static void set_seed_impl(U &, unsigned int)
    {
        pagmo_throw(not_implemented_error,
                    "The set_seed() method has been invoked but it is not implemented in the UDP");
    }
    template <typename U, enable_if_t<pagmo::has_set_seed<U>::value && override_has_set_seed<U>::value, int> = 0>
    static bool has_set_seed_impl(const U &p)
    {
        return p.has_set_seed();
    }
    template <typename U, enable_if_t<pagmo::has_set_seed<U>::value && !override_has_set_seed<U>::value, int> = 0>
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

    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<prob_inner_base>(this), m_value);
    }
    T m_value;
};

} // namespace detail

/// Problem class.
/**
 * \image html problem.png
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
 * where \f$\mathbf x \in \mathbb R^{n_x}\f$ is called *decision vector* or
 * *chromosome*, \f$\mathbf{lb}, \mathbf{ub} \in \mathbb R^{n_x}\f$ are the *box-bounds*,
 * \f$ \mathbf f: \mathbb R^{n_x} \rightarrow \mathbb R^{n_{obj}}\f$ define the *objectives*,
 * \f$ \mathbf c_e:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ec}}\f$ are non linear *equality constraints*,
 * and \f$ \mathbf c_i:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ic}}\f$ are non linear *inequality constraints*.
 * Note that the objectives and constraints may also depend from an added value \f$s\f$ seeding the
 * values of any number of stochastic variables. This allows also for stochastic programming
 * tasks to be represented by this class. A tolerance is considered for the verification of the constraints and is set
 * by default to zero, but it can be modified via the problem::set_c_tol() method.
 *
 * In order to define an optimizaztion problem in PaGMO, the user must first define a class
 * (or a struct) whose methods describe the properties of the problem and allow to compute
 * the objective function, the gradient, the constraints, etc. In PaGMO, we refer to such
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
 * The <tt>%fitness()</tt> method is expected to return the fitness of the input decision vector, while
 * <tt>%get_bounds()</tt> is expected to return the box bounds of the problem,
 * \f$(\mathbf{lb}, \mathbf{ub})\f$, which also implicitly define the dimension of the problem.
 * The <tt>%fitness()</tt> and <tt>%get_bounds()</tt> methods of the UDP are accessible from the corresponding
 * problem::fitness()
 * and problem::get_bounds() methods (see their documentation for details).
 *
 * The two mandatory methods above allow to define a single objective, deterministic, derivative-free, unconstrained
 * optimization problem. In order to consider more complex cases, the UDP may implement one or more of the following
 * methods:
 * @code{.unparsed}
 * vector_double::size_type get_nobj() const;
 * vector_double::size_type get_nec() const;
 * vector_double::size_type get_nic() const;
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
 * @endcode
 *
 * See the documentation of the corresponding methods in this class for details on how the optional
 * methods in the UDP are used by pagmo::problem.
 *
 * **NOTE** the only allowed operations on an object belonging to this class, after it has been moved, are assignment
 * and destruction.
 */
class problem
{
    // Enable the generic ctor only if T is not a problem (after removing
    // const/reference qualifiers).
    template <typename T>
    using generic_ctor_enabler = enable_if_t<!std::is_same<problem, uncvref_t<T>>::value, int>;

public:
    /// Constructor from a user-defined problem of type \p T
    /**
     * **NOTE** this constructor is not enabled if, after the removal of cv and reference qualifiers,
     * \p T is of type pagmo::problem (that is, this constructor does not compete with the copy/move
     * constructors of pagmo::problem).
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
     *   different from the dimensionality of the upper bounds, etc. - note that infinite bounds are allowed),
     * - the <tt>%gradient_sparsity()</tt> and <tt>%hessians_sparsity()</tt> methods of the UDP fail basic sanity checks
     *   (e.g., they return vectors with repeated indices, they contain indices exceeding the problem's dimensions,
     *   etc.).
     * @throws unspecified any exception thrown by methods of the UDP invoked during construction or by memory errors
     * in strings and standard containers.
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit problem(T &&x)
        : m_ptr(::new detail::prob_inner<uncvref_t<T>>(std::forward<T>(x))), m_fevals(0u), m_gevals(0u), m_hevals(0u)
    {
        // 1 - Bounds.
        auto bounds = ptr()->get_bounds();
        detail::check_problem_bounds(bounds);
        m_lb = std::move(bounds.first);
        m_ub = std::move(bounds.second);
        // 2 - Number of objectives.
        m_nobj = ptr()->get_nobj();
        if (!m_nobj) {
            pagmo_throw(std::invalid_argument, "The number of objectives cannot be zero");
        }
        // NOTE: here we check that we can always compute nobj + nec + nic safely.
        if (m_nobj > std::numeric_limits<decltype(m_nobj)>::max() / 3u) {
            pagmo_throw(std::invalid_argument, "The number of objectives is too large");
        }
        // 3 - Constraints.
        m_nec = ptr()->get_nec();
        if (m_nec > std::numeric_limits<decltype(m_nec)>::max() / 3u) {
            pagmo_throw(std::invalid_argument, "The number of equality constraints is too large");
        }
        m_nic = ptr()->get_nic();
        if (m_nic > std::numeric_limits<decltype(m_nic)>::max() / 3u) {
            pagmo_throw(std::invalid_argument, "The number of inequality constraints is too large");
        }
        // 4 - Presence of gradient and its sparsity.
        // NOTE: all these m_has_* attributes refer to the presence of the features in the UDP.
        m_has_gradient = ptr()->has_gradient();
        m_has_gradient_sparsity = ptr()->has_gradient_sparsity();
        // 5 - Presence of Hessians and their sparsity.
        m_has_hessians = ptr()->has_hessians();
        m_has_hessians_sparsity = ptr()->has_hessians_sparsity();
        // 5bis - Is this a stochastic problem?
        m_has_set_seed = ptr()->has_set_seed();
        // 6 - Name.
        m_name = ptr()->get_name();
        // 7 - Check the sparsities, and cache their sizes.
        if (m_has_gradient_sparsity) {
            // If the problem provides gradient sparsity, get it, check it
            // and store its size.
            const auto gs = ptr()->gradient_sparsity();
            check_gradient_sparsity(gs);
            m_gs_dim = gs.size();
        } else {
            // If the problem does not provide gradient sparsity, we assume dense
            // sparsity. We can compute easily the expected size of the sparsity
            // in this case.
            const auto nx = get_nx();
            const auto nf = get_nf();
            if (nx > std::numeric_limits<vector_double::size_type>::max() / nf) {
                pagmo_throw(std::invalid_argument, "The size of the (dense) gradient sparsity is too large");
            }
            m_gs_dim = nx * nf;
        }
        // Same as above for the hessians.
        if (m_has_hessians_sparsity) {
            const auto hs = ptr()->hessians_sparsity();
            check_hessians_sparsity(hs);
            for (const auto &one_hs : hs) {
                m_hs_dim.push_back(one_hs.size());
            }
        } else {
            const auto nx = get_nx();
            const auto nf = get_nf();
            if (nx == std::numeric_limits<vector_double::size_type>::max()
                || nx / 2u > std::numeric_limits<vector_double::size_type>::max() / (nx + 1u)) {
                pagmo_throw(std::invalid_argument, "The size of the (dense) hessians "
                                                   "sparsity is too large");
            }
            // We resize rather than push back here, so that an std::length_error is called quickly rather
            // than an std::bad_alloc after waiting the growth
            m_hs_dim.resize(nf);
            for (vector_double::size_type i = 0u; i < nf; ++i) {
                m_hs_dim[i] = (nx * (nx - 1u) / 2u + nx); // lower triangular
            }
        }
        // 8 - Constraint tolerance
        m_c_tol.resize(m_nec + m_nic);
    }

    /// Copy constructor.
    /**
     * The copy constructor will deep copy the input problem \p other.
     *
     * @param other the problem to be copied.
     *
     * @throws unspecified any exception thrown by:
     * - memory allocation errors in standard containers,
     * - the copying of the internal UDP.
     */
    problem(const problem &other)
        : m_ptr(other.ptr()->clone()), m_fevals(other.m_fevals.load()), m_gevals(other.m_gevals.load()),
          m_hevals(other.m_hevals.load()), m_lb(other.m_lb), m_ub(other.m_ub), m_nobj(other.m_nobj), m_nec(other.m_nec),
          m_nic(other.m_nic), m_c_tol(other.m_c_tol), m_has_gradient(other.m_has_gradient),
          m_has_gradient_sparsity(other.m_has_gradient_sparsity), m_has_hessians(other.m_has_hessians),
          m_has_hessians_sparsity(other.m_has_hessians_sparsity), m_has_set_seed(other.m_has_set_seed),
          m_name(other.m_name), m_gs_dim(other.m_gs_dim), m_hs_dim(other.m_hs_dim)
    {
    }

    /// Move constructor.
    /**
     * @param other the problem from which \p this will be move-constructed.
     */
    problem(problem &&other) noexcept
        : m_ptr(std::move(other.m_ptr)), m_fevals(other.m_fevals.load()), m_gevals(other.m_gevals.load()),
          m_hevals(other.m_hevals.load()), m_lb(std::move(other.m_lb)), m_ub(std::move(other.m_ub)),
          m_nobj(other.m_nobj), m_nec(other.m_nec), m_nic(other.m_nic), m_c_tol(std::move(other.m_c_tol)),
          m_has_gradient(other.m_has_gradient), m_has_gradient_sparsity(other.m_has_gradient_sparsity),
          m_has_hessians(other.m_has_hessians), m_has_hessians_sparsity(other.m_has_hessians_sparsity),
          m_has_set_seed(other.m_has_set_seed), m_name(std::move(other.m_name)), m_gs_dim(other.m_gs_dim),
          m_hs_dim(std::move(other.m_hs_dim))
    {
    }

    /// Move assignment operator
    /**
     * @param other the assignment target.
     *
     * @return a reference to \p this.
     */
    problem &operator=(problem &&other) noexcept
    {
        if (this != &other) {
            m_ptr = std::move(other.m_ptr);
            m_fevals.store(other.m_fevals.load());
            m_gevals.store(other.m_gevals.load());
            m_hevals.store(other.m_hevals.load());
            m_lb = std::move(other.m_lb);
            m_ub = std::move(other.m_ub);
            m_nobj = other.m_nobj;
            m_nec = other.m_nec;
            m_nic = other.m_nic;
            m_c_tol = std::move(other.m_c_tol);
            m_has_gradient = other.m_has_gradient;
            m_has_gradient_sparsity = other.m_has_gradient_sparsity;
            m_has_hessians = other.m_has_hessians;
            m_has_hessians_sparsity = other.m_has_hessians_sparsity;
            m_has_set_seed = other.m_has_set_seed;
            m_name = std::move(other.m_name);
            m_gs_dim = other.m_gs_dim;
            m_hs_dim = std::move(other.m_hs_dim);
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
    problem &operator=(const problem &other)
    {
        // Copy ctor + move assignment.
        return *this = problem(other);
    }

    /// Extract a pointer to the UDP.
    /**
     * This method will extract a pointer to the internal instance of the UDP. If \p T is not the same type
     * as the UDP used during construction (after removal of cv and reference qualifiers), this method will
     * return \p nullptr.
     *
     * @return a const pointer to the internal UDP, or \p nullptr
     * if \p T does not correspond exactly to the original problem type used
     * in the constructor.
     */
    template <typename T>
    const T *extract() const
    {
        auto p = dynamic_cast<const detail::prob_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    /// Check if the UDP is of type \p T.
    /**
     * @return \p true if the UDP is of type \p T, \p false otherwise.
     */
    template <typename T>
    bool is() const
    {
        return extract<T>() != nullptr;
    }

    /// Fitness.
    /**
     * This method will invoke the <tt>%fitness()</tt> method of the UDP to compute the fitness of the
     * input decision vector \p dv. The return value of the <tt>%fitness()</tt> method of the UDP is expected to have a
     * dimension of \f$n_{f} = n_{obj} + n_{ec} + n_{ic}\f$
     * and to contain the concatenated values of \f$\mathbf f, \mathbf c_e\f$ and \f$\mathbf c_i\f$ (in this order).
     *
     * In addition to invoking the <tt>%fitness()</tt> method of the UDP, this method will perform sanity checks on
     * \p dv and on the returned fitness vector. A successful call of this method will increase the internal fitness
     * evaluation counter (see problem::get_fevals()).
     *
     * @param dv the decision vector.
     *
     * @return the fitness of \p dv.
     *
     * @throws std::invalid_argument if either
     * - the length of \p dv differs from the value returned by get_nx(), or
     * - the length of the returned fitness vector differs from the the value returned by get_nf().
     * @throws unspecified any exception thrown by the <tt>%fitness()</tt> method of the UDP.
     */
    vector_double fitness(const vector_double &dv) const
    {
        // 1 - checks the decision vector
        check_decision_vector(dv);
        // 2 - computes the fitness
        vector_double retval(ptr()->fitness(dv));
        // 3 - checks the fitness vector
        check_fitness_vector(retval);
        // 4 - increments fitness evaluation counter
        ++m_fevals;
        return retval;
    }

    /// Gradient.
    /**
     * This method will compute the gradient of the input decision vector \p dv by invoking
     * the <tt>%gradient()</tt> method of the UDP. The <tt>%gradient()</tt> method of the UDP must return
     * a sparse representation of the gradient: the \f$ k\f$-th term of the gradient vector
     * is expected to contain \f$ \frac{\partial f_i}{\partial x_j}\f$, where the pair \f$(i,j)\f$
     * is the \f$k\f$-th element of the sparsity pattern (collection of index pairs), as returned by
     * problem::gradient_sparsity().
     *
     * If the UDP satisfies pagmo::has_gradient, this method will forward \p dv to the <tt>%gradient()</tt>
     * method of the UDP after sanity checks. The output of the <tt>%gradient()</tt>
     * method of the UDP will also be checked before being returned. If the UDP does not satisfy
     * pagmo::has_gradient, an error will be raised.
     *
     * A successful call of this method will increase the internal gradient evaluation counter (see
     * problem::get_gevals()).
     *
     * @param dv the decision vector whose gradient will be computed.
     *
     * @return the gradient of \p dv.
     *
     * @throws std::invalid_argument if either:
     * - the length of \p dv differs from the output of get_nx(), or
     * - the returned gradient vector does not have the same size as the output of pagmo::sparsity_pattern().
     * @throws not_implemented_error if the UDP does not satisfy pagmo::has_gradient.
     * @throws unspecified any exception thrown by the <tt>%gradient()</tt> method of the UDP.
     */
    vector_double gradient(const vector_double &dv) const
    {
        // 1 - checks the decision vector
        check_decision_vector(dv);
        // 2 - compute the gradients
        vector_double retval(ptr()->gradient(dv));
        // 3 - checks the gradient vector
        check_gradient_vector(retval);
        // 4 - increments gradient evaluation counter
        ++m_gevals;
        return retval;
    }

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
     * @return a flag signalling the availability of the gradient.
     */
    bool has_gradient() const
    {
        return m_has_gradient;
    }

    /// Gradient sparsity pattern.
    /**
     * This method will return the gradient sparsity pattern of the problem. The gradient sparsity pattern is a
     * collection of the indexes \f$(i,j)\f$ of the non-zero elements of
     * \f$ g_{ij} = \frac{\partial f_i}{\partial x_j}\f$.
     *
     * If problem::has_gradient_sparsity() returns \p true,
     * then the <tt>%gradient_sparsity()</tt> method of the UDP will be invoked, and its result returned (after sanity
     * checks). Otherwise, a a dense pattern is assumed and the returned vector will be
     * \f$((0,0),(0,1), ... (0,n_x-1), ...(n_f-1,n_x-1))\f$.
     *
     * @return the gradient sparsity pattern.
     *
     * @throws std::invalid_argument if the sparsity pattern returned by the UDP is invalid (specifically, if
     * it contains duplicate pairs of indices or if the indices in the pattern are incompatible with the properties of
     * the problem).
     * @throws not_implemented_error if the <tt>%gradient_sparsity()</tt> method of the UDP is invoked without being
     * available. This indicates in general an inconsistency in the implementation of the UDP.
     * @throws unspecified memory errors in standard containers.
     */
    sparsity_pattern gradient_sparsity() const
    {
        if (has_gradient_sparsity()) {
            auto retval = ptr()->gradient_sparsity();
            check_gradient_sparsity(retval);
            return retval;
        }
        return detail::dense_gradient(get_nf(), get_nx());
    }

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
     * **NOTE** regardless of what this method returns, the problem::gradient_sparsity() method will always return
     * a sparsity pattern: if the UDP does not provide the gradient sparsity, PaGMO will assume that the sparsity
     * pattern of the gradient is dense. See problem::gradient_sparsity() for more details.
     *
     * @return a flag signalling the availability of the gradient sparsity in the UDP.
     */
    bool has_gradient_sparsity() const
    {
        return m_has_gradient_sparsity;
    }

    /// Hessians.
    /**
     * This method will compute the hessians of the input decision vector \p dv by invoking
     * the <tt>%hessians()</tt> method of the UDP. The <tt>%hessians()</tt> method of the UDP must return
     * a sparse representation of the hessians: the element \f$ l\f$ of the returned vector contains
     * \f$ h^l_{ij} = \frac{\partial f^2_l}{\partial x_i\partial x_j}\f$
     * in the order specified by the \f$ l\f$-th element of the
     * hessians sparsity pattern (a vector of index pairs \f$(i,j)\f$)
     * as returned by problem::hessians_sparsity(). Since
     * the hessians are symmetric, their sparse representation contains only lower triangular elements.
     *
     * If the UDP satisfies pagmo::has_hessians, this method will forward \p dv to the <tt>%hessians()</tt>
     * method of the UDP after sanity checks. The output of the <tt>%hessians()</tt>
     * method of the UDP will also be checked before being returned. If the UDP does not satisfy
     * pagmo::has_hessians, an error will be raised.
     *
     * A successful call of this method will increase the internal hessians evaluation counter (see
     * problem::get_hevals()).
     *
     * @param dv the decision vector whose hessians will be computed.
     *
     * @return the hessians of \p dv.
     *
     * @throws std::invalid_argument if either:
     * - the length of \p dv differs from the output of get_nx(), or
     * - the length of each hessian returned does not match the corresponding hessians sparsity pattern dimensions.
     * @throws not_implemented_error if the UDP does not satisfy pagmo::has_hessians.
     * @throws unspecified any exception thrown by the <tt>%hessians()</tt> method of the UDP.
     */
    std::vector<vector_double> hessians(const vector_double &dv) const
    {
        // 1 - checks the decision vector
        check_decision_vector(dv);
        // 2 - computes the hessians
        std::vector<vector_double> retval(ptr()->hessians(dv));
        // 3 - checks the hessians
        check_hessians_vector(retval);
        // 4 - increments hessians evaluation counter
        ++m_hevals;
        return retval;
    }

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
     * @return a flag signalling the availability of the hessians.
     */
    bool has_hessians() const
    {
        return m_has_hessians;
    }

    /// Hessians sparsity pattern.
    /**
     * This method will return the hessians sparsity pattern of the problem. Each component \f$ l\f$ of the hessians
     * sparsity pattern is a collection of the indices \f$(i,j)\f$ of the non-zero elements of
     * \f$h^l_{ij} = \frac{\partial f^l}{\partial x_i\partial x_j}\f$. Since the Hessian matrix
     * is symmetric, only lower triangular elements are allowed.
     *
     * If problem::has_hessians_sparsity() returns \p true,
     * then the <tt>%hessians_sparsity()</tt> method of the UDP will be invoked, and its result returned (after sanity
     * checks). Otherwise, a dense pattern is assumed and \f$n_f\f$ sparsity patterns
     * containing \f$((0,0),(1,0), (1,1), (2,0) ... (n_x-1,n_x-1))\f$ will be returned.
     *
     * @return the hessians sparsity pattern.
     *
     * @throws std::invalid_argument if the sparsity pattern returned by the UDP is invalid (specifically, if
     * its size is invalid, if it contains duplicate pairs of indices or if the returned indices do not correspond
     * to  a lower triangular representation of a symmetric matrix).
     * @throws not_implemented_error if the <tt>%hessians_sparsity()</tt> method of the UDP is invoked without being
     * available. This indicates in general an inconsistency in the implementation of the UDP.
     * @throws unspecified memory errors in standard containers.
     */
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        if (m_has_hessians_sparsity) {
            auto retval = ptr()->hessians_sparsity();
            check_hessians_sparsity(retval);
            return retval;
        }
        return detail::dense_hessians(get_nf(), get_nx());
    }

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
     * **NOTE** regardless of what this method returns, the problem::hessians_sparsity() method will always return
     * a vector of sparsity patterns: if the UDP does not provide the hessians sparsity, PaGMO will assume that the
     * sparsity pattern of the hessians is dense. See problem::hessians_sparsity() for more details.
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

    /// Fitness dimension.
    /**
     * @return \f$ n_{f}\f$, the dimension of the fitness, which is the
     * sum of \f$n_{obj}\f$, \f$n_{ec}\f$ and \f$n_{ic}\f$
     */
    vector_double::size_type get_nf() const
    {
        return m_nobj + m_nic + m_nec;
    }

    /// Box-bounds.
    /**
     * @return \f$ (\mathbf{lb}, \mathbf{ub}) \f$, the box-bounds, as returned by
     * the <tt>%get_bounds()</tt> method of the UDP. Infinities in the bounds are allowed.
     *
     * @throws unspecified any exception thrown by memory errors in standard containers.
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return std::make_pair(m_lb, m_ub);
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
    /// Set the constraint tolerance.
    /**
     * @param c_tol a vector containing the tolerances to use when
     * checking for constraint feasibility.
     *
     * @throws std::invalid_argument if the size of \p c_tol differs from the number of constraints, or if
     * any of its elements is negative or NaN.
     */
    void set_c_tol(const vector_double &c_tol)
    {
        if (c_tol.size() != this->get_nc()) {
            pagmo_throw(std::invalid_argument, "The tolerance vector size should be: " + std::to_string(this->get_nc())
                                                   + ", while a size of: " + std::to_string(c_tol.size())
                                                   + " was detected.");
        }
        for (decltype(c_tol.size()) i = 0; i < c_tol.size(); ++i) {
            if (std::isnan(c_tol[i])) {
                pagmo_throw(std::invalid_argument,
                            "The tolerance vector has a NaN value at the index " + std::to_string(i));
            }
            if (c_tol[i] < 0.) {
                pagmo_throw(std::invalid_argument,
                            "The tolerance vector has a negative value at the index " + std::to_string(i));
            }
        }
        m_c_tol = c_tol;
    }
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
        return m_fevals.load();
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
        return m_gevals.load();
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
        return m_hevals.load();
    }

    /// Set the seed for the stochastic variables.
    /**
     * Sets the seed to be used in the fitness function to instantiate
     * all stochastic variables. If the UDP satisfies pagmo::has_set_seed, then
     * its <tt>%set_seed()</tt> method will be invoked. Otherwise, an error will be raised.
     *
     * @param seed seed.
     *
     * @throws not_implemented_error if the UDP does not satisfy pagmo::has_set_seed.
     * @throws unspecified any exception thrown by the <tt>%set_seed()</tt> method of the UDP.
     */
    void set_seed(unsigned seed)
    {
        ptr()->set_seed(seed);
    }

    /// Feasibility of a decision vector.
    /**
     * This method will checks the feasibility of the fitness corresponding to
     * a decision vector \p x against
     * the tolerances returned by problem::get_c_tol().
     *
     * **NOTE** This will cause one fitness evaluation.
     *
     * @param x a decision vector.
     *
     * @return \p true if the decision vector results in a feasible fitness, \p false otherwise.
     *
     * @throws unspecified any exception thrown by problem::feasibility_f() or problem::fitness().
     */
    bool feasibility_x(const vector_double &x) const
    {
        // Wrong dimensions of x will trigger exceptions in the called functions
        return feasibility_f(fitness(x));
    }

    /// Feasibility of a fitness vector.
    /**
     * This method will check the feasibility of a fitness vector \p f against
     * the tolerances returned by problem::get_c_tol().
     *
     * @param f a fitness vector.
     *
     * @return \p true if the fitness vector is feasible, \p false otherwise.
     *
     * @throws std::invalid_argument if the size of \p f is not the same as the output of problem::get_nf().
     */
    bool feasibility_f(const vector_double &f) const
    {
        if (f.size() != get_nf()) {
            pagmo_throw(std::invalid_argument,
                        "The fitness passed as argument has dimension of: " + std::to_string(f.size())
                            + ", while the problem defines a fitness size of: " + std::to_string(get_nf()));
        }
        auto feas_eq
            = detail::test_eq_constraints(f.data() + get_nobj(), f.data() + get_nobj() + get_nec(), get_c_tol().data());
        auto feas_ineq = detail::test_ineq_constraints(f.data() + get_nobj() + get_nec(), f.data() + f.size(),
                                                       get_c_tol().data() + get_nec());
        return feas_eq.first + feas_ineq.first == get_nc();
    }

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
     * @return a flag signalling the availability of the <tt>%set_seed()</tt> method.
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

    /// Problem's extra info.
    /**
     * If the UDP satisfies pagmo::has_extra_info, then this method will return the output of its
     * <tt>%get_extra_info()</tt>
     * method. Otherwise, an empty string will be returned.
     *
     * @return extra info about the UDP.
     *
     * @throws unspecified any exception thrown by the <tt>%get_extra_info()</tt> method of the UDP.
     */
    std::string get_extra_info() const
    {
        return ptr()->get_extra_info();
    }

    /// Streaming operator
    /**
     * This function will stream to \p os a human-readable representation of the input
     * problem \p p.
     *
     * @param os input <tt>std::ostream</tt>.
     * @param p pagmo::problem object to be streamed.
     *
     * @return a reference to \p os.
     *
     * @throws unspecified any exception thrown by querying various problem properties and streaming them into \p os.
     */
    friend std::ostream &operator<<(std::ostream &os, const problem &p)
    {
        os << "Problem name: " << p.get_name();
        if (p.is_stochastic()) {
            stream(os, " [stochastic]");
        }
        os << "\n\tGlobal dimension:\t\t\t" << p.get_nx() << '\n';
        os << "\tFitness dimension:\t\t\t" << p.get_nf() << '\n';
        os << "\tNumber of objectives:\t\t\t" << p.get_nobj() << '\n';
        os << "\tEquality constraints dimension:\t\t" << p.get_nec() << '\n';
        os << "\tInequality constraints dimension:\t" << p.get_nic() << '\n';
        if (p.get_nec() + p.get_nic() > 0u) {
            stream(os, "\tTolerances on constraints:\t", p.get_c_tol(), '\n');
        }
        os << "\tLower bounds: ";
        stream(os, p.get_bounds().first, '\n');
        os << "\tUpper bounds: ";
        stream(os, p.get_bounds().second, '\n');
        stream(os, "\n\tHas gradient: ", p.has_gradient(), '\n');
        stream(os, "\tUser implemented gradient sparsity: ", p.m_has_gradient_sparsity, '\n');
        if (p.has_gradient()) {
            stream(os, "\tExpected gradients: ", p.m_gs_dim, '\n');
        }
        stream(os, "\tHas hessians: ", p.has_hessians(), '\n');
        stream(os, "\tUser implemented hessians sparsity: ", p.m_has_hessians_sparsity, '\n');
        if (p.has_hessians()) {
            stream(os, "\tExpected hessian components: ", p.m_hs_dim, '\n');
        }
        stream(os, "\n\tFunction evaluations: ", p.get_fevals(), '\n');
        if (p.has_gradient()) {
            stream(os, "\tGradient evaluations: ", p.get_gevals(), '\n');
        }
        if (p.has_hessians()) {
            stream(os, "\tHessians evaluations: ", p.get_hevals(), '\n');
        }

        const auto extra_str = p.get_extra_info();
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
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(m_ptr, m_fevals.load(), m_gevals.load(), m_hevals.load(), m_lb, m_ub, m_nobj, m_nec, m_nic, m_c_tol,
           m_has_gradient, m_has_gradient_sparsity, m_has_hessians, m_has_hessians_sparsity, m_has_set_seed, m_name,
           m_gs_dim, m_hs_dim);
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
    void load(Archive &ar)
    {
        ar(m_ptr);
        unsigned long long tmp;
        ar(tmp);
        m_fevals.store(tmp);
        ar(tmp);
        m_gevals.store(tmp);
        ar(tmp);
        m_hevals.store(tmp);
        ar(m_lb, m_ub, m_nobj, m_nec, m_nic, m_c_tol, m_has_gradient, m_has_gradient_sparsity, m_has_hessians,
           m_has_hessians_sparsity, m_has_set_seed, m_name, m_gs_dim, m_hs_dim);
    }

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

    // A small helper to check if a vector containes unique elements.
    // This version for floating point types is also protected against possible nans
    template <typename U, detail::enable_if_is_floating_point<U> = 0>
    static bool all_unique(std::vector<U> x)
    {
        std::sort(x.begin(), x.end(), detail::less_than_f<U>);
        auto it = std::unique(x.begin(), x.end(), detail::equal_to_f<U>);
        return it == x.end();
    }
    // The version for non floating point types is not protected vs possible nans
    // (e.g if used with std::pair it could be troublesome)
    template <typename U, detail::enable_if_is_not_floating_point<U> = 0>
    static bool all_unique(std::vector<U> x)
    {
        std::sort(x.begin(), x.end());
        auto it = std::unique(x.begin(), x.end());
        return it == x.end();
    }

    void check_gradient_sparsity(const sparsity_pattern &gs) const
    {
        const auto nx = get_nx();
        const auto nf = get_nf();
        // 1 - We check that the gradient sparsity pattern has
        // valid indices.
        for (const auto &pair : gs) {
            if ((pair.first >= nf) || (pair.second >= nx)) {
                pagmo_throw(std::invalid_argument, "Invalid pair detected in the gradient sparsity pattern: ("
                                                       + std::to_string(pair.first) + ", " + std::to_string(pair.second)
                                                       + ")\nFitness dimension is: " + std::to_string(nf)
                                                       + "\nDecision vector dimension is: " + std::to_string(nx));
            }
        }
        // 2 - We check all pairs are unique
        if (!all_unique(gs)) {
            pagmo_throw(std::invalid_argument,
                        "Multiple entries of the same index pair was detected in the gradient sparsity pattern");
        }
    }
    void check_hessians_sparsity(const std::vector<sparsity_pattern> &hs) const
    {
        // 1 - We check that a hessian sparsity is provided for each component
        // of the fitness
        const auto nf = get_nf();
        if (hs.size() != nf) {
            pagmo_throw(std::invalid_argument, "Invalid dimension of the hessians_sparsity: "
                                                   + std::to_string(hs.size()) + ", expected: " + std::to_string(nf));
        }
        // 2 - We check that all hessian sparsity patterns have
        // valid indexes.
        for (const auto &one_hs : hs) {
            check_hessian_sparsity(one_hs);
        }
    }
    void check_hessian_sparsity(const sparsity_pattern &hs) const
    {
        const auto nx = get_nx();
        // 1 - We check that the hessian sparsity pattern has
        // valid indexes. Assuming a lower triangular representation of
        // a symmetric matrix. Example, for a 4x4 dense symmetric
        // [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)]
        for (const auto &pair : hs) {
            if ((pair.first >= nx) || (pair.second > pair.first)) {
                pagmo_throw(std::invalid_argument, "Invalid pair detected in the hessians sparsity pattern: ("
                                                       + std::to_string(pair.first) + ", " + std::to_string(pair.second)
                                                       + ")\nDecision vector dimension is: " + std::to_string(nx)
                                                       + "\nNOTE: hessian is a symmetric matrix and PaGMO represents "
                                                         "it as lower triangular: i.e (i,j) is not valid if j>i");
            }
        }
        // 2 -  We check all pairs are unique
        if (!all_unique(hs)) {
            pagmo_throw(std::invalid_argument,
                        "Multiple entries of the same index pair were detected in the hessian sparsity pattern");
        }
    }
    void check_decision_vector(const vector_double &dv) const
    {
        // 1 - check decision vector for length consistency
        if (dv.size() != get_nx()) {
            pagmo_throw(std::invalid_argument, "Length of decision vector is " + std::to_string(dv.size())
                                                   + ", should be " + std::to_string(get_nx()));
        }
        // 2 - Here is where one could check if the decision vector
        // is in the bounds. At the moment not implemented
    }

    void check_fitness_vector(const vector_double &f) const
    {
        auto nf = get_nf();
        // Checks dimension of returned fitness
        if (f.size() != nf) {
            pagmo_throw(std::invalid_argument,
                        "Fitness length is: " + std::to_string(f.size()) + ", should be " + std::to_string(nf));
        }
    }

    void check_gradient_vector(const vector_double &gr) const
    {
        // Checks that the gradient vector returned has the same dimensions of the sparsity_pattern
        if (gr.size() != m_gs_dim) {
            pagmo_throw(std::invalid_argument,
                        "Gradients returned: " + std::to_string(gr.size()) + ", should be " + std::to_string(m_gs_dim));
        }
    }

    void check_hessians_vector(const std::vector<vector_double> &hs) const
    {
        // Checks that the hessians returned have the same dimensions of the
        // corresponding sparsity patterns
        for (decltype(hs.size()) i = 0u; i < hs.size(); ++i) {
            if (hs[i].size() != m_hs_dim[i]) {
                pagmo_throw(std::invalid_argument, "On the hessian no. " + std::to_string(i) + ": Components returned: "
                                                       + std::to_string(hs[i].size()) + ", should be "
                                                       + std::to_string(m_hs_dim[i]));
            }
        }
    }

private:
    // Pointer to the inner base problem
    std::unique_ptr<detail::prob_inner_base> m_ptr;
    // Atomic counter for calls to the fitness
    mutable std::atomic<unsigned long long> m_fevals;
    // Atomic counter for calls to the gradient
    mutable std::atomic<unsigned long long> m_gevals;
    // Atomic counter for calls to the hessians
    mutable std::atomic<unsigned long long> m_hevals;
    // Various problem properties determined at construction time
    // from the concrete problem. These will be constant for the lifetime
    // of problem, but we cannot mark them as such because of serialization.
    vector_double m_lb;
    vector_double m_ub;
    vector_double::size_type m_nobj;
    vector_double::size_type m_nec;
    vector_double::size_type m_nic;
    vector_double m_c_tol;
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
};

} // namespaces

PAGMO_REGISTER_PROBLEM(pagmo::null_problem)

#endif
