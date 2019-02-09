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

#ifndef PAGMO_BATCH_FITNESS_EVALUATOR_HPP
#define PAGMO_BATCH_FITNESS_EVALUATOR_HPP

#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <pagmo/detail/bfe_impl.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#define PAGMO_REGISTER_BATCH_FITNESS_EVALUATOR(bfe)                                                                    \
    CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::batch_fitness_evaluator_inner<bfe>, "udbfe " #bfe)

namespace pagmo
{

template <typename T>
class has_bfe_call_operator
{
    template <typename U>
    using call_t
        = decltype(std::declval<const U &>()(std::declval<const problem &>(), std::declval<const vector_double &>()));
    static const bool implementation_defined = std::is_same<detected_t<call_t, T>, vector_double>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_bfe_call_operator<T>::value;

template <typename T>
class is_udbfe
{
    static const bool implementation_defined
        = std::is_same<T, uncvref_t<T>>::value && std::is_default_constructible<T>::value
          && std::is_copy_constructible<T>::value && std::is_move_constructible<T>::value
          && std::is_destructible<T>::value && has_bfe_call_operator<T>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udbfe<T>::value;

namespace detail
{

struct batch_fitness_evaluator_inner_base {
    virtual ~batch_fitness_evaluator_inner_base() {}
    virtual std::unique_ptr<batch_fitness_evaluator_inner_base> clone() const = 0;
    virtual vector_double operator()(const problem &, const vector_double &) const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual thread_safety get_thread_safety() const = 0;
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

template <typename T>
struct batch_fitness_evaluator_inner final : batch_fitness_evaluator_inner_base {
    // We just need the def ctor, delete everything else.
    batch_fitness_evaluator_inner() = default;
    batch_fitness_evaluator_inner(const batch_fitness_evaluator_inner &) = delete;
    batch_fitness_evaluator_inner(batch_fitness_evaluator_inner &&) = delete;
    batch_fitness_evaluator_inner &operator=(const batch_fitness_evaluator_inner &) = delete;
    batch_fitness_evaluator_inner &operator=(batch_fitness_evaluator_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit batch_fitness_evaluator_inner(const T &x) : m_value(x) {}
    explicit batch_fitness_evaluator_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of batch_fitness_evaluator.
    virtual std::unique_ptr<batch_fitness_evaluator_inner_base> clone() const override final
    {
        return make_unique<batch_fitness_evaluator_inner>(m_value);
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
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<batch_fitness_evaluator_inner_base>(this), m_value);
    }
    T m_value;
};

} // namespace detail

class default_bfe
{
public:
    vector_double operator()(const problem &, const vector_double &) const
    {
        return vector_double{};
    }
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

class batch_fitness_evaluator
{
    // Enable the generic ctor only if T is not a bfe (after removing
    // const/reference qualifiers), and if T is a udbfe.
    template <typename T>
    using generic_ctor_enabler
        = enable_if_t<!std::is_same<batch_fitness_evaluator, uncvref_t<T>>::value && is_udbfe<uncvref_t<T>>::value,
                      int>;

public:
    // Default ctor.
    batch_fitness_evaluator() : batch_fitness_evaluator(default_bfe{}) {}
    // Constructor from a UDBFE.
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit batch_fitness_evaluator(T &&x)
        : m_ptr(detail::make_unique<detail::batch_fitness_evaluator_inner<uncvref_t<T>>>(std::forward<T>(x)))
    {
        // Assign the name.
        m_name = ptr()->get_name();
        // Assign the thread safety level.
        m_thread_safety = ptr()->get_thread_safety();
    }
    // Copy constructor.
    batch_fitness_evaluator(const batch_fitness_evaluator &other)
        : m_ptr(other.ptr()->clone()), m_name(other.m_name), m_thread_safety(other.m_thread_safety)
    {
    }
    // Move constructor.
    batch_fitness_evaluator(batch_fitness_evaluator &&other) noexcept
        : m_ptr(std::move(other.m_ptr)), m_name(std::move(other.m_name)),
          m_thread_safety(std::move(other.m_thread_safety))
    {
    }
    // Move assignment operator
    batch_fitness_evaluator &operator=(batch_fitness_evaluator &&other) noexcept
    {
        if (this != &other) {
            m_ptr = std::move(other.m_ptr);
            m_name = std::move(other.m_name);
            m_thread_safety = std::move(other.m_thread_safety);
        }
        return *this;
    }
    // Copy assignment operator
    batch_fitness_evaluator &operator=(const batch_fitness_evaluator &other)
    {
        // Copy ctor + move assignment.
        return *this = batch_fitness_evaluator(other);
    }
    /// Extract a const pointer to the UDBFE used for construction.
    /**
     * This method will extract a const pointer to the internal instance of the UDBFE. If \p T is not the same type
     * as the UDBFE used during construction (after removal of cv and reference qualifiers), this method will
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
     * @return a const pointer to the internal UDBFE, or \p nullptr
     * if \p T does not correspond exactly to the original UDBFE type used
     * in the constructor.
     */
    template <typename T>
    const T *extract() const noexcept
    {
        auto p = dynamic_cast<const detail::batch_fitness_evaluator_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    /// Extract a pointer to the UDBFE used for construction.
    /**
     * This method will extract a pointer to the internal instance of the UDBFE. If \p T is not the same type
     * as the UDBFE used during construction (after removal of cv and reference qualifiers), this method will
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
     *    methods on the internal UDBFE instance. Assigning a new UDBFE via this pointer is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a pointer to the internal UDBFE, or \p nullptr
     * if \p T does not correspond exactly to the original UDBFE type used
     * in the constructor.
     */
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::batch_fitness_evaluator_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    /// Check if the UDBFE used for construction is of type \p T.
    /**
     * @return \p true if the UDBFE used for construction is of type \p T, \p false otherwise.
     */
    template <typename T>
    bool is() const
    {
        return extract<T>() != nullptr;
    }
    /// Call operator.
    /**
     *
     */
    vector_double operator()(const problem &p, const vector_double &dvs) const
    {
        // Check the input dvs.
        detail::bfe_check_input_dvs(p, dvs);
        // Invoke the call operator from the UDBFE.
        auto retval((*ptr())(p, dvs));
        // Check the produced vector of fitnesses.
        detail::bfe_check_output_fvs(p, dvs, retval);
        return retval;
    }
    /// Save to archive.
    /**
     * This method will save \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDBFE and of primitive types.
     */
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(m_ptr, m_name, m_thread_safety);
    }
    /// Load from archive.
    /**
     * This method will deserialize into \p this the content of \p ar.
     *
     * @param ar source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of the UDBFE and of primitive types.
     */
    template <typename Archive>
    void load(Archive &ar)
    {
        // Deserialize in a separate object and move it in later, for exception safety.
        batch_fitness_evaluator tmp_bfe;
        ar(tmp_bfe.m_ptr, tmp_bfe.m_name, tmp_bfe.m_thread_safety);
        *this = std::move(tmp_bfe);
    }

private:
    // Just two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::batch_fitness_evaluator_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::batch_fitness_evaluator_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }

private:
    // Pointer to the inner base bfe
    std::unique_ptr<detail::batch_fitness_evaluator_inner_base> m_ptr;
    // Various properties determined at construction time
    // from the udbfe. These will be constant for the lifetime
    // of bfe, but we cannot mark them as such because we want to be
    // able to assign and deserialise bfes.
    std::string m_name;
    // Thread safety.
    thread_safety m_thread_safety;
};

} // namespace pagmo

PAGMO_REGISTER_BATCH_FITNESS_EVALUATOR(pagmo::default_bfe)

#endif
