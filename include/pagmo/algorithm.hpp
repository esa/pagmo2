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

#ifndef PAGMO_ALGORITHM_HPP
#define PAGMO_ALGORITHM_HPP

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>

#define PAGMO_S11N_ALGORITHM_EXPORT_KEY(algo)                                                                          \
    BOOST_CLASS_EXPORT_KEY2(pagmo::detail::algo_inner<algo>, "uda " #algo)                                             \
    BOOST_CLASS_TRACKING(pagmo::detail::algo_inner<algo>, boost::serialization::track_never)

#define PAGMO_S11N_ALGORITHM_IMPLEMENT(algo) BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::detail::algo_inner<algo>)

#define PAGMO_S11N_ALGORITHM_EXPORT(algo)                                                                              \
    PAGMO_S11N_ALGORITHM_EXPORT_KEY(algo)                                                                              \
    PAGMO_S11N_ALGORITHM_IMPLEMENT(algo)

namespace pagmo
{

/// Detect \p set_verbosity() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * void set_verbosity(unsigned);
 * @endcode
 * The \p set_verbosity() method is part of the interface for the definition of an algorithm
 * (see pagmo::algorithm).
 */
template <typename T>
class has_set_verbosity
{
    template <typename U>
    using set_verbosity_t = decltype(std::declval<U &>().set_verbosity(1u));
    static const bool implementation_defined = std::is_same<void, detected_t<set_verbosity_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_set_verbosity<T>::value;

/// Detect \p has_set_verbosity() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * bool has_set_verbosity() const;
 * @endcode
 * The \p has_set_verbosity() method is part of the interface for the definition of an algorithm
 * (see pagmo::algorithm).
 */
template <typename T>
class override_has_set_verbosity
{
    template <typename U>
    using has_set_verbosity_t = decltype(std::declval<const U &>().has_set_verbosity());
    static const bool implementation_defined = std::is_same<bool, detected_t<has_set_verbosity_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool override_has_set_verbosity<T>::value;

/// Detect \p evolve() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * population evolve(const population &) const;
 * @endcode
 * The \p evolve() method is part of the interface for the definition of an algorithm
 * (see pagmo::algorithm).
 */
template <typename T>
class has_evolve
{
    template <typename U>
    using evolve_t = decltype(std::declval<const U &>().evolve(std::declval<const population &>()));
    static const bool implementation_defined = std::is_same<population, detected_t<evolve_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_evolve<T>::value;

namespace detail
{

// Specialise this to true in order to disable all the UDA checks and mark a type
// as a UDA regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python algos.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_uda_checks : std::false_type {
};
} // namespace detail

/// Detect user-defined algorithms (UDA).
/**
 * This type trait will be \p true if \p T is not cv/reference qualified, it is destructible, default, copy and move
 * constructible, and if it satisfies the pagmo::has_evolve type trait.
 *
 * Types satisfying this type trait can be used as user-defined algorithms (UDA) in pagmo::algorithm.
 */
template <typename T>
class is_uda
{
    static const bool implementation_defined
        = detail::disjunction<detail::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                                  std::is_copy_constructible<T>, std::is_move_constructible<T>,
                                                  std::is_destructible<T>, has_evolve<T>>,
                              detail::disable_uda_checks<T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_uda<T>::value;

namespace detail
{

struct PAGMO_DLL_PUBLIC_INLINE_CLASS algo_inner_base {
    virtual ~algo_inner_base() {}
    virtual std::unique_ptr<algo_inner_base> clone() const = 0;
    virtual population evolve(const population &pop) const = 0;
    virtual void set_seed(unsigned) = 0;
    virtual bool has_set_seed() const = 0;
    virtual void set_verbosity(unsigned) = 0;
    virtual bool has_set_verbosity() const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    virtual thread_safety get_thread_safety() const = 0;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
struct PAGMO_DLL_PUBLIC_INLINE_CLASS algo_inner final : algo_inner_base {
    // We just need the def ctor, delete everything else.
    algo_inner() = default;
    algo_inner(const algo_inner &) = delete;
    algo_inner(algo_inner &&) = delete;
    algo_inner &operator=(const algo_inner &) = delete;
    algo_inner &operator=(algo_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit algo_inner(const T &x) : m_value(x) {}
    explicit algo_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of algorithm.
    virtual std::unique_ptr<algo_inner_base> clone() const override final
    {
        return detail::make_unique<algo_inner>(m_value);
    }
    // Mandatory methods.
    virtual population evolve(const population &pop) const override final
    {
        return m_value.evolve(pop);
    }
    // Optional methods
    virtual void set_seed(unsigned seed) override final
    {
        set_seed_impl(m_value, seed);
    }
    virtual bool has_set_seed() const override final
    {
        return has_set_seed_impl(m_value);
    }
    virtual void set_verbosity(unsigned level) override final
    {
        set_verbosity_impl(m_value, level);
    }
    virtual bool has_set_verbosity() const override final
    {
        return has_set_verbosity_impl(m_value);
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
    template <typename U, enable_if_t<pagmo::has_set_seed<U>::value, int> = 0>
    static void set_seed_impl(U &a, unsigned seed)
    {
        a.set_seed(seed);
    }
    template <typename U, enable_if_t<!pagmo::has_set_seed<U>::value, int> = 0>
    static void set_seed_impl(U &, unsigned)
    {
        pagmo_throw(not_implemented_error,
                    "The set_seed() method has been invoked but it is not implemented in the UDA");
    }
    template <typename U,
              enable_if_t<detail::conjunction<pagmo::has_set_seed<U>, override_has_set_seed<U>>::value, int> = 0>
    static bool has_set_seed_impl(const U &a)
    {
        return a.has_set_seed();
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
    template <typename U, enable_if_t<pagmo::has_set_verbosity<U>::value, int> = 0>
    static void set_verbosity_impl(U &value, unsigned level)
    {
        value.set_verbosity(level);
    }
    template <typename U, enable_if_t<!pagmo::has_set_verbosity<U>::value, int> = 0>
    static void set_verbosity_impl(U &, unsigned)
    {
        pagmo_throw(not_implemented_error,
                    "The set_verbosity() method has been invoked but it is not implemented in the UDA");
    }
    template <
        typename U,
        enable_if_t<detail::conjunction<pagmo::has_set_verbosity<U>, override_has_set_verbosity<U>>::value, int> = 0>
    static bool has_set_verbosity_impl(const U &a)
    {
        return a.has_set_verbosity();
    }
    template <typename U, enable_if_t<detail::conjunction<pagmo::has_set_verbosity<U>,
                                                          detail::negation<override_has_set_verbosity<U>>>::value,
                                      int> = 0>
    static bool has_set_verbosity_impl(const U &)
    {
        return true;
    }
    template <typename U, enable_if_t<!pagmo::has_set_verbosity<U>::value, int> = 0>
    static bool has_set_verbosity_impl(const U &)
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

    // Serialization
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        detail::archive(ar, boost::serialization::base_object<algo_inner_base>(*this), m_value);
    }
    T m_value;
};

} // namespace detail

} // namespace pagmo

namespace boost
{

template <typename T>
struct is_virtual_base_of<pagmo::detail::algo_inner_base, pagmo::detail::algo_inner<T>> : false_type {
};

} // namespace boost

namespace pagmo
{

/// Algorithm class.
/**
 * \image html algo_no_text.png
 *
 * This class represents an optimization algorithm. An algorithm can be
 * stochastic, deterministic, population based, derivative-free, using hessians,
 * using gradients, a meta-heuristic, evolutionary, etc.. Via this class pagmo offers a common interface to
 * all types of algorithms that can be applied to find solution to a generic matematical
 * programming problem as represented by the pagmo::problem class.
 *
 * In order to define an optimizaztion algorithm in pagmo, the user must first define a class
 * (or a struct) whose methods describe the properties of the algorithm and implement its logic.
 * In pagmo, we refer to such a class as a **user-defined algorithm**, or UDA for short. Once defined and instantiated,
 * a UDA can then be used to construct an instance of this class, pagmo::algorithm, which
 * provides a generic interface to optimization algorithms.
 *
 * Every UDA must implement at least the following method:
 * @code{.unparsed}
 * population evolve(const population&) const;
 * @endcode
 *
 * The <tt>%evolve()</tt> method takes as input a pagmo::population, and it is expected to return
 * a new population generated by the *evolution* (or *optimisation*) of the original population.
 * In addition to providing the above method, a UDA must also be default, copy and move constructible.
 *
 * Additional optional methods can be implemented in a UDA:
 * @code{.unparsed}
 * void set_seed(unsigned);
 * bool has_set_seed() const;
 * void set_verbosity(unsigned);
 * bool has_set_verbosity() const;
 * std::string get_name() const;
 * std::string get_extra_info() const;
 * thread_safety get_thread_safety() const;
 * @endcode
 *
 * See the documentation of the corresponding methods in this class for details on how the optional
 * methods in the UDA are used by pagmo::algorithm.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The only operations allowed on a moved-from :cpp:class:`pagmo::algorithm` are destruction,
 *    assignment, and the invocation of the :cpp:func:`~pagmo::algorithm::is_valid()` member function.
 *    Any other operation will result in undefined behaviour.
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC algorithm
{
    // Enable the generic ctor only if T is not an algorithm (after removing
    // const/reference qualifiers), and if T is a uda.
    template <typename T>
    using generic_ctor_enabler = enable_if_t<
        detail::conjunction<detail::negation<std::is_same<algorithm, uncvref_t<T>>>, is_uda<uncvref_t<T>>>::value, int>;

public:
    // Default constructor.
    algorithm();

private:
    void generic_ctor_impl();

public:
    /// Constructor from a user-defined algorithm of type \p T
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is not enabled if, after the removal of cv and reference qualifiers,
     *    ``T`` is of type :cpp:class:`pagmo::algorithm` (that is, this constructor does not compete with the copy/move
     *    constructors of :cpp:class:`pagmo::algorithm`), or if  ``T`` does not satisfy :cpp:class:`pagmo::is_uda`.
     *
     * \endverbatim
     *
     * This constructor will construct a pagmo::algorithm from the UDA (user-defined algorithm) \p x of type \p T. In
     * order for the construction to be successful, the UDA must implement a minimal set of methods,
     * as described in the documentation of pagmo::algorithm. The constructor will examine the properties of \p x and
     * store them as data members of \p this.
     *
     * @param x the UDA.
     *
     * @throws unspecified any exception thrown by methods of the UDA invoked during construction or by memory errors
     * in strings and standard containers.
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    explicit algorithm(T &&x) : m_ptr(detail::make_unique<detail::algo_inner<uncvref_t<T>>>(std::forward<T>(x)))
    {
        generic_ctor_impl();
    }
    // Copy constructor
    algorithm(const algorithm &);
    // Move ctor.
    algorithm(algorithm &&) noexcept;
    // Move assignment.
    algorithm &operator=(algorithm &&) noexcept;
    // Copy assignment.
    algorithm &operator=(const algorithm &);
    /// Assignment from a user-defined algorithm of type \p T
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This operator is not enabled if, after the removal of cv and reference qualifiers,
     *    ``T`` is of type :cpp:class:`pagmo::algorithm` (that is, this operator does not compete with the copy/move
     *    assignment operators of :cpp:class:`pagmo::algorithm`), or if ``T`` does not satisfy
     *    :cpp:class:`pagmo::is_uda`.
     *
     * \endverbatim
     *
     * This operator will set the internal UDA to ``x`` by constructing a pagmo::algorithm from ``x``, and then
     * move-assigning the result to ``this``.
     *
     * @param x the UDA.
     *
     * @return a reference to ``this``.
     *
     * @throws unspecified any exception thrown by the constructor from UDA.
     */
    template <typename T, generic_ctor_enabler<T> = 0>
    algorithm &operator=(T &&x)
    {
        return (*this) = algorithm(std::forward<T>(x));
    }

    /// Extract a const pointer to the UDA.
    /**
     * This method will extract a const pointer to the internal instance of the UDA. If \p T is not the same type
     * as the UDA used during construction (after removal of cv and reference qualifiers), this method will
     * return \p nullptr.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime of
     *    ``this`` and ``delete`` must never be called on the pointer.
     *
     * \endverbatim
     *
     * @return a const pointer to the internal UDA, or \p nullptr
     * if \p T does not correspond exactly to the original UDA type used
     * in the constructor.
     */
    template <typename T>
    const T *extract() const noexcept
    {
        auto p = dynamic_cast<const detail::algo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    /// Extract a pointer to the UDA.
    /**
     * This method will extract a pointer to the internal instance of the UDA. If \p T is not the same type
     * as the UDA used during construction (after removal of cv and reference qualifiers), this method will
     * return \p nullptr.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
     *    of ``this`` and ``delete`` must never be called on the pointer.
     *
     * \endverbatim
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    The ability to extract a mutable pointer is provided only in order to allow to call non-const
     *    methods on the internal UDA instance. Assigning a new UDA via this pointer is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a pointer to the internal UDA, or \p nullptr
     * if \p T does not correspond exactly to the original UDA type used
     * in the constructor.
     */
    template <typename T>
    T *extract() noexcept
    {
        auto p = dynamic_cast<detail::algo_inner<T> *>(ptr());
        return p == nullptr ? nullptr : &(p->m_value);
    }

    /// Checks the user-defined algorithm type at run-time.
    /**
     * @return \p true if the user-defined algorithm is \p T, \p false otherwise.
     */
    template <typename T>
    bool is() const noexcept
    {
        return extract<T>() != nullptr;
    }

    // Evolve method.
    population evolve(const population &) const;

    // Set the seed for the stochastic evolution.
    void set_seed(unsigned);

    /// Check if a <tt>%set_seed()</tt> method is available in the UDA.
    /**
     * This method will return \p true if a <tt>%set_seed()</tt> method is available in the UDA, \p false otherwise.
     *
     * The availability of the a <tt>%set_seed()</tt> method is determined as follows:
     * - if the UDA does not satisfy pagmo::has_set_seed, then this method will always return \p false;
     * - if the UDA satisfies pagmo::has_set_seed but it does not satisfy pagmo::override_has_set_seed,
     *   then this method will always return \p true;
     * - if the UDA satisfies both pagmo::has_set_seed and pagmo::override_has_set_seed,
     *   then this method will return the output of the <tt>%has_set_seed()</tt> method of the UDA.
     *
     * @return a flag signalling the availability of the <tt>%set_seed()</tt> method in the UDA.
     */
    bool has_set_seed() const
    {
        return m_has_set_seed;
    }

    /// Alias for algorithm::has_set_seed().
    /**
     * @return the output of algorithm::has_set_seed().
     */
    bool is_stochastic() const
    {
        return has_set_seed();
    }

    // Set the verbosity of logs and screen output.
    void set_verbosity(unsigned);

    /// Check if a <tt>%set_verbosity()</tt> method is available in the UDA.
    /**
     * This method will return \p true if a <tt>%set_verbosity()</tt> method is available in the UDA, \p false
     * otherwise.
     *
     * The availability of the a <tt>%set_verbosity()</tt> method is determined as follows:
     * - if the UDA does not satisfy pagmo::has_set_verbosity, then this method will always return \p false;
     * - if the UDA satisfies pagmo::has_set_verbosity but it does not satisfy pagmo::override_has_set_verbosity,
     *   then this method will always return \p true;
     * - if the UDA satisfies both pagmo::has_set_verbosity and pagmo::override_has_set_verbosity,
     *   then this method will return the output of the <tt>%has_set_verbosity()</tt> method of the UDA.
     *
     * @return a flag signalling the availability of the <tt>%set_verbosity()</tt> method in the UDA.
     */
    bool has_set_verbosity() const
    {
        return m_has_set_verbosity;
    }

    /// Algorithm's name.
    /**
     * If the UDA satisfies pagmo::has_name, then this method will return the output of its <tt>%get_name()</tt> method.
     * Otherwise, an implementation-defined name based on the type of the UDA will be returned.
     *
     * @return the algorithm's name.
     *
     * @throws unspecified any exception thrown by copying an \p std::string object.
     */
    std::string get_name() const
    {
        return m_name;
    }

    // Algorithm's extra info.
    std::string get_extra_info() const;

    /// Algorithm's thread safety level.
    /**
     * If the UDA satisfies pagmo::has_get_thread_safety, then this method will return the output of its
     * <tt>%get_thread_safety()</tt> method. Otherwise, thread_safety::basic will be returned.
     * That is, pagmo assumes by default that is it safe to operate concurrently on distinct UDA instances.
     *
     * @return the thread safety level of the UDA.
     */
    thread_safety get_thread_safety() const
    {
        return m_thread_safety;
    }

    // Check if the algorithm is valid.
    bool is_valid() const;

    /// Save to archive.
    /**
     * This method will save \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDA and of primitive types.
     */
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_ptr, m_has_set_seed, m_has_set_verbosity, m_name, m_thread_safety);
    }
    /// Load from archive.
    /**
     * This method will load a pagmo::algorithm from \p ar into \p this.
     *
     * @param ar source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of the UDA and of primitive types.
     */
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        algorithm tmp;
        detail::from_archive(ar, tmp.m_ptr, tmp.m_has_set_seed, tmp.m_has_set_verbosity, tmp.m_name,
                             tmp.m_thread_safety);
        *this = std::move(tmp);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    // Two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::algo_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::algo_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }

private:
    std::unique_ptr<detail::algo_inner_base> m_ptr;
    // Various algorithm properties determined at construction time
    // from the concrete algorithm. These will be constant for the lifetime
    // of algorithm, but we cannot mark them as such because of serialization.
    // the extra_info string cannot be here as it must reflect the changes from set_seed
    bool m_has_set_seed;
    bool m_has_set_verbosity;
    std::string m_name;
    thread_safety m_thread_safety;
};

// Streaming operator for algorithm.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const algorithm &);

} // namespace pagmo

// Disable tracking for the serialisation of algorithm.
BOOST_CLASS_TRACKING(pagmo::algorithm, boost::serialization::track_never)

#endif
