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

#ifndef PAGMO_ISLAND_HPP
#define PAGMO_ISLAND_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/any.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_virtual_base_of.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/archipelago_fwd.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/task_queue.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>

#define PAGMO_S11N_ISLAND_EXPORT_KEY(isl)                                                                              \
    BOOST_CLASS_EXPORT_KEY2(pagmo::detail::isl_inner<isl>, "udi " #isl)                                                \
    BOOST_CLASS_TRACKING(pagmo::detail::isl_inner<isl>, boost::serialization::track_never)

#define PAGMO_S11N_ISLAND_IMPLEMENT(isl) BOOST_CLASS_EXPORT_IMPLEMENT(pagmo::detail::isl_inner<isl>)

#define PAGMO_S11N_ISLAND_EXPORT(isl)                                                                                  \
    PAGMO_S11N_ISLAND_EXPORT_KEY(isl)                                                                                  \
    PAGMO_S11N_ISLAND_IMPLEMENT(isl)

namespace pagmo
{

// Fwd declaration.
class PAGMO_PUBLIC island;

/// Detect \p run_evolve() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * void run_evolve(island &) const;
 * @endcode
 * The \p run_evolve() method is part of the interface for the definition of an island
 * (see pagmo::island).
 */
template <typename T>
class has_run_evolve
{
    template <typename U>
    using run_evolve_t = decltype(std::declval<const U &>().run_evolve(std::declval<island &>()));
    static const bool implementation_defined = std::is_same<void, detected_t<run_evolve_t, T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_run_evolve<T>::value;

namespace detail
{

// Specialise this to true in order to disable all the UDI checks and mark a type
// as a UDI regardless of the features provided by it.
// NOTE: this is needed when implementing the machinery for Python islands.
// NOTE: leave this as an implementation detail for now.
template <typename>
struct disable_udi_checks : std::false_type {
};
} // namespace detail

/// Detect user-defined islands (UDI).
/**
 * This type trait will be \p true if \p T is not cv/reference qualified, it is destructible, default, copy and move
 * constructible, and if it satisfies the pagmo::has_run_evolve type trait.
 *
 * Types satisfying this type trait can be used as user-defined islands (UDI) in pagmo::island.
 */
template <typename T>
class is_udi
{
    static const bool implementation_defined
        = detail::disjunction<detail::conjunction<std::is_same<T, uncvref_t<T>>, std::is_default_constructible<T>,
                                                  std::is_copy_constructible<T>, std::is_move_constructible<T>,
                                                  std::is_destructible<T>, has_run_evolve<T>>,
                              detail::disable_udi_checks<T>>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udi<T>::value;

namespace detail
{

struct PAGMO_PUBLIC isl_inner_base {
    virtual ~isl_inner_base() {}
    virtual std::unique_ptr<isl_inner_base> clone() const = 0;
    virtual void run_evolve(island &) const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T>
struct PAGMO_PUBLIC isl_inner final : isl_inner_base {
    // We just need the def ctor, delete everything else.
    isl_inner() = default;
    isl_inner(const isl_inner &) = delete;
    isl_inner(isl_inner &&) = delete;
    isl_inner &operator=(const isl_inner &) = delete;
    isl_inner &operator=(isl_inner &&) = delete;
    // Constructors from T.
    explicit isl_inner(const T &x) : m_value(x) {}
    explicit isl_inner(T &&x) : m_value(std::move(x)) {}
    // The clone method, used in the copy constructor of island.
    virtual std::unique_ptr<isl_inner_base> clone() const override final
    {
        return detail::make_unique<isl_inner>(m_value);
    }
    // The mandatory run_evolve() method.
    virtual void run_evolve(island &isl) const override final
    {
        m_value.run_evolve(isl);
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
        detail::archive(ar, boost::serialization::base_object<isl_inner_base>(*this), m_value);
    }
    T m_value;
};

PAGMO_PUBLIC void wait_f(const std::future<void> &) noexcept;

PAGMO_PUBLIC bool future_has_exception(std::future<void> &) noexcept;

PAGMO_PUBLIC bool future_running(const std::future<void> &);

} // namespace detail

} // namespace pagmo

namespace boost
{

template <typename T>
struct is_virtual_base_of<pagmo::detail::isl_inner_base, pagmo::detail::isl_inner<T>> : false_type {
};

} // namespace boost

namespace pagmo
{

namespace detail
{
// NOTE: this construct is used to create a RAII-style object at the beginning
// of island::wait()/island::wait_check(). Normally this object's constructor and destructor will not
// do anything, but in Python we need to override this getter so that it returns
// a RAII object that unlocks the GIL, otherwise we could run into deadlocks in Python
// if isl::wait()/isl::wait_check() holds the GIL while waiting.
PAGMO_PUBLIC extern std::function<boost::any()> wait_raii_getter;

// NOTE: this structure holds an std::function that implements the logic for the selection of the UDI
// type in the constructor of island_data. The logic is decoupled so that we can override the default logic with
// alternative implementations (e.g., use a process-based island rather than the default thread island if prob, algo,
// etc. do not provide thread safety).
PAGMO_PUBLIC extern std::function<void(const algorithm &, const population &,
                                       std::unique_ptr<detail::isl_inner_base> &)>
    island_factory;

// NOTE: the idea with this class is that we use it to store the data members of pagmo::island, and,
// within pagmo::island, we store a pointer to an instance of this struct. The reason for this approach
// is that, like this, we can provide sensible move semantics: just move the internal pointer of pagmo::island.
struct PAGMO_PUBLIC island_data {
    island_data();
    // This is the main ctor, from an algo and a population. The UDI type will be selected
    // by the island_factory functor.
    template <typename Algo, typename Pop>
    explicit island_data(Algo &&a, Pop &&p)
        : algo(std::make_shared<algorithm>(std::forward<Algo>(a))),
          pop(std::make_shared<population>(std::forward<Pop>(p)))
    {
        island_factory(*algo, *pop, isl_ptr);
    }
    // As above, but the UDI is explicitly passed by the user.
    template <typename Isl, typename Algo, typename Pop>
    explicit island_data(Isl &&isl, Algo &&a, Pop &&p)
        : isl_ptr(detail::make_unique<isl_inner<uncvref_t<Isl>>>(std::forward<Isl>(isl))),
          algo(std::make_shared<algorithm>(std::forward<Algo>(a))),
          pop(std::make_shared<population>(std::forward<Pop>(p)))
    {
    }
    // This is used only in the copy ctor of island. It's equivalent to the ctor from Algo + pop,
    // the island will come from the clone() method of an isl_inner.
    template <typename Algo, typename Pop>
    explicit island_data(std::unique_ptr<isl_inner_base> &&ptr, Algo &&a, Pop &&p)
        : isl_ptr(std::move(ptr)), algo(std::make_shared<algorithm>(std::forward<Algo>(a))),
          pop(std::make_shared<population>(std::forward<Pop>(p)))
    {
    }
    // Delete all the rest, make sure we don't implicitly rely on any of this.
    island_data(const island_data &) = delete;
    island_data(island_data &&) = delete;
    island_data &operator=(const island_data &) = delete;
    island_data &operator=(island_data &&) = delete;
    // The data members.
    // NOTE: isl_ptr has no associated mutex, as it's supposed to be fully
    // thread-safe on its own.
    std::unique_ptr<isl_inner_base> isl_ptr;
    // Algo and pop need a mutex to regulate concurrent access
    // while the island is evolving.
    // NOTE: see the explanation in island::get_algorithm() about why
    // we store algo/pop as shared_ptrs.
    std::mutex algo_mutex;
    std::shared_ptr<algorithm> algo;
    std::mutex pop_mutex;
    std::shared_ptr<population> pop;
    std::vector<std::future<void>> futures;
    // This will be explicitly set only during archipelago::push_back().
    // In all other situations, it will be null.
    archipelago *archi_ptr = nullptr;
    task_queue queue;
};
} // namespace detail

/// Evolution status.
/**
 * This enumeration contains status flags used to represent the current
 * status of asynchronous evolution/optimisation in pagmo::island and pagmo::archipelago.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. seealso::
 *
 *    :cpp:func:`pagmo::island::status()` and :cpp:func:`pagmo::archipelago::status()`.
 *
 * \endverbatim
 */
enum class evolve_status {
    idle = 0,       ///< No asynchronous operations are ongoing, and no error was generated
                    /// by an asynchronous operation in the past
    busy = 1,       ///< Asynchronous operations are ongoing, and no error was generated
                    /// by an asynchronous operation in the past
    idle_error = 2, ///< Idle with error: no asynchronous operations are ongoing, but an error
                    /// was generated by an asynchronous operation in the past
    busy_error = 3  ///< Busy with error: asynchronous operations are ongoing, and an error
                    /// was generated by an asynchronous operation in the past
};

namespace detail
{

// NOTE: in C++11 hashing of enums might not be available. Provide our own.
struct island_status_hasher {
    std::size_t operator()(evolve_status es) const noexcept
    {
        return std::hash<int>{}(static_cast<int>(es));
    }
};

// A map to link a human-readable description to evolve_status.
PAGMO_PUBLIC extern const std::unordered_map<evolve_status, std::string, island_status_hasher> island_statuses;

} // namespace detail

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Provide the stream operator overload for evolve_status.
PAGMO_PUBLIC std::ostream &operator<<(std::ostream &, evolve_status);

#endif

/// Island class.
/**
 * \image html island_no_text.png
 *
 * \verbatim embed:rst:leading-asterisk
 *
 * In the pagmo jargon, an island is a class that encapsulates three entities:
 *
 * - a user-defined island (UDI),
 * - an :cpp:class:`~pagmo::algorithm`,
 * - a :cpp:class:`~pagmo::population`.
 *
 * Through the UDI, the island class manages the asynchronous evolution (or optimisation)
 * of its :cpp:class:`~pagmo::population` via the algorithm's :cpp:func:`~pagmo::algorithm::evolve()`
 * method. Depending on the UDI, the evolution might take place in a separate thread (e.g., if the UDI is a
 * :cpp:class:`~pagmo::thread_island`), in a separate process (see :cpp:class:`~pagmo::fork_island`) or even
 * in a separate machine. The evolution
 * is always asynchronous (i.e., running in the "background") and it is initiated by a call
 * to the :cpp:func:`~pagmo::island::evolve()` method. At any time the user can query the state of the island
 * and fetch its internal data members. The user can explicitly wait for pending evolutions
 * to conclude by calling the :cpp:func:`~pagmo::island::wait()` and :cpp:func:`~pagmo::island::wait_check()`
 * methods. The status of ongoing evolutions in the island can be queried via :cpp:func:`~pagmo::island::status()`.
 *
 * \endverbatim
 *
 * Typically, pagmo users will employ an already-available UDI (such as pagmo::thread_island) in
 * conjunction with this class, but advanced users can implement their own UDI types. A user-defined
 * island must implement the following method:
 * @code{.unparsed}
 * void run_evolve(island &) const;
 * @endcode
 *
 * The <tt>run_evolve()</tt> method of
 * the UDI will use the input island algorithm's algorithm::evolve() method to evolve the input island's
 * pagmo::population. Once the evolution is finished, <tt>run_evolve()</tt> will then replace the population and the
 * algorithm of the input island with, respectively, the evolved population and the algorithm used for the evolution.
 *
 * In addition to the mandatory <tt>run_evolve()</tt> method, a UDI may implement the following optional methods:
 * @code{.unparsed}
 * std::string get_name() const;
 * std::string get_extra_info() const;
 * @endcode
 *
 * See the documentation of the corresponding methods in this class for details on how the optional
 * methods in the UDI are used by pagmo::island.
 *
 * Note that, due to the asynchronous nature of pagmo::island, a UDI has certain requirements regarding thread safety.
 * Specifically, ``run_evolve()`` is always called in a separate thread of execution, and consequently:
 * - multiple UDI objects may be calling their own ``run_evolve()`` method concurrently,
 * - in a specific UDI object, any method from the public API of the UDI may be called while ``run_evolve()`` is
 *   running concurrently in another thread (the only exception being the destructor, which will wait for the end
 *   of any ongoing evolution before taking any action). Thus, UDI writers must ensure that actions such as copying
 *   the UDI, calling the optional methods (such as ``%get_name()``), etc. can be safely performed while the island
 *   is evolving.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from :cpp:class:`pagmo::island` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * \endverbatim
 */
class PAGMO_PUBLIC island
{
    // Handy shortcut.
    using idata_t = detail::island_data;
    // archi needs access to the internal of island.
    friend class PAGMO_PUBLIC archipelago;
    // NOTE: the idea in the move members and the dtor is that
    // we want to wait *and* erase any future in the island, before doing
    // the move/destruction. Thus we use this small wrapper.
    void wait_check_ignore();

public:
    // Default constructor.
    island();
    // Copy constructor.
    island(const island &);
    // Move constructor.
    island(island &&) noexcept;

private:
    template <typename Algo, typename Pop>
    using algo_pop_enabler = enable_if_t<
        detail::conjunction<std::is_constructible<algorithm, Algo &&>, std::is_same<population, uncvref_t<Pop>>>::value,
        int>;

public:
    /// Constructor from algorithm and population.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``a`` can be used to construct a
     *    :cpp:class:`pagmo::algorithm` and :cpp:class:`p` is an instance of :cpp:class:`pagmo::population`.
     *
     * This constructor will use *a* to construct the internal algorithm, and *p* to construct
     * the internal population. The UDI type will be inferred from the :cpp:type:`~pagmo::thread_safety` properties
     * of the algorithm and the population's problem. Specifically:
     *
     * - if both the algorithm and the problem provide at least the basic :cpp:type:`~pagmo::thread_safety`
     *   guarantee, or if the current platform is *not* POSIX, then the UDI type will be
     *   :cpp:class:`~pagmo::thread_island`;
     * - otherwise, the UDI type will be :cpp:class:`~pagmo::fork_island`.
     *
     * Note that on non-POSIX platforms, :cpp:class:`~pagmo::thread_island` will always be selected as the UDI type,
     * but island evolutions will fail if the algorithm and/or problem do not provide at least the
     * basic :cpp:type:`~pagmo::thread_safety` guarantee.
     *
     * \endverbatim
     *
     * @param a the input algorithm.
     * @param p the input population.
     *
     * @throws unspecified any exception thrown by:
     * - memory allocation errors,
     * - the constructors of pagmo::algorithm and pagmo::population.
     */
    template <typename Algo, typename Pop, algo_pop_enabler<Algo, Pop> = 0>
    explicit island(Algo &&a, Pop &&p)
        : m_ptr(detail::make_unique<idata_t>(std::forward<Algo>(a), std::forward<Pop>(p)))
    {
    }

private:
    // NOTE: here and elsewhere, we don't have to put the constraint that Isl is not pagmo::island,
    // like we do in pagmo::problem/pagmo::algorithm: pagmo::island does not satisfy the interface
    // requirements of a UDI, thus it is impossible to create a pagmo::island containing another
    // pagmo::island as a UDI.
    template <typename Isl, typename Algo, typename Pop>
    using isl_algo_pop_enabler
        = enable_if_t<detail::conjunction<is_udi<uncvref_t<Isl>>, std::is_constructible<algorithm, Algo &&>,
                                          std::is_same<population, uncvref_t<Pop>>>::value,
                      int>;

public:
    /// Constructor from UDI, algorithm and population.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if:
     *
     *    - ``Isl`` satisfies :cpp:class:`pagmo::is_udi`,
     *    - ``a`` can be used to construct a :cpp:class:`pagmo::algorithm`,
     *    - ``p`` is an instance of :cpp:class:`pagmo::population`.
     *
     * \endverbatim
     *
     * This constructor will use \p isl to construct the internal UDI, \p a to construct the internal algorithm,
     * and \p p to construct the internal population.
     *
     * @param isl the input UDI.
     * @param a the input algorithm.
     * @param p the input population.
     *
     * @throws unspecified any exception thrown by:
     * - memory allocation errors,
     * - the constructors of \p Isl, pagmo::algorithm and pagmo::population.
     */
    template <typename Isl, typename Algo, typename Pop, isl_algo_pop_enabler<Isl, Algo, Pop> = 0>
    explicit island(Isl &&isl, Algo &&a, Pop &&p)
        : m_ptr(detail::make_unique<idata_t>(std::forward<Isl>(isl), std::forward<Algo>(a), std::forward<Pop>(p)))
    {
    }

private:
    template <typename Algo, typename Prob>
    using algo_prob_enabler = enable_if_t<
        detail::conjunction<std::is_constructible<algorithm, Algo &&>, std::is_constructible<problem, Prob &&>>::value,
        int>;

public:
    /// Constructor from algorithm, problem, size and seed.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``a`` can be used to construct a
     *    :cpp:class:`pagmo::algorithm`, and ``p`` can be used to construct a
     *    :cpp:class:`pagmo::problem`.
     *
     * \endverbatim
     *
     * This constructor will construct a pagmo::population \p pop from \p p, \p size and \p seed, and it will
     * then invoke island(Algo &&, Pop &&) with \p a and \p pop as arguments.
     *
     * @param a the input algorithm.
     * @param p the input problem.
     * @param size the population size.
     * @param seed the population seed.
     *
     * @throws unspecified any exception thrown by the invoked pagmo::population constructor or by
     * island(Algo &&, Pop &&).
     */
    template <typename Algo, typename Prob, algo_prob_enabler<Algo, Prob> = 0>
    explicit island(Algo &&a, Prob &&p, population::size_type size, unsigned seed = pagmo::random_device::next())
        : island(std::forward<Algo>(a), population(std::forward<Prob>(p), size, seed))
    {
    }

private:
    template <typename Algo, typename Prob, typename Bfe>
    using algo_prob_bfe_enabler = enable_if_t<
        detail::conjunction<std::is_constructible<algorithm, Algo &&>, std::is_constructible<problem, Prob &&>,
                            std::is_constructible<bfe, Bfe &&>>::value,
        int>;

public:
    /// Constructor from algorithm, problem, batch fitness evaluator, size and seed.
    /**
     * \verbatim embed:rst:leading-asterisk
     * This constructor is equivalent to the previous one, the only difference being that
     * the population's individuals will be initialised using the input :cpp:class:`~pagmo::bfe`
     * or UDBFE *b*.
     * \endverbatim
     *
     * @param a the input algorithm.
     * @param p the input problem.
     * @param b the input (user-defined) batch fitness evaluator.
     * @param size the population size.
     * @param seed the population seed.
     *
     * @throws unspecified any exception thrown by the invoked pagmo::population constructor or by
     * island(Algo &&, Pop &&).
     */
    template <typename Algo, typename Prob, typename Bfe, algo_prob_bfe_enabler<Algo, Prob, Bfe> = 0>
    explicit island(Algo &&a, Prob &&p, Bfe &&b, population::size_type size,
                    unsigned seed = pagmo::random_device::next())
        : island(std::forward<Algo>(a), population(std::forward<Prob>(p), std::forward<Bfe>(b), size, seed))
    {
    }

private:
    template <typename Isl, typename Algo, typename Prob>
    using isl_algo_prob_enabler
        = enable_if_t<detail::conjunction<is_udi<uncvref_t<Isl>>, std::is_constructible<algorithm, Algo &&>,
                                          std::is_constructible<problem, Prob &&>>::value,
                      int>;

public:
    /// Constructor from UDI, algorithm, problem, size and seed.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``Isl`` satisfies :cpp:class:`pagmo::is_udi`, ``a`` can be used to
     *    construct a :cpp:class:`pagmo::algorithm`, and ``p`` can be used to construct a
     *    :cpp:class:`pagmo::problem`.
     *
     * \endverbatim
     *
     * This constructor will construct a pagmo::population \p pop from \p p, \p size and \p seed, and it will
     * then invoke island(Isl &&, Algo &&, Pop &&) with \p isl, \p a and \p pop as arguments.
     *
     * @param isl the input UDI.
     * @param a the input algorithm.
     * @param p the input problem.
     * @param size the population size.
     * @param seed the population seed.
     *
     * @throws unspecified any exception thrown by:
     * - the invoked pagmo::population constructor,
     * - island(Isl &&, Algo &&, Pop &&),
     * - the invoked constructor of \p Isl.
     */
    template <typename Isl, typename Algo, typename Prob, isl_algo_prob_enabler<Isl, Algo, Prob> = 0>
    explicit island(Isl &&isl, Algo &&a, Prob &&p, population::size_type size,
                    unsigned seed = pagmo::random_device::next())
        : island(std::forward<Isl>(isl), std::forward<Algo>(a), population(std::forward<Prob>(p), size, seed))
    {
    }

private:
    template <typename Isl, typename Algo, typename Prob, typename Bfe>
    using isl_algo_prob_bfe_enabler = enable_if_t<
        detail::conjunction<is_udi<uncvref_t<Isl>>, std::is_constructible<algorithm, Algo &&>,
                            std::is_constructible<problem, Prob &&>, std::is_constructible<bfe, Bfe &&>>::value,
        int>;

public:
    /// Constructor from UDI, algorithm, problem, batch fitness evaluator, size and seed.
    /**
     * \verbatim embed:rst:leading-asterisk
     * This constructor is equivalent to the previous one, the only difference being that
     * the population's individuals will be initialised using the input :cpp:class:`~pagmo::bfe`
     * or UDBFE *b*.
     * \endverbatim
     *
     * @param isl the input UDI.
     * @param a the input algorithm.
     * @param p the input problem.
     * @param b the input (user-defined) batch fitness evaluator.
     * @param size the population size.
     * @param seed the population seed.
     *
     * @throws unspecified any exception thrown by the invoked pagmo::population constructor or by
     * island(Algo &&, Pop &&).
     */
    template <typename Isl, typename Algo, typename Prob, typename Bfe,
              isl_algo_prob_bfe_enabler<Isl, Algo, Prob, Bfe> = 0>
    explicit island(Isl &&isl, Algo &&a, Prob &&p, Bfe &&b, population::size_type size,
                    unsigned seed = pagmo::random_device::next())
        : island(std::forward<Isl>(isl), std::forward<Algo>(a),
                 population(std::forward<Prob>(p), std::forward<Bfe>(b), size, seed))
    {
    }
    // Destructor.
    ~island();
    // Move assignment.
    island &operator=(island &&) noexcept;
    island &operator=(const island &);
    /// Extract a const pointer to the UDI used for construction.
    /**
     * This method will extract a const pointer to the internal instance of the UDI. If \p T is not the same type
     * as the UDI used during construction (after removal of cv and reference qualifiers), this method will
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
     * @return a const pointer to the internal UDI, or \p nullptr
     * if \p T does not correspond exactly to the original UDI type used
     * in the constructor.
     */
    template <typename T>
    const T *extract() const noexcept
    {
        auto isl = dynamic_cast<const detail::isl_inner<T> *>(m_ptr->isl_ptr.get());
        return isl == nullptr ? nullptr : &(isl->m_value);
    }
    /// Extract a pointer to the UDI used for construction.
    /**
     * This method will extract a pointer to the internal instance of the UDI. If \p T is not the same type
     * as the UDI used during construction (after removal of cv and reference qualifiers), this method will
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
     *    methods on the internal UDI instance. Assigning a new UDI via this pointer is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a pointer to the internal UDI, or \p nullptr
     * if \p T does not correspond exactly to the original UDI type used
     * in the constructor.
     */
    template <typename T>
    T *extract() noexcept
    {
        auto isl = dynamic_cast<detail::isl_inner<T> *>(m_ptr->isl_ptr.get());
        return isl == nullptr ? nullptr : &(isl->m_value);
    }
    /// Check if the UDI used for construction is of type \p T.
    /**
     * @return \p true if the UDI used in construction is of type \p T, \p false otherwise.
     */
    template <typename T>
    bool is() const noexcept
    {
        return extract<T>() != nullptr;
    }
    /// Launch evolution.
    /**
     * This method will evolve the island's pagmo::population using the
     * island's pagmo::algorithm. The evolution happens asynchronously:
     * a call to island::evolve() will create an evolution task that will be pushed
     * to a queue, and then return immediately.
     * The tasks in the queue are consumed
     * by a separate thread of execution managed by the pagmo::island object.
     * Each task will invoke the <tt>run_evolve()</tt>
     * method of the UDI \p n times consecutively to perform the actual evolution.
     * The island's population will be updated at the end of each <tt>run_evolve()</tt>
     * invocation. Exceptions raised inside the
     * tasks are stored within the island object, and can be re-raised by calling wait_check().
     *
     * It is possible to call this method multiple times to enqueue multiple evolution tasks, which
     * will be consumed in a FIFO (first-in first-out) fashion. The user may call island::wait() or island::wait_check()
     * to block until all tasks have been completed, and to fetch exceptions raised during the execution of the tasks.
     * island::status() can be used to query the status of the asynchronous operations in the island.
     *
     * @param n the number of times the <tt>run_evolve()</tt> method of the UDI will be called
     * within the evolution task.
     *
     * @throws unspecified any exception thrown by:
     * - threading primitives,
     * - memory allocation errors,
     * - the public interface of \p std::future.
     */
    void evolve(unsigned n = 1);
    // Block until evolution ends and re-raise the first stored exception.
    void wait_check();
    // Block until evolution ends.
    void wait();
    // Status of the island.
    evolve_status status() const;
    // Get the algorithm.
    algorithm get_algorithm() const;
    // Set the algorithm.
    void set_algorithm(algorithm);
    // Get the population.
    population get_population() const;
    // Set the population.
    void set_population(population);
    // Get the thread safety of the island's members.
    std::array<thread_safety, 2> get_thread_safety() const;
    // Island's name.
    std::string get_name() const;
    // Island's extra info.
    std::string get_extra_info() const;
    /// Save to archive.
    /**
     * This method will save \p this to the archive \p ar.
     *
     * It is safe to call this method while the island is evolving.
     *
     * @param ar the target archive.
     *
     * @throws unspecified any exception thrown by:
     * - the serialization of pagmo::algorithm, pagmo::population and of the UDI type,
     * - get_algorithm() and get_population().
     */
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_ptr->isl_ptr, get_algorithm(), get_population());
    }
    /// Load from archive.
    /**
     * This method will load into \p this the content of \p ar.
     * This method will wait until any ongoing evolution in \p this is finished
     * before returning.
     *
     * @param ar the source archive.
     *
     * @throws unspecified any exception thrown by the deserialization of pagmo::algorithm, pagmo::population and of the
     * UDI type.
     */
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        // Deserialize into tmp island, and then move assign it.
        island tmp_island;
        // NOTE: no need to lock access to these, as there is no evolution going on in tmp_island.
        detail::from_archive(ar, tmp_island.m_ptr->isl_ptr, *tmp_island.m_ptr->algo, *tmp_island.m_ptr->pop);
        *this = std::move(tmp_island);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    std::unique_ptr<idata_t> m_ptr;
};

// Stream operator for pagmo::island.
PAGMO_PUBLIC std::ostream &operator<<(std::ostream &, const island &);

} // namespace pagmo

#endif
