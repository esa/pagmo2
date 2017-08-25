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

#ifndef PAGMO_ISLAND_HPP
#define PAGMO_ISLAND_HPP

#include <algorithm>
#include <array>
#include <boost/any.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <functional>
#include <future>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/task_queue.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/type_traits.hpp>

/// Macro for the registration of the serialization functionality for user-defined islands.
/**
 * This macro should always be invoked after the declaration of a user-defined island: it will register
 * the island with pagmo's serialization machinery. The macro should be called in the root namespace
 * and using the fully qualified name of the island to be registered. For example:
 * @code{.unparsed}
 * namespace my_namespace
 * {
 *
 * class my_island
 * {
 *    // ...
 * };
 *
 * }
 *
 * PAGMO_REGISTER_ISLAND(my_namespace::my_island)
 * @endcode
 */
#define PAGMO_REGISTER_ISLAND(isl) CEREAL_REGISTER_TYPE_WITH_NAME(pagmo::detail::isl_inner<isl>, "udi " #isl)

namespace pagmo
{

// Fwd declaration.
class island;

/// Group of migrating individuals.
/**
 * This triple is used to represent groups of individuals migrating within a pagmo::archipelago.
 * The three elements of the tuple represent, respectively, the IDs, decision vectors and
 * fitness vectors of the migrants. This type, in other words, is a stripped down version
 * of pagmo::population containing only the data necessary for migration.
 */
typedef std::tuple<std::vector<unsigned long long>, std::vector<vector_double>, std::vector<vector_double>> migrants_t;

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
}

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
        = (std::is_same<T, uncvref_t<T>>::value && std::is_default_constructible<T>::value
           && std::is_copy_constructible<T>::value && std::is_move_constructible<T>::value
           && std::is_destructible<T>::value && has_run_evolve<T>::value)
          || detail::disable_udi_checks<T>::value;

public:
    /// Value of the type trait.
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udi<T>::value;

namespace detail
{

struct isl_inner_base {
    virtual ~isl_inner_base()
    {
    }
    virtual std::unique_ptr<isl_inner_base> clone() const = 0;
    virtual void run_evolve(island &) const = 0;
    virtual std::string get_name() const = 0;
    virtual std::string get_extra_info() const = 0;
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

template <typename T>
struct isl_inner final : isl_inner_base {
    // We just need the def ctor, delete everything else.
    isl_inner() = default;
    isl_inner(const isl_inner &) = delete;
    isl_inner(isl_inner &&) = delete;
    isl_inner &operator=(const isl_inner &) = delete;
    isl_inner &operator=(isl_inner &&) = delete;
    // Constructors from T.
    explicit isl_inner(const T &x) : m_value(x)
    {
    }
    explicit isl_inner(T &&x) : m_value(std::move(x))
    {
    }
    // The clone method, used in the copy constructor of island.
    virtual std::unique_ptr<isl_inner_base> clone() const override final
    {
        return make_unique<isl_inner>(m_value);
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
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<isl_inner_base>(this), m_value);
    }
    T m_value;
};

// NOTE: this is just a simple wrapper to force noexcept behaviour on std::future::wait().
// If f.wait() throws something, the program will terminate. A valid std::future should not
// throw, but technically the standard does not guarantee that. Having this noexcept wrapper
// simplifies reasoning about exception behaviour in wait(), wait_check(), etc.
inline void wait_f(const std::future<void> &f) noexcept
{
    assert(f.valid());
    f.wait();
}

// Small helper to determine if a future holds an exception.
// The noexcept reasoning is the same as above. Here we could fail
// because of memory errors, but there's not much we can do in such
// a case.
inline bool future_has_exception(std::future<void> &f) noexcept
{
    assert(f.valid());
    // Try to get the error.
    try {
        f.get();
    } catch (...) {
        // An error was generated. Re-store the exception into the future
        // and return true.
        // http://en.cppreference.com/w/cpp/experimental/make_exceptional_future
        std::promise<void> p;
        p.set_exception(std::current_exception());
        f = p.get_future();
        return true;
    }
    // No error was raised. Need to reconstruct f to a valid state.
    std::promise<void> p;
    p.set_value();
    f = p.get_future();
    return false;
}

// Small helper to check if a future is still running.
inline bool future_running(const std::future<void> &f)
{
    return f.wait_for(std::chrono::duration<int>::zero()) != std::future_status::ready;
}
}

/// Thread island.
/**
 * This class is a user-defined island (UDI) that will run evolutions directly inside
 * the separate thread of execution within pagmo::island.
 */
class thread_island
{
public:
    /// Island's name.
    /**
     * @return <tt>"Thread island"</tt>.
     */
    std::string get_name() const
    {
        return "Thread island";
    }
    void run_evolve(island &) const;
    /// Serialization support.
    /**
     * This class is stateless, no data will be saved to or loaded from the archive.
     */
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

class archipelago;

namespace detail
{
// NOTE: this construct is used to create a RAII-style object at the beginning
// of island::wait()/island::wait_check(). Normally this object's constructor and destructor will not
// do anything, but in Python we need to override this getter so that it returns
// a RAII object that unlocks the GIL, otherwise we could run into deadlocks in Python
// if isl::wait()/isl::wait_check() holds the GIL while waiting.
template <typename = void>
struct wait_raii {
    static std::function<boost::any()> getter;
};

// NOTE: the default implementation just returns a defcted boost::any, whose ctor and dtor
// will have no effect.
template <typename T>
std::function<boost::any()> wait_raii<T>::getter = []() { return boost::any{}; };

// NOTE: this structure holds an std::function that implements the logic for the selection of the UDI
// type in the constructor of island_data. The logic is decoupled so that we can override the default logic with
// alternative implementations (e.g., use a process-based island rather than the default thread island if prob, algo,
// etc. do not provide thread safety).
template <typename = void>
struct island_factory {
    using func_t
        = std::function<void(const algorithm &, const population &, std::unique_ptr<detail::isl_inner_base> &)>;
    static func_t s_func;
};

// This is the default UDI type selector. It will always select thread_island as UDI.
inline void default_island_factory(const algorithm &, const population &, std::unique_ptr<detail::isl_inner_base> &ptr)
{
    ptr = make_unique<isl_inner<thread_island>>();
}

// Static init.
template <typename T>
typename island_factory<T>::func_t island_factory<T>::s_func = default_island_factory;

// NOTE: the idea with this class is that we use it to store the data members of pagmo::island, and,
// within pagmo::island, we store a pointer to an instance of this struct. The reason for this approach
// is that, like this, we can provide sensible move semantics: just move the internal pointer of pagmo::island.
struct island_data {
    // NOTE: thread_island is ok as default choice, as the null_prob/null_algo
    // are both thread safe.
    island_data()
        : isl_ptr(make_unique<isl_inner<thread_island>>()), algo(std::make_shared<algorithm>()),
          pop(std::make_shared<population>())
    {
    }
    // This is the main ctor, from an algo and a population. The UDI type will be selected
    // by the island_factory functor.
    template <typename Algo, typename Pop>
    explicit island_data(Algo &&a, Pop &&p)
        : algo(std::make_shared<algorithm>(std::forward<Algo>(a))),
          pop(std::make_shared<population>(std::forward<Pop>(p)))
    {
        island_factory<>::s_func(*algo, *pop, isl_ptr);
    }
    // As above, but the UDI is explicitly passed by the user.
    template <typename Isl, typename Algo, typename Pop>
    explicit island_data(Isl &&isl, Algo &&a, Pop &&p)
        : isl_ptr(make_unique<isl_inner<uncvref_t<Isl>>>(std::forward<Isl>(isl))),
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
    // threads-safe on its own.
    std::unique_ptr<isl_inner_base> isl_ptr;
    // Algo and pop need a mutex to regulate concurrent access
    // while the island is evolving.
    std::mutex algo_mutex;
    // NOTE: see the explanation in island::get_algorithm() about why
    // we store algo/pop as shared_ptrs.
    std::shared_ptr<algorithm> algo;
    std::mutex pop_mutex;
    std::shared_ptr<population> pop;
    std::vector<std::future<void>> futures;
    // This will be explicitly set only during archipelago::push_back().
    // In all other situations, it will be null.
    archipelago *archi_ptr = nullptr;
    task_queue queue;
};
}

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

template <typename = void>
struct island_static_data {
    // A map to link a human-readable description to evolve_status.
    // NOTE: in C++11 hashing of enums might not be available. Provide our own.
    struct status_hasher {
        std::size_t operator()(evolve_status es) const
        {
            return std::hash<int>{}(static_cast<int>(es));
        }
    };
    using status_map_t = std::unordered_map<evolve_status, std::string, status_hasher>;
    static const status_map_t statuses;
};

template <typename T>
const typename island_static_data<T>::status_map_t island_static_data<T>::statuses
    = {{evolve_status::idle, "idle"},
       {evolve_status::busy, "busy"},
       {evolve_status::idle_error, "idle - **error occurred**"},
       {evolve_status::busy_error, "busy - **error occurred**"}};
}

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Provide the stream operator overload for evolve_status.
inline std::ostream &operator<<(std::ostream &os, evolve_status es)
{
    return os << detail::island_static_data<>::statuses.at(es);
}

#endif

/// Island class.
/**
 * \image html island_no_text.png
 *
 * In the pagmo jargon, an island is a class that encapsulates three entities:
 * - a user-defined island (UDI),
 * - a pagmo::algorithm,
 * - a pagmo::population.
 *
 * Through the UDI, the island class manages the asynchronous evolution (or optimisation)
 * of its pagmo::population via the algorithm's algorithm::evolve() method. Depending
 * on the UDI, the evolution might take place in a separate thread (e.g., if the UDI is a
 * pagmo::thread_island), in a separate process or even in a separate machine. The evolution
 * is always asynchronous (i.e., running in the "background") and it is initiated by a call
 * to the evolve() method. At any time the user can query the state of the island
 * and fetch its internal data members. The user can explicitly wait for pending evolutions
 * to conclude by calling the wait() and wait_check() methods. The status of
 * ongoing evolutions in the island can be queried via status().
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
 * pagmo::population and, once the evolution is finished, will replace the population of the input island with the
 * evolved population. Since internally the pagmo::island class uses a separate thread of execution to provide
 * asynchronous behaviour, a UDI needs to guarantee a certain degree of thread-safety: it must be possible to interact
 * with the UDI while evolution is ongoing (e.g., it must be possible to copy the UDI while evolution is undergoing, or
 * call the <tt>%get_name()</tt>, <tt>%get_extra_info()</tt> methods, etc.), otherwise the behaviour will be undefined.
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
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from :cpp:class:`pagmo::island` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * \endverbatim
 */
class island
{
    // Handy shortcut.
    using idata_t = detail::island_data;
    // archi needs access to the internal of island.
    friend class archipelago;
    // NOTE: the idea in the move members and the dtor is that
    // we want to wait *and* erase any future in the island, before doing
    // the move/destruction. Thus we use this small wrapper.
    void wait_check_ignore()
    {
        try {
            wait_check();
            // LCOV_EXCL_START
        } catch (...) {
        }
        // LCOV_EXCL_STOP
    }

public:
    /// Default constructor.
    /**
     * The default constructor will initialise an island containing a UDI of type pagmo::thread_island,
     * and default-constructed pagmo::algorithm and pagmo::population.
     *
     * @throws unspecified any exception thrown by any invoked constructor or by memory allocation failures.
     */
    island() : m_ptr(detail::make_unique<idata_t>())
    {
    }
    /// Copy constructor.
    /**
     * The copy constructor will initialise an island containing a copy of <tt>other</tt>'s UDI, population
     * and algorithm. It is safe to call this constructor while \p other is evolving.
     *
     * @param other the island tht will be copied.
     *
     * @throws unspecified any exception thrown by:
     * - get_population() and get_algorithm(),
     * - memory allocation errors,
     * - the copy constructors of pagmo::algorithm and pagmo::population.
     */
    island(const island &other)
        : m_ptr(detail::make_unique<idata_t>(other.m_ptr->isl_ptr->clone(), other.get_algorithm(),
                                             other.get_population()))
    {
        // NOTE: the idata_t ctor will set the archi ptr to null. The archi ptr is never copied.
        assert(m_ptr->archi_ptr == nullptr);
    }
    /// Move constructor.
    /**
     * The move constructor will transfer the state of \p other into \p this, after any ongoing
     * evolution in \p other is finished.
     *
     * @param other the island that will be moved.
     */
    island(island &&other) noexcept
    {
        other.wait_check_ignore();
        m_ptr = std::move(other.m_ptr);
    }

private:
    template <typename Algo, typename Pop>
    using algo_pop_enabler = enable_if_t<std::is_constructible<algorithm, Algo &&>::value
                                             && std::is_same<population, uncvref_t<Pop>>::value,
                                         int>;

public:
    /// Constructor from algorithm and population.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``a`` can be used to construct a
     *    :cpp:class`pagmo::algorithm` and :cpp:class:`p` is an instance of :cpp:class:`pagmo::population`.
     *
     * \endverbatim
     *
     * This constructor will use \p a to construct the internal algorithm, and \p p to construct
     * the internal population. A default-constructed pagmo::thread_island will be the internal UDI.
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
    template <typename Isl, typename Algo, typename Pop>
    using isl_algo_pop_enabler
        = enable_if_t<is_udi<uncvref_t<Isl>>::value && std::is_constructible<algorithm, Algo &&>::value
                          && std::is_same<population, uncvref_t<Pop>>::value,
                      int>;

public:
    /// Constructor from UDI, algorithm and population.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if:
     *     - ``Isl`` satisfies :cpp:class::`pagmo::is_udi`,
     *     - ``a`` can be used to construct a :cpp:class:`pagmo::algorithm`,
     *     - ``p`` is an instance of pagmo::population.
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
    using algo_prob_enabler
        = enable_if_t<std::is_constructible<algorithm, Algo &&>::value
                          && std::is_constructible<population, Prob &&, population::size_type, unsigned>::value,
                      int>;

public:
    /// Constructor from algorithm, problem, size and seed.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``a`` can be used to construct a
     *    pagmo::algorithm, and ``p``, ``size`` and ``seed`` can be used to construct a :cpp:class:`pagmo::population`.
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
    template <typename Isl, typename Algo, typename Prob>
    using isl_algo_prob_enabler
        = enable_if_t<is_udi<uncvref_t<Isl>>::value && std::is_constructible<algorithm, Algo &&>::value
                          && std::is_constructible<population, Prob &&, population::size_type, unsigned>::value,
                      int>;

public:
    /// Constructor from UDI, algorithm, problem, size and seed.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if ``Isl`` satisfies :cpp:class:`pagmo::is_udi`, ``a`` can be used to
     *    construct a :cpp:class:`pagmo::algorithm`, and ``p``, ``size`` and ``seed`` can be used to construct a
     *    :cpp:class:`pagmo::population`.
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
    /// Destructor.
    /**
     * If the island has not been moved-from, the destructor will call island::wait_check(),
     * ignoring any exception that might be thrown.
     */
    ~island()
    {
        // If the island has been moved from, don't do anything.
        if (m_ptr) {
            wait_check_ignore();
        }
    }
    /// Move assignment.
    /**
     * Move assignment will transfer the state of \p other into \p this, after any ongoing
     * evolution in \p this and \p other is finished.
     *
     * @param other the island tht will be moved.
     *
     * @return a reference to \p this.
     */
    island &operator=(island &&other) noexcept
    {
        if (this != &other) {
            if (m_ptr) {
                wait_check_ignore();
            }
            other.wait_check_ignore();
            m_ptr = std::move(other.m_ptr);
        }
        return *this;
    }
    /// Copy assignment.
    /**
     * Copy assignment is implemented as copy construction followed by move assignment.
     *
     * @param other the island tht will be copied.
     *
     * @return a reference to \p this.
     *
     * @throws unspecified any exception thrown by the copy constructor.
     */
    island &operator=(const island &other)
    {
        if (this != &other) {
            *this = island(other);
        }
        return *this;
    }
    void evolve(unsigned = 1);
    /// Block until evolution ends and re-raise the first stored exception.
    /**
     * This method will block until all the evolution tasks enqueued via island::evolve() have been completed.
     * If one task enqueued after the last call to wait_check() threw an exception, the exception will be re-thrown
     * by this method. If more than one task enqueued after the last call to wait_check() threw an exception,
     * this method will re-throw the exception raised by the first enqueued task that threw, and the exceptions
     * from all the other tasks that threw will be ignored.
     *
     * Note that wait_check() resets the status of the island: after a call to wait_check(), status() will always
     * return evolve_status::idle.
     *
     * @throws unspecified any exception thrown by evolution tasks.
     */
    void wait_check()
    {
        auto iwr = detail::wait_raii<>::getter();
        (void)iwr;
        for (auto it = m_ptr->futures.begin(); it != m_ptr->futures.end(); ++it) {
            assert(it->valid());
            try {
                it->get();
            } catch (...) {
                // If any of the futures stores an exception, we will re-raise it.
                // But first, we need to get all the other futures and erase the futures
                // vector.
                // NOTE: everything is this block is noexcept.
                for (it = it + 1; it != m_ptr->futures.end(); ++it) {
                    detail::wait_f(*it);
                }
                m_ptr->futures.clear();
                throw;
            }
        }
        m_ptr->futures.clear();
    }
    /// Block until evolution ends.
    /**
     * This method will block until all the evolution tasks enqueued via island::evolve() have been completed.
     * Exceptions thrown by the enqueued tasks can be re-raised via wait_check(): they will **not** be re-thrown
     * by this method. Also, contrary to wait_check(), this method will **not** reset the status of the island:
     * after a call to wait(), status() will always return either evolve_status::idle or evolve_status::idle_error.
     */
    void wait()
    {
        // NOTE: we use this function in move ops and in the dtor, which are all noexcept. In theory we could
        // end up aborting in case the wait_raii mechanism throws in such cases. We could also end up aborting
        // due to memory failures in future_has_exception().
        // NOTE: the idea here is that, after a wait() call, all the futures of successful tasks have been erased,
        // with at most 1 surviving future from the first throwing task. This way, wait() does some cleaning up
        // behind the scenes, without changing the behaviour of successive wait_check() and status() calls: wait_check()
        // will still re-throw the first exception, and status() will still return idle_error.
        auto iwr = detail::wait_raii<>::getter();
        (void)iwr;
        const auto it_f = m_ptr->futures.end();
        auto it_first_exc = it_f;
        for (auto it = m_ptr->futures.begin(); it != it_f; ++it) {
            // Wait on the task.
            detail::wait_f(*it);
            if (it_first_exc == it_f && detail::future_has_exception(*it)) {
                // Store an iterator to the throwing future.
                it_first_exc = it;
            }
        }
        if (it_first_exc == it_f) {
            // No exceptions were raised, just clear everything.
            m_ptr->futures.clear();
        } else {
            // We had a throwing future: preserve it and erase all the other futures.
            auto tmp_f(std::move(*it_first_exc));
            m_ptr->futures.clear();
            m_ptr->futures.emplace_back(std::move(tmp_f));
        }
    }
    /// Status of the island.
    /**
     * This method will return a pagmo::evolve_status flag indicating the current status of
     * asynchronous operations in the island. The flag will be:
     *
     * * evolve_status::idle if the island is currently not evolving and no exceptions
     *   were thrown by evolution tasks since the last call to wait_check();
     * * evolve_status::busy if the island is evolving and no exceptions
     *   have (yet) been thrown by evolution tasks since the last call to wait_check();
     * * evolve_status::idle_error if the island is currently not evolving and at least one
     *   evolution task threw an exception since the last call to wait_check();
     * * evolve_status::busy_error if the island is currently evolving and at least one
     *   evolution task has already thrown an exception since the last call to wait_check().
     *
     * Note that after a call to wait_check(), status() will always return evolve_status::idle,
     * and after a call to wait(), status() will always return either evolve_status::idle or
     * evolve_status::idle_error.
     *
     * @return a flag indicating the current status of asynchronous operations in the island.
     */
    evolve_status status() const
    {
        // Error flag. It will be set to true if at least one completed task raised an exception.
        bool error = false;
        // Iterate over all current evolve() tasks.
        for (auto &f : m_ptr->futures) {
            if (detail::future_running(f)) {
                // We have at least one busy task. The return status will be either "busy"
                // or "busy_error", depending on whether at least one completed task raised an
                // exception.
                if (error) {
                    return evolve_status::busy_error;
                }
                return evolve_status::busy;
            }
            // The current task is not running. Check if it generated an error.
            // NOTE: the '||' is because this flag, once set to true, needs to stay true.
            error = error || detail::future_has_exception(f);
        }
        if (error) {
            // All tasks have finished and at least one generated an error.
            return evolve_status::idle_error;
        }
        // All tasks have finished, no errors generated.
        return evolve_status::idle;
    }
    /// Get the algorithm.
    /**
     * It is safe to call this method while the island is evolving.
     *
     * @return a copy of the island's algorithm.
     *
     * @throws unspecified any exception thrown by threading primitives or by the invoked
     * copy constructor.
     */
    algorithm get_algorithm() const
    {
        // NOTE: we use this convoluted method involving shared pointers, instead of just
        // locking and returning a copy, to accommodate Python. Due to the way the GIL works,
        // we need to be very careful about not using the Python interpreter while holding a C++ lock:
        // since the interpreter may release the GIL at any time, we can easily run into deadlocks
        // due to lock order inversion. So, instead of locking in C++ and then potentially calling into
        // Python to perform the copy, we first get a shallow copy of the algo (which involves only C++
        // operations), release the C++ lock and then call into Python to perform the copy.
        // NOTE: we need to protect with a mutex here because m_ptr->algo might be set concurrently
        // by set_algorithm() below, and we guarantee strong thread safety for this method.
        // NOTE: it might be possible to replace the locks with atomic operations:
        // http://en.cppreference.com/w/cpp/memory/shared_ptr/atomic
        std::unique_lock<std::mutex> lock(m_ptr->algo_mutex);
        auto new_algo_ptr = m_ptr->algo;
        lock.unlock();
        assert(new_algo_ptr);
        return *new_algo_ptr;
    }
    /// Set the algorithm.
    /**
     * It is safe to call this method while the island is evolving.
     *
     * @param algo the algorithm that will be copied into the island.
     *
     * @throws unspecified any exception thrown by threading primitives, memory allocation erros
     * or the invoked copy constructor.
     */
    void set_algorithm(algorithm algo)
    {
        auto new_algo_ptr = std::make_shared<algorithm>(std::move(algo));
        // NOTE: here the lock's dtor will run **before** the dtor of algo.
        // This helps guaranteeing that no python operation is run while the lock
        // is held (it *should* not happen anyway as algo, after the move, is now
        // "empty" - it has no UDA - but this pattern adds another layer of safety).
        std::lock_guard<std::mutex> lock(m_ptr->algo_mutex);
        m_ptr->algo = new_algo_ptr;
    }
    /// Get the population.
    /**
     * It is safe to call this method while the island is evolving.
     *
     * @return a copy of the island's population.
     *
     * @throws unspecified any exception thrown by threading primitives or by the invoked
     * copy constructor.
     */
    population get_population() const
    {
        std::unique_lock<std::mutex> lock(m_ptr->pop_mutex);
        auto new_pop_ptr = m_ptr->pop;
        lock.unlock();
        assert(new_pop_ptr);
        return *new_pop_ptr;
    }
    /// Set the population.
    /**
     * It is safe to call this method while the island is evolving.
     *
     * @param pop the population that will be copied into the island.
     *
     * @throws unspecified any exception thrown by threading primitives, memory allocation errors
     * or by the invoked copy constructor.
     */
    void set_population(population pop)
    {
        auto new_pop_ptr = std::make_shared<population>(std::move(pop));
        std::lock_guard<std::mutex> lock(m_ptr->pop_mutex);
        m_ptr->pop = new_pop_ptr;
    }
    /// Get the thread safety of the island's members.
    /**
     * It is safe to call this method while the island is evolving.
     *
     * @return an array containing the pagmo::thread_safety values for the internal algorithm and population's
     * problem (as returned by pagmo::algorithm::get_thread_safety() and pagmo::problem::get_thread_safety()).
     *
     * @throws unspecified any exception thrown by threading primitives.
     */
    std::array<thread_safety, 2> get_thread_safety() const
    {
        std::array<thread_safety, 2> retval;
        {
            std::lock_guard<std::mutex> lock(m_ptr->algo_mutex);
            retval[0] = m_ptr->algo->get_thread_safety();
        }
        {
            std::lock_guard<std::mutex> lock(m_ptr->pop_mutex);
            retval[1] = m_ptr->pop->get_problem().get_thread_safety();
        }
        return retval;
    }
    /// Island's name.
    /**
     * If the UDI satisfies pagmo::has_name, then this method will return the output of its <tt>%get_name()</tt> method.
     * Otherwise, an implementation-defined name based on the type of the UDI will be returned.
     *
     * It is safe to call this method while the island is evolving.
     *
     * @return the name of the UDI.
     *
     * @throws unspecified any exception thrown by the <tt>%get_name()</tt> method of the UDI.
     */
    std::string get_name() const
    {
        return m_ptr->isl_ptr->get_name();
    }
    /// Island's extra info.
    /**
     * If the UDI satisfies pagmo::has_extra_info, then this method will return the output of its
     * <tt>%get_extra_info()</tt> method. Otherwise, an empty string will be returned.
     *
     * It is safe to call this method while the island is evolving.
     *
     * @return extra info about the UDI.
     *
     * @throws unspecified any exception thrown by the <tt>%get_extra_info()</tt> method of the UDI.
     */
    std::string get_extra_info() const
    {
        return m_ptr->isl_ptr->get_extra_info();
    }
    /// Stream operator for pagmo::island.
    /**
     * This operator will stream to \p os a human-readable representation of \p isl.
     *
     * It is safe to call this method while the island is evolving.
     *
     * @param os the target stream.
     * @param isl the island.
     *
     * @return a reference to \p os.
     *
     * @throws unspecified any exception thrown by:
     * - the stream operators of fundamental types, pagmo::algorithm and pagmo::population,
     * - pagmo::island::get_extra_info(), pagmo::island::get_algorithm(), pagmo::island::get_population().
     */
    friend std::ostream &operator<<(std::ostream &os, const island &isl)
    {
        stream(os, "Island name: ", isl.get_name());
        stream(os, "\n\tStatus: ", isl.status(), "\n\n");
        const auto extra_str = isl.get_extra_info();
        if (!extra_str.empty()) {
            stream(os, "Extra info:\n", extra_str, "\n\n");
        }
        stream(os, "Algorithm: " + isl.get_algorithm().get_name(), "\n\n");
        stream(os, "Problem: " + isl.get_population().get_problem().get_name(), "\n\n");
        stream(os, "Population size: ", isl.get_population().size(), "\n");
        stream(os, "\tChampion decision vector: ", isl.get_population().champion_x(), "\n");
        stream(os, "\tChampion fitness: ", isl.get_population().champion_f(), "\n");
        return os;
    }
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
    void save(Archive &ar) const
    {
        ar(m_ptr->isl_ptr, get_algorithm(), get_population());
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
    void load(Archive &ar)
    {
        // Deserialize into tmp island, and then move assign it.
        island tmp_island;
        // NOTE: no need to lock access to these, as there is no evolution going on in tmp_island.
        ar(tmp_island.m_ptr->isl_ptr);
        ar(*tmp_island.m_ptr->algo);
        ar(*tmp_island.m_ptr->pop);
        *this = std::move(tmp_island);
    }

private:
    std::unique_ptr<idata_t> m_ptr;
};

/// Run evolve.
/**
 * This method will use copies of <tt>isl</tt>'s
 * algorithm and population, obtained via island::get_algorithm() and island::get_population(),
 * to evolve the input island's population. The evolved population will be assigned to \p isl
 * using island::set_population().
 *
 * @param isl the pagmo::island that will undergo evolution.
 *
 * @throws std::invalid_argument if <tt>isl</tt>'s algorithm or problem do not provide
 * at least the pagmo::thread_safety::basic thread safety guarantee.
 * @throws unspecified any exception thrown by island::get_algorithm(), island::get_population(),
 * island::set_population().
 */
inline void thread_island::run_evolve(island &isl) const
{
    const auto i_ts = isl.get_thread_safety();
    if (static_cast<int>(i_ts[0]) < static_cast<int>(thread_safety::basic)) {
        pagmo_throw(std::invalid_argument,
                    "the 'thread_island' UDI requires an algorithm providing at least the 'basic' "
                    "thread safety guarantee");
    }
    if (static_cast<int>(i_ts[1]) < static_cast<int>(thread_safety::basic)) {
        pagmo_throw(std::invalid_argument, "the 'thread_island' UDI requires a problem providing at least the 'basic' "
                                           "thread safety guarantee");
    }
    isl.set_population(isl.get_algorithm().evolve(isl.get_population()));
}

/// Archipelago.
/**
 * \image html archi_no_text.png
 *
 * An archipelago is a collection of pagmo::island objects which provides a convenient way to perform
 * multiple optimisations in parallel.
 *
 * The interface of pagmo::archipelago mirrors partially the interface
 * of pagmo::island: the evolution is initiated by a call to evolve(), and at any time the user can query the
 * state of the archipelago and access its island members. The user can explicitly wait for pending evolutions
 * to conclude by calling the wait() and wait_check() methods. The status of
 * ongoing evolutions in the archipelago can be queried via status().
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from :cpp:class:`pagmo::archipelago` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * \endverbatim
 */
// TODO: topology:
// - copy constructor support,
// - documentation,
// - push back support,
// - additional ctors with topo args.
class archipelago
{
    using container_t = std::vector<std::unique_ptr<island>>;
    using size_type_implementation = container_t::size_type;
    using iterator_implementation = boost::indirect_iterator<container_t::iterator>;
    using const_iterator_implementation = boost::indirect_iterator<container_t::const_iterator>;

    // NOTE: same utility method as in pagmo::island, see there.
    void wait_check_ignore()
    {
        try {
            wait_check();
        } catch (...) {
        }
    }

public:
    /// The size type of the archipelago.
    /**
     * This is an unsigned integer type used to represent the number of islands in the
     * archipelago.
     */
    using size_type = size_type_implementation;
    /// Database of migrating individuals.
    /**
     * \rststar
     * A migrants database is a vector whose :math:`i`-th entry
     * contains the group of individuals (represented as a :cpp:type:`~pagmo::migrants_t`)
     * that have been selected for migration from the :math:`i`-th island of the archipelago.
     * \endrststar
     */
    using migrants_db_t = std::vector<migrants_t>;

private:
    using idx_map_t = std::unordered_map<const island *, size_type>;

public:
    /// Mutable iterator.
    /**
     * Dereferencing a mutable iterator will yield a reference to an island within the archipelago.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    Mutable iterators are provided solely in order to allow calling non-const methods
     *    on the islands. Assigning an island via a mutable iterator will result in undefined behaviour.
     *
     * \endverbatim
     */
    using iterator = iterator_implementation;
    /// Const iterator.
    /**
     * Dereferencing a const iterator will yield a const reference to an island within the archipelago.
     */
    using const_iterator = const_iterator_implementation;
    /// Default constructor.
    /**
     * The default constructor will initialise an empty archipelago with a pagmo::unconnected topology.
     */
    archipelago() = default;
    /// Copy constructor.
    /**
     * The copy constructor will perform a deep copy of \p other.
     *
     * @param other the archipelago that will be copied.
     *
     * @throws unspecified any exception thrown by:
     * - archipelago::push_back(),
     * - archipelago::get_topology(),
     * - archipelago::get_migrants_db().
     */
    archipelago(const archipelago &other)
    {
        for (const auto &iptr : other.m_islands) {
            // This will end up copying the island members,
            // assign the archi pointer, and associating ids to island pointers.
            // NOTE: on construction the topology of this will be unconnected,
            // it will be overridden below.
            push_back(*iptr);
        }
        // Set the topology.
        m_topo = other.get_topology();
        // Set the migrants.
        m_migrants = other.get_migrants_db();
    }
    /// Move constructor.
    /**
     * The move constructor will wait for any ongoing evolution in \p other to finish
     * and it will then transfer the state of \p other into \p this.
     *
     * @param other the archipelago that will be moved.
     */
    archipelago(archipelago &&other) noexcept
    {
        // NOTE: in move operations we have to wait, because the ongoing
        // island evolutions are interacting with their hosting archi 'other'.
        // We cannot just move in the vector of islands.
        other.wait_check_ignore();
        // Move in the islands, make sure other is cleared.
        m_islands = std::move(other.m_islands);
        other.m_islands.clear();
        // Re-direct the archi pointers to point to this.
        for (const auto &iptr : m_islands) {
            iptr->m_ptr->archi_ptr = this;
        }
        // Move over the indices, clear other.
        // NOTE: the indices are still valid as above we just moved in a vector
        // of unique_ptrs, without changing their content.
        m_idx_map = std::move(other.m_idx_map);
        other.m_idx_map.clear();
        // Move over the topology. No need to clear here as we know
        // in which state the topology will be in after the move.
        m_topo = std::move(other.m_topo);
        // Move over the migrants, clear other.
        m_migrants = std::move(other.m_migrants);
        other.m_migrants.clear();
    }

private:
#if defined(_MSC_VER)
    template <typename... Args>
    using n_ctor_enabler = int;
#else
    template <typename... Args>
    using n_ctor_enabler = enable_if_t<std::is_constructible<island, Args &&...>::value, int>;
#endif
    // The "default" constructor from n islands. Just forward
    // the input arguments to n calls to push_back().
    template <typename... Args>
    void n_ctor(size_type n, Args &&... args)
    {
        for (size_type i = 0; i < n; ++i) {
            // NOTE: don't forward, in order to avoid moving twice
            // from the same objects. This also ensures that, when
            // using a ctor without seed, the seed is set to random
            // for each island.
            push_back(args...);
        }
    }
    // These two implement the constructor which contains a seed argument.
    // We need to constrain with enable_if otherwise, due to the default
    // arguments in the island constructors, the wrong constructor would be called.
    template <typename Algo, typename Prob, typename S1, typename S2,
              enable_if_t<std::is_constructible<algorithm, Algo &&>::value, int> = 0>
    void n_ctor(size_type n, Algo &&a, Prob &&p, S1 size, S2 seed)
    {
        std::mt19937 eng(static_cast<std::mt19937::result_type>(static_cast<unsigned>(seed)));
        std::uniform_int_distribution<unsigned> udist;
        for (size_type i = 0; i < n; ++i) {
            push_back(a, p, size, udist(eng));
        }
    }
    template <typename Isl, typename Algo, typename Prob, typename S1, typename S2,
              enable_if_t<is_udi<uncvref_t<Isl>>::value, int> = 0>
    void n_ctor(size_type n, Isl &&isl, Algo &&a, Prob &&p, S1 size, S2 seed)
    {
        std::mt19937 eng(static_cast<std::mt19937::result_type>(static_cast<unsigned>(seed)));
        std::uniform_int_distribution<unsigned> udist;
        for (size_type i = 0; i < n; ++i) {
            push_back(isl, a, p, size, udist(eng));
        }
    }

public:
    /// Constructor from \p n islands.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if the parameter pack ``Args``
     *    can be used to construct a :cpp:class:`pagmo::island`.
     *
     * \endverbatim
     *
     * This constructor will first initialize an empty archipelago with pagmo::unconnected topology, and it will then
     * forward \p n times the input arguments \p args to the push_back() method, thus creating an archipelago
     * with \p n islands.
     *
     * If the parameter pack \p args contains an argument which
     * would be interpreted as a seed by the invoked island constructor, then this seed
     * will be used to initialise a random number generator that in turn will be used to generate
     * the seeds of populations of the islands that will be created within the archipelago. In other words,
     * passing a seed argument to this constructor will not generate \p n islands with the same
     * seed, but \p n islands whose population seeds have been randomly generated starting from
     * the supplied seed argument.
     *
     * @param n the desired number of islands.
     * @param args the arguments that will be used for the construction of each island.
     *
     * @throws unspecified any exception thrown by the invoked pagmo::island constructor
     * or by archipelago::push_back().
     */
    template <typename... Args, n_ctor_enabler<Args...> = 0>
    explicit archipelago(size_type n, Args &&... args)
    {
        n_ctor(n, std::forward<Args>(args)...);
    }
    /// Copy assignment.
    /**
     * Copy assignment is implemented as copy construction followed by a move assignment.
     *
     * @param other the assignment argument.
     *
     * @return a reference to \p this.
     *
     * @throws unspecified any exception thrown by the copy constructor.
     */
    archipelago &operator=(const archipelago &other)
    {
        if (this != &other) {
            *this = archipelago(other);
        }
        return *this;
    }
    /// Move assignment.
    /**
     * Move assignment will transfer the state of \p other into \p this, after any ongoing
     * evolution in \p this and \p other has finished.
     *
     * @param other the assignment argument.
     *
     * @return a reference to \p this.
     */
    archipelago &operator=(archipelago &&other) noexcept
    {
        if (this != &other) {
            // NOTE: as in the move ctor, we need to wait on other and this as well.
            // This mirrors the island's behaviour.
            wait_check_ignore();
            other.wait_check_ignore();
            // Move in the islands, clear other.
            m_islands = std::move(other.m_islands);
            other.m_islands.clear();
            // Re-direct the archi pointers to point to this.
            for (const auto &iptr : m_islands) {
                iptr->m_ptr->archi_ptr = this;
            }
            // Move the indices map, clear other.
            m_idx_map = std::move(other.m_idx_map);
            other.m_idx_map.clear();
            // Move over the topology.
            m_topo = std::move(other.m_topo);
            // Move over the migrants, clear other.
            m_migrants = std::move(other.m_migrants);
            other.m_migrants.clear();
        }
        return *this;
    }
    /// Destructor.
    /**
     * The destructor will call archipelago::wait_check() internally, ignoring any exception that might be thrown,
     * and run checks in debug mode.
     */
    ~archipelago()
    {
        // NOTE: this is not strictly necessary, but it will not hurt. And, if we add further
        // sanity checks, we know the archi is stopped.
        wait_check_ignore();
        // NOTE: we made sure in the move ctor/assignment that the island vector, the migrants and the indices
        // map are all cleared out after a move. Thus we can safely assert the following.
        assert(std::all_of(m_islands.begin(), m_islands.end(),
                           [this](const std::unique_ptr<island> &iptr) { return iptr->m_ptr->archi_ptr == this; }));
        assert(m_migrants.size() == m_islands.size());
        assert(m_idx_map.size() == m_islands.size());
#if !defined(NDEBUG)
        for (size_type i = 0; i < m_islands.size(); ++i) {
            assert(std::get<0>(m_migrants[i]).size() == std::get<1>(m_migrants[i]).size());
            assert(std::get<1>(m_migrants[i]).size() == std::get<2>(m_migrants[i]).size());
            assert(m_idx_map.find(m_islands[i].get()) != m_idx_map.end());
            assert(m_idx_map.find(m_islands[i].get())->second == i);
        }
#endif
    }
    /// Mutable island access.
    /**
     * This subscript operator can be used to access the <tt>i</tt>-th island of the archipelago (that is,
     * the <tt>i</tt>-th island that was inserted via push_back()). References returned by this method are valid even
     * after a push_back() invocation. Assignment and destruction of the archipelago will invalidate island references
     * obtained via this method.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    The mutable version of the subscript operator exists solely to allow calling non-const methods
     *    on the islands. Assigning an island via a reference obtained through this operator will result
     *    in undefined behaviour.
     *
     * \endverbatim
     *
     * @param i the index of the island to be accessed.
     *
     * @return a reference to the <tt>i</tt>-th island of the archipelago.
     *
     * @throws std::out_of_range if \p i is not less than the size of the archipelago.
     */
    island &operator[](size_type i)
    {
        if (i >= size()) {
            pagmo_throw(std::out_of_range,
                        "cannot access the island at index " + std::to_string(i)
                            + ": the archipelago has a size of only " + std::to_string(size()));
        }
        return *m_islands[i];
    }
    /// Const island access.
    /**
     * This subscript operator can be used to access the <tt>i</tt>-th island of the archipelago (that is,
     * the <tt>i</tt>-th island that was inserted via push_back()). References returned by this method are valid even
     * after a push_back() invocation. Assignment and destruction of the archipelago will invalidate island references
     * obtained via this method.
     *
     * @param i the index of the island to be accessed.
     *
     * @return a const reference to the <tt>i</tt>-th island of the archipelago.
     *
     * @throws std::out_of_range if \p i is not less than the size of the archipelago.
     */
    const island &operator[](size_type i) const
    {
        if (i >= size()) {
            pagmo_throw(std::out_of_range,
                        "cannot access the island at index " + std::to_string(i)
                            + ": the archipelago has a size of only " + std::to_string(size()));
        }
        return *m_islands[i];
    }
    /// Size.
    /**
     * @return the number of islands in the archipelago.
     */
    size_type size() const
    {
        return m_islands.size();
    }

private:
    template <typename... Args>
    using push_back_enabler = n_ctor_enabler<Args...>;

public:
    /// Add island.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This method is enabled only if the parameter pack ``Args``
     *    can be used to construct a :cpp:class:`pagmo::island`.
     *
     * \endverbatim
     *
     * This method will construct an island from the supplied arguments and add it to the archipelago.
     * Islands are added at the end of the archipelago (that is, the new island will have an index
     * equal to the value of size() before the call to this method). pagmo::topology::push_back()
     * will also be called on the pagmo::topology associated with this archipelago.
     *
     * @param args the arguments that will be used for the construction of the island.
     *
     * @throws std::overflow_error if the size of the archipelago is greater than an
     * implementation-defined maximum.
     * @throws unspecified any exception thrown by:
     * - memory allocation errors,
     * - threading primitives,
     * - pagmo::topology::push_back(),
     * - the invoked constructor of pagmo::island.
     */
    template <typename... Args, push_back_enabler<Args...> = 0>
    void push_back(Args &&... args)
    {
        // Construct the new island and assign the pointer into this.
        auto new_island = detail::make_unique<island>(std::forward<Args>(args)...);
        new_island->m_ptr->archi_ptr = this;

        // Try to make space for the new island in the islands vector
        // and in the migration db.
        m_islands.reserve(detail::safe_increment(m_islands.size()));
        {
            std::lock_guard<std::mutex> lock(m_migrants_mutex);
            m_migrants.reserve(detail::safe_increment(m_migrants.size()));
        }

        // Map the new island idx.
        {
            // NOTE: if anything fails here, we won't have modified the state of the archi
            // (apart from reserving memory).
            std::lock_guard<std::mutex> lock(m_idx_map_mutex);
            assert(m_idx_map.find(new_island.get()) == m_idx_map.end());
            m_idx_map.emplace(new_island.get(), m_islands.size());
        }

        // Add an empty entry to the migrants db.
        try {
            std::lock_guard<std::mutex> lock(m_migrants_mutex);
            m_migrants.emplace_back();
        } catch (...) {
            // LCOV_EXCL_START
            // NOTE: we get here only if the lock throws, because we made space for the
            // new migrants above already. Better to abort in such case, as we have no
            // reasonable path for recovering from this.
            std::cerr << "An unrecoverable error arose while adding an island to the archipelago, aborting now."
                      << std::endl;
            std::abort();
            // LCOV_EXCL_STOP
        }

        // Actually add the island. This cannot fail as we already reserved space.
        m_islands.emplace_back(std::move(new_island));

        // Finally, push back the topology. This is required to be thread safe, no need for locks.
        // If this fails, we will have a possibly *bad* topology in the archi, but this can
        // always happen via a bogus set_topology() and there's nothing we can do about it.
        m_topo.push_back();
    }
    /// Evolve archipelago.
    /**
     * This method will call island::evolve() on all the islands of the archipelago.
     * The input parameter \p n will be passed to the invocations of island::evolve() for each island.
     * archipelago::status() can be used to query the status of the asynchronous operations in the
     * archipelago.
     *
     * @param n the parameter that will be passed to island::evolve().
     *
     * @throws unspecified any exception thrown by island::evolve().
     */
    void evolve(unsigned n = 1)
    {
        for (auto &iptr : m_islands) {
            iptr->evolve(n);
        }
    }
    /// Block until all evolutions have finished.
    /**
     * This method will call island::wait() on all the islands of the archipelago. Exceptions thrown by island
     * evolutions can be re-raised via wait_check(): they will **not** be re-thrown by this method. Also, contrary to
     * wait_check(), this method will **not** reset the status of the archipelago: after a call to wait(), status() will
     * always return either evolve_status::idle or evolve_status::idle_error.
     */
    void wait() noexcept
    {
        for (const auto &iptr : m_islands) {
            iptr->wait();
        }
    }
    /// Block until all evolutions have finished and raise the first exception that was encountered.
    /**
     * This method will call island::wait_check() on all the islands of the archipelago (following
     * the order in which the islands were inserted into the archipelago).
     * The first exception raised by island::wait_check() will be re-raised by this method,
     * and all the exceptions thrown by the other calls to island::wait_check() will be ignored.
     *
     * Note that wait_check() resets the status of the archipelago: after a call to wait_check(), status() will always
     * return evolve_status::idle.
     *
     * @throws unspecified any exception thrown by any evolution task queued in the archipelago's
     * islands.
     */
    void wait_check()
    {
        for (auto it = m_islands.begin(); it != m_islands.end(); ++it) {
            try {
                (*it)->wait_check();
            } catch (...) {
                for (it = it + 1; it != m_islands.end(); ++it) {
                    (*it)->wait_check_ignore();
                }
                throw;
            }
        }
    }
    /// Status of the archipelago.
    /**
     * This method will return a pagmo::evolve_status flag indicating the current status of
     * asynchronous operations in the archipelago. The flag will be:
     *
     * * evolve_status::idle if, for all the islands in the archipelago, island::status() returns
     *   evolve_status::idle;
     * * evolve_status::busy if, for at least one island in the archipelago, island::status() returns
     *   evolve_status::busy, and for no island island::status() returns an error status;
     * * evolve_status::idle_error if no island in the archipelago is busy and for at least one island
     *   island::status() returns evolve_status::idle_error;
     * * evolve_status::busy_error if, for at least one island in the archipelago, island::status() returns
     *   an error status and at least one island is busy.
     *
     * Note that after a call to wait_check(), status() will always return evolve_status::idle,
     * and after a call to wait(), status() will always return either evolve_status::idle or
     * evolve_status::idle_error.
     *
     * @return a flag indicating the current status of asynchronous operations in the archipelago.
     */
    evolve_status status() const
    {
        decltype(m_islands.size()) n_idle = 0, n_busy = 0, n_idle_error = 0, n_busy_error = 0;
        for (const auto &iptr : m_islands) {
            switch (iptr->status()) {
                case evolve_status::idle:
                    ++n_idle;
                    break;
                case evolve_status::busy:
                    ++n_busy;
                    break;
                case evolve_status::idle_error:
                    ++n_idle_error;
                    break;
                case evolve_status::busy_error:
                    ++n_busy_error;
                    break;
            }
        }

        // If we have at last one busy error, the global state
        // is also busy error.
        if (n_busy_error) {
            return evolve_status::busy_error;
        }

        // The other error case: at least one island is idle with error.
        if (n_idle_error) {
            if (n_busy) {
                // At least one island is idle with error and at least one island is busy:
                // return busy error.
                return evolve_status::busy_error;
            }
            // No island is busy at all, at least one island is idle with error.
            return evolve_status::idle_error;
        }

        // No error in any island. If they are all idle, the global state is idle,
        // otherwise busy.
        return n_idle == m_islands.size() ? evolve_status::idle : evolve_status::busy;
    }
    /// Mutable begin iterator.
    /**
     * This method will return a mutable iterator pointing to the beginning of the internal island container. That is,
     * the returned iterator will either point to the first island of the archipelago (if size() is nonzero)
     * or it will be the same iterator returned by archipelago::end() (is size() is zero).
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    Mutable iterators are provided solely in order to allow calling non-const methods
     *    on the islands. Assigning an island via a mutable iterator will be undefined behaviour.
     *
     * \endverbatim
     *
     * @return a mutable iterator to the beginning of the island container.
     */
    iterator begin()
    {
        return iterator(m_islands.begin());
    }
    /// Mutable end iterator.
    /**
     * This method will return a mutable iterator pointing to the end of the internal island container.
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    Mutable iterators are provided solely in order to allow calling non-const methods
     *    on the islands. Assigning an island via a mutable iterator will be undefined behaviour.
     *
     * \endverbatim
     *
     * @return a mutable iterator to the end of the island container.
     */
    iterator end()
    {
        return iterator(m_islands.end());
    }
    /// Const begin iterator.
    /**
     * This method will return a const iterator pointing to the beginning of the internal island container. That is,
     * the returned iterator will either point to the first island of the archipelago (if size() is nonzero)
     * or it will be the same iterator returned by archipelago::end() const (is size() is zero).
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * @return a const iterator to the beginning of the island container.
     */
    const_iterator begin() const
    {
        return const_iterator(m_islands.begin());
    }
    /// Const end iterator.
    /**
     * This method will return a const iterator pointing to the end of the internal island container.
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * @return a const iterator to the end of the island container.
     */
    const_iterator end() const
    {
        return const_iterator(m_islands.end());
    }
    /// Stream operator.
    /**
     * This operator will stream to \p os a human-readable representation of the input
     * archipelago \p archi.
     *
     * @param os the target stream.
     * @param archi the archipelago that will be streamed.
     *
     * @return a reference to \p os.
     *
     * @throws unspecified any exception thrown by:
     * - the streaming of primitive types,
     * - island::get_algorithm(), island::get_population().
     */
    friend std::ostream &operator<<(std::ostream &os, const archipelago &archi)
    {
        stream(os, "Number of islands: ", archi.size(), "\n");
        // NOTE: this is thread-safe (topology stores a copy of the name upon construction).
        stream(os, "Topology: ", archi.m_topo.get_name(), "\n");
        stream(os, "Status: ", archi.status(), "\n\n");
        stream(os, "Islands summaries:\n\n");
        detail::table t({"#", "Type", "Algo", "Prob", "Size", "Status"}, "\t");
        for (decltype(archi.size()) i = 0; i < archi.size(); ++i) {
            const auto pop = archi[i].get_population();
            t.add_row(i, archi[i].get_name(), archi[i].get_algorithm().get_name(), pop.get_problem().get_name(),
                      pop.size(), archi[i].status());
        }
        stream(os, t);
        return os;
    }
    /// Get the fitness vectors of the islands' champions.
    /**
     * @return a collection of the fitness vectors of the islands' champions.
     *
     * @throws unspecified any exception thrown by population::champion_f() or
     * by memory errors in standard containers.
     */
    std::vector<vector_double> get_champions_f() const
    {
        std::vector<vector_double> retval;
        for (const auto &isl_ptr : m_islands) {
            retval.emplace_back(isl_ptr->get_population().champion_f());
        }
        return retval;
    }
    /// Get the decision vectors of the islands' champions.
    /**
     * @return a collection of the decision vectors of the islands' champions.
     *
     * @throws unspecified any exception thrown by population::champion_x() or
     * by memory errors in standard containers.
     */
    std::vector<vector_double> get_champions_x() const
    {
        std::vector<vector_double> retval;
        for (const auto &isl_ptr : m_islands) {
            retval.emplace_back(isl_ptr->get_population().champion_x());
        }
        return retval;
    }
    /// Get a copy of the topology.
    /**
     * @return a copy of the pagmo::topology associated to this archipelago.
     */
    topology get_topology() const
    {
        // NOTE: the copy ctor of topology is required to be thread-safe.
        return m_topo;
    }
    /// Set the topology.
    /**
     * This method will set the archipelago's topology to \p topo. The method will
     * wait for any ongoing evolution to stop before setting the topology.
     *
     * @param topo the pagmo::topology that will be set for this archipelago.
     */
    void set_topology(topology topo)
    {
        // NOTE: make sure we finish any ongoing evolution before setting the topology.
        // The assignment will trigger the destructor of the UDT, so we need to make
        // sure there's no interaction with the UDT happening.
        wait_check_ignore();
        m_topo = std::move(topo);
    }
    /// Get the migrants database.
    /**
     * This method will return the database of migrating individuals. See the description
     * of archipelago::migrants_db_t for information on the content of the returned object.
     *
     * Each element of the migrants database is guaranteed to contain a triple of vectors
     * of equal sizes.
     *
     * @return a copy of the database of migrating individuals.
     *
     * @throws unspecified any exception thrown by failures in threading primitives.
     */
    migrants_db_t get_migrants_db() const
    {
        std::lock_guard<std::mutex> lock(m_migrants_mutex);
        return m_migrants;
    }
    /// Get the migrants from a specific island.
    /**
     * This method will return the group of individuals that are currently in the migration
     * queue for the \f$ i \f$-th island.
     *
     * The returned triple is guaranteed to contain vectors of equal size.
     *
     * @param i the index of the island whose migrants will be returned.
     *
     * @return a copy of the migrants from the \f$ i \f$-th island of the archipelago.
     *
     * @throws std::out_of_range if \p i is not less than the size of the migration database.
     * @throws unspecified any exception thrown by failures in threading primitives.
     */
    migrants_t get_migrants(size_type i) const
    {
        std::lock_guard<std::mutex> lock(m_migrants_mutex);
        if (i >= m_migrants.size()) {
            pagmo_throw(std::out_of_range,
                        "cannot access the migrants of the island at index " + std::to_string(i)
                            + ": the migrants database has a size of only " + std::to_string(m_migrants.size()));
        }
        return m_migrants[i];
    }
    /// Get the connections to an island.
    /**
     * This method will return a pair containing the indices of the islands connecting into the
     * \f$ i \f$-th island, together with the weights of the connections. Internally, it will use
     * topology::get_connections(), and thus it is guaranteed that the returned vectors are of
     * equal size.
     *
     * @param i the index of the island whose connections will be returned.
     *
     * @return a pair representing the connections to the \f$ i \f$-th island and their weights.
     *
     * @throws unspecified any exception thrown by:
     * - topology::get_connections(),
     * - memory errors in standard containers,
     * - ``boost::numeric_cast()``.
     */
    std::pair<std::vector<size_type>, vector_double> get_island_connections(size_type i) const
    {
        // NOTE: we need to go through this conversion as technically we cannot be certain
        // that size_t and size_type are the same type.
        // NOTE: get_connections() is required to be thread-safe.
        auto tmp = m_topo.get_connections(boost::numeric_cast<std::size_t>(i));
        std::pair<std::vector<size_type>, vector_double> retval;
        retval.first.reserve(static_cast<decltype(retval.first.size())>(tmp.first.size()));
        std::transform(tmp.first.begin(), tmp.first.end(), std::back_inserter(retval.first),
                       [](const std::size_t &n) { return boost::numeric_cast<size_type>(n); });
        retval.second = std::move(tmp.second);
        return retval;
    }
    /// Get the index of an island.
    /**
     * This method will return the index of the island \p isl in the archipelago. If \p isl does
     * not belong to the archipelago, an error will be reaised.
     *
     * @param isl the island whose index will be returned.
     *
     * @return the index of \p isl in the archipelago.
     *
     * @throws std::invalid_argument if \p isl does not belong to the archipelago.
     * @throws unspecified any exception thrown by failures in threading primitives.
     */
    size_type get_island_idx(const island &isl) const
    {
        std::lock_guard<std::mutex> lock(m_idx_map_mutex);
        const auto ret = m_idx_map.find(&isl);
        if (ret == m_idx_map.end()) {
            pagmo_throw(
                std::invalid_argument,
                "the index of an island in an archipelago was requested, but the island is not in the archipelago");
        }
        return ret->second;
    }
    /// Save to archive.
    /**
     * This method will save to \p ar the islands of the archipelago.
     *
     * @param ar the output archive.
     *
     * @throws unspecified any exception thrown by the serialization of pagmo::island, pagmo::topology
     * or primitive types.
     */
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(m_islands, get_topology(), get_migrants_db());
    }
    /// Load from archive.
    /**
     * This method will load into \p this the content of \p ar, after any ongoing evolution
     * in \p this has finished.
     *
     * @param ar the input archive.
     *
     * @throws unspecified any exception thrown by the deserialization of pagmo::island, pagmo::topology
     * or primitive types, or by memory errors in standard containers.
     */
    template <typename Archive>
    void load(Archive &ar)
    {
        // NOTE: the idea here is that we will be loading the member of archi one by one in
        // separate variables, move assign the loaded data into a tmp archi and finally move-assign
        // the tmp archi into this. This allows the method to be exception safe, and to have
        // archi objects always in a consistent state at every stage of the deserialization.

        // The tmp archi. This is def-cted and idle, we will be able to move-in data without
        // worrying about synchronization.
        archipelago tmp;

        // The islands.
        container_t tmp_islands;
        ar(tmp_islands);

        // Map the islands to indices.
        idx_map_t tmp_idx_map;
        for (size_type i = 0; i < tmp_islands.size(); ++i) {
            tmp_idx_map.emplace(tmp_islands[i].get(), i);
        }

        // The topology.
        topology tmp_topo;
        ar(tmp_topo);

        // The migrants.
        migrants_db_t tmp_migrants;
        ar(tmp_migrants);

        // From now on, everything is noexcept.
        tmp.m_islands = std::move(tmp_islands);
        tmp.m_idx_map = std::move(tmp_idx_map);
        tmp.m_topo = std::move(tmp_topo);
        tmp.m_migrants = std::move(tmp_migrants);

        // NOTE: this final assignment will take care of setting the islands' archi pointers
        // appropriately via archi's move assignment operator.
        *this = std::move(tmp);
    }

private:
    // The islands.
    container_t m_islands;
    // The map from island pointers to indices in the archi.
    mutable std::mutex m_idx_map_mutex;
    idx_map_t m_idx_map;
    // The topology.
    topology m_topo = topology{unconnected{}};
    // The migrants.
    mutable std::mutex m_migrants_mutex;
    migrants_db_t m_migrants;
};

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
inline void island::evolve(unsigned n)
{
    // First add an empty future, so that if an exception is thrown
    // we will not have modified m_futures, nor we will have a future
    // in flight which we cannot wait upon.
    m_ptr->futures.emplace_back();
    try {
        // Move assign a new future provided by the enqueue() method.
        // NOTE: enqueue either returns a valid future, or throws without
        // having enqueued any task.
        m_ptr->futures.back() = m_ptr->queue.enqueue([this, n]() {
            // Init a random engine, it will be used to randomly pick
            // migrants from island connecting to this island in an archi.
            std::mt19937 eng(static_cast<std::mt19937::result_type>(pagmo::random_device::next()));
            // Migration probability distribution.
            std::uniform_real_distribution<double> pdist(0., 1.);
            // Cache the archi pointer.
            const auto aptr = this->m_ptr->archi_ptr;
            // Figure out what is the island's index in the archi, if we are
            // in an archi. Otherwise, this variable will be unused.
            const auto isl_idx = aptr ? aptr->get_island_idx(*this) : 0u;
            for (auto i = 0u; i < n; ++i) {
                // A vector of migrants. It will remain empty if the island
                // is not in an archi or if there's no candidate migrants.
                migrants_t migrants;
                if (aptr) {
                    // Pre-evolution hook for an island belonging to an archi.
                    // Get all the connections into the islands. This will return
                    // a pair with the incoming connections and their weights (i.e.,
                    // the migration probabilities).
                    const auto connections = aptr->get_island_connections(isl_idx);
                    assert(connections.first.size() == connections.second.size());
                    if (connections.first.size()) {
                        // Do something only if we actually have incoming connections.
                        // Pick a random island index among the island connecting to this.
                        std::uniform_int_distribution<decltype(connections.first.size())> idist(
                            0u, connections.first.size() - 1u);
                        const auto idx = idist(eng);
                        if (pdist(eng) < connections.second[static_cast<decltype(connections.second.size())>(idx)]) {
                        }
                    }
                }
                this->m_ptr->isl_ptr->run_evolve(*this);
            }
        });
        // LCOV_EXCL_START
    } catch (...) {
        // We end up here only if enqueue threw. In such a case, we need to cleanup
        // the empty future we added above before re-throwing and exiting.
        m_ptr->futures.pop_back();
        throw;
        // LCOV_EXCL_STOP
    }
}
}

PAGMO_REGISTER_ISLAND(pagmo::thread_island)

#endif
