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

#include <algorithm>
#include <array>
#include <boost/any.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/config.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/task_queue.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>

#if defined(PAGMO_WITH_FORK_ISLAND)

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ios>
#include <ostream>
#include <sstream>
#include <tuple>

#include <sys/types.h>
#include <unistd.h>

#endif

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
    virtual ~isl_inner_base() {}
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
    explicit isl_inner(const T &x) : m_value(x) {}
    explicit isl_inner(T &&x) : m_value(std::move(x)) {}
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
} // namespace detail

/// Thread island.
/**
 * This class is a user-defined island (UDI) that will run evolutions directly inside
 * the separate thread of execution within pagmo::island.
 *
 * thread_island is the UDI type automatically selected by the constructors of pagmo::island
 * on non-POSIX platforms or when both the island's problem and algorithm provide at least the
 * pagmo::thread_safety::basic thread safety guarantee.
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

#if defined(PAGMO_WITH_FORK_ISLAND)

// Fork island: will offload the evolution to a child process created with the fork() system call.
class fork_island
{
    // Small RAII wrapper around a pipe.
    struct pipe_t {
        // Def ctor: will create the pipe.
        pipe_t() : r_status(true), w_status(true)
        {
            int fd[2];
            // LCOV_EXCL_START
            if (pipe(fd) == -1) {
                pagmo_throw(std::runtime_error, "Unable to create a pipe with the pipe() function. The error code is "
                                                    + std::to_string(errno) + " and the error message is: '"
                                                    + std::strerror(errno) + "'");
            }
            // LCOV_EXCL_STOP
            // The pipe was successfully opened, copy over
            // the r/w descriptors.
            rd = fd[0];
            wd = fd[1];
        }
        // Try to close the reading end if it has not been closed already.
        void close_r()
        {
            if (r_status) {
                // LCOV_EXCL_START
                if (close(rd) == -1) {
                    pagmo_throw(
                        std::runtime_error,
                        "Unable to close the reading end of a pipe with the close() function. The error code is "
                            + std::to_string(errno) + " and the error message is: '" + std::strerror(errno) + "'");
                }
                // LCOV_EXCL_STOP
                r_status = false;
            }
        }
        // Try to close the writing end if it has not been closed already.
        void close_w()
        {
            if (w_status) {
                // LCOV_EXCL_START
                if (close(wd) == -1) {
                    pagmo_throw(
                        std::runtime_error,
                        "Unable to close the writing end of a pipe with the close() function. The error code is "
                            + std::to_string(errno) + " and the error message is: '" + std::strerror(errno) + "'");
                }
                // LCOV_EXCL_STOP
                w_status = false;
            }
        }
        ~pipe_t()
        {
            // Attempt to close the pipe on destruction.
            try {
                close_r();
                close_w();
                // LCOV_EXCL_START
            } catch (const std::runtime_error &re) {
                // We are in a dtor, the error is not recoverable.
                std::cerr << "An unrecoverable error was raised while trying to close a pipe in the pipe's destructor. "
                             "The full error message is:\n"
                          << re.what() << "\n\nExiting now." << std::endl;
                std::exit(1);
            }
            // LCOV_EXCL_STOP
        }
        // Wrapper around the read() function.
        ssize_t read(void *buf, std::size_t count) const
        {
            auto retval = ::read(rd, buf, count);
            // LCOV_EXCL_START
            if (retval == -1) {
                pagmo_throw(std::runtime_error,
                            "Unable to read from a pipe with the read() function. The error code is "
                                + std::to_string(errno) + " and the error message is: '" + std::strerror(errno) + "'");
            }
            // LCOV_EXCL_STOP
            return retval;
        }
        // Wrapper around the write() function.
        ssize_t write(const void *buf, std::size_t count) const
        {
            auto retval = ::write(wd, buf, count);
            // LCOV_EXCL_START
            if (retval == -1) {
                pagmo_throw(std::runtime_error,
                            "Unable to write to a pipe with the write() function. The error code is "
                                + std::to_string(errno) + " and the error message is: '" + std::strerror(errno) + "'");
            }
            // LCOV_EXCL_STOP
            return retval;
        }
        // The file descriptors of the two ends of the pipe.
        int rd, wd;
        // Flag to signal the status of the two ends
        // of the pipe: true for open, false for closed.
        bool r_status, w_status;
    };
    // The structure we use to pass messages from the child to the parent:
    // - int, status flag,
    // - string, error message,
    // - the algorithm used for evolution,
    // - the evolved population.
    using message_t = std::tuple<int, std::string, algorithm, population>;

public:
    // NOTE: we need to implement these because of the m_pid member,
    // which has a trivial def ctor and which is missing the copy/move ctors.
    // m_pid is only informational and it is relevant only while the evolution
    // is undergoing, we will not copy it or serialize it.
    fork_island() : m_pid(0) {}
    fork_island(const fork_island &) : fork_island() {}
    fork_island(fork_island &&) : fork_island() {}
    void run_evolve(island &) const;
    std::string get_name() const
    {
        return "Fork island";
    }
    // Extra info: report the child process' ID, if evolution
    // is active.
    std::string get_extra_info() const
    {
        const auto pid = m_pid.load();
        if (pid) {
            return "\tChild PID: " + std::to_string(pid);
        }
        return "\tNo active child.";
    }
    // Get the PID of the child.
    pid_t get_child_pid() const
    {
        return m_pid.load();
    }
    template <typename Archive>
    void serialize(Archive &)
    {
    }

private:
    mutable std::atomic<pid_t> m_pid;
};

#endif

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

// This is the default UDI type selector. It will select the thread_island if both algorithm
// and population provide at least the basic thread safety guarantee. Otherwise, it will select
// the fork_island, if available.
inline void default_island_factory(const algorithm &algo, const population &pop,
                                   std::unique_ptr<detail::isl_inner_base> &ptr)
{
    (void)algo;
    (void)pop;
#if defined(PAGMO_WITH_FORK_ISLAND)
    if (static_cast<int>(algo.get_thread_safety()) < static_cast<int>(thread_safety::basic)
        || static_cast<int>(pop.get_problem().get_thread_safety()) < static_cast<int>(thread_safety::basic)) {
        ptr = make_unique<isl_inner<fork_island>>();
        return;
    }
#endif
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

template <typename = void>
struct island_static_data {
    // A map to link a human-readable description to evolve_status.
    // NOTE: in C++11 hashing of enums might not be available. Provide our own.
    struct status_hasher {
        std::size_t operator()(evolve_status es) const noexcept
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
} // namespace detail

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
    island() : m_ptr(detail::make_unique<idata_t>()) {}
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
    using algo_pop_enabler = enable_if_t<
        std::is_constructible<algorithm, Algo &&>::value && std::is_same<population, uncvref_t<Pop>>::value, int>;

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
     *    :cpp:class:`pagmo::algorithm`, and ``p``, ``size`` and ``seed`` can be used to construct a
     *    :cpp:class:`pagmo::population`.
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
    const T *extract() const
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
    T *extract()
    {
        auto isl = dynamic_cast<detail::isl_inner<T> *>(m_ptr->isl_ptr.get());
        return isl == nullptr ? nullptr : &(isl->m_value);
    }
    /// Check if the UDI used for construction is of type \p T.
    /**
     * @return \p true if the UDI used in construction is of type \p T, \p false otherwise.
     */
    template <typename T>
    bool is() const
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
    void evolve(unsigned n = 1)
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
                for (auto i = 0u; i < n; ++i) {
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
 * using island::set_population(), and the algorithm used for the evolution will be assigned
 * to \p isl using island::set_algorithm().
 *
 * @param isl the pagmo::island that will undergo evolution.
 *
 * @throws std::invalid_argument if <tt>isl</tt>'s algorithm or problem do not provide
 * at least the pagmo::thread_safety::basic thread safety guarantee.
 * @throws unspecified any exception thrown by:
 * - island::get_algorithm(), island::get_population(),
 * - island::set_algorithm(), island::set_population(),
 * - algorithm::evolve().
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
    // Get out a copy of the algorithm for evolution.
    auto algo = isl.get_algorithm();
    // Replace the island's population with the evolved population.
    isl.set_population(algo.evolve(isl.get_population()));
    // Replace the island's algorithm with the algorithm used for the evolution.
    // NOTE: if set_algorithm() fails, we will have the new population with the
    // original algorithm, which is still a valid state for the island.
    isl.set_algorithm(std::move(algo));
}

#if defined(PAGMO_WITH_FORK_ISLAND)

inline void fork_island::run_evolve(island &isl) const
{
    // A message that will be used both by parent and child.
    message_t m;
    // The pipe.
    pipe_t p;
    // Try to fork now.
    auto child_pid = fork();
    // LCOV_EXCL_START
    if (child_pid == -1) {
        // Forking failed.
        pagmo_throw(std::runtime_error,
                    "Cannot fork the process in a fork_island with the fork() function. The error code is "
                        + std::to_string(errno) + " and the error message is: '" + std::strerror(errno) + "'");
    }
    // LCOV_EXCL_STOP
    if (child_pid) {
        // We are in the parent.
        // Small raii helper to ensure that the pid of the child is atomically
        // set on construction, and reset to zero by the dtor.
        struct pid_setter {
            explicit pid_setter(std::atomic<pid_t> &ap, pid_t pid) : m_ap(ap)
            {
                m_ap.store(pid);
            }
            ~pid_setter()
            {
                m_ap.store(0);
            }
            std::atomic<pid_t> &m_ap;
        };
        pid_setter ps(m_pid, child_pid);
        try {
            // Close the write descriptor, we don't need to send anything to the child.
            p.close_w();
            // Prepare a local buffer and a stringstream, then read the data from the child.
            // NOTE: make the buffer small enough that its size can be represented by any
            // integral type.
            char buffer[100];
            std::stringstream ss;
            {
                cereal::BinaryInputArchive iarchive(ss);
                while (true) {
                    const auto read_bytes = p.read(static_cast<void *>(buffer), sizeof(buffer));
                    if (!read_bytes) {
                        // EOF, break out.
                        break;
                    }
                    ss.write(buffer, static_cast<std::streamsize>(read_bytes));
                }
                iarchive(m);
            }
            // Close the read descriptor.
            p.close_r();
        } catch (...) {
            // Something failed above. As a cleanup action, try to kill the child
            // before re-raising the error.
            if (kill(child_pid, SIGTERM) == -1 && errno != ESRCH) {
                // LCOV_EXCL_START
                // The signal delivery to the child failed, and not because
                // the child does not exist any more (if the child did not exist,
                // errno would be ESRCH).
                std::cerr << "An unrecoverable error was raised while handling another error in the parent process "
                             "of a fork_island. Giving up now."
                          << std::endl;
                std::exit(1);
                // LCOV_EXCL_STOP
            }
            // Re-raise.
            throw;
        }
        // At this point, we have received the data from the child, and we can insert
        // it into isl, or raise an error.
        if (std::get<0>(m)) {
            pagmo_throw(std::runtime_error, "The run_evolve() method of fork_island raised an error in the "
                                            "child process. The full error message reported by the child is:\n"
                                                + std::get<1>(m));
        }
        isl.set_algorithm(std::move(std::get<2>(m)));
        isl.set_population(std::move(std::get<3>(m)));
    } else {
        // NOTE: we won't get any coverage data from the child process, so just disable
        // lcov for this whole block.
        //
        // LCOV_EXCL_START
        //
        // We are in the child.
        //
        // Small helpers to serialize a message and send the contents of a string
        // stream back to the parent. This is split in 2 separate functions
        // because we can handle errors in serialize_message(), but not in send_ss().
        auto serialize_message = [](std::stringstream &ss, const message_t &ms) {
            cereal::BinaryOutputArchive oarchive(ss);
            oarchive(ms);
        };
        auto send_ss = [&p](std::stringstream &ss) {
            // NOTE: make the buffer small enough that its size can be represented by any
            // integral type.
            char buffer[100];
            std::size_t read_bytes;
            while (!ss.eof()) {
                // Copy a chunk of data from the stream to the local buffer.
                ss.read(buffer, static_cast<std::streamsize>(sizeof(buffer)));
                // Figure out how much we actually read.
                read_bytes = static_cast<std::size_t>(ss.gcount());
                assert(read_bytes <= sizeof(buffer));
                // Now let's send the current content of the buffer to the parent.
                p.write(static_cast<const void *>(buffer), read_bytes);
            }
        };
        // Fatal error message.
        constexpr char fatal_msg[]
            = "An unrecoverable error was raised while handling another error in the child process "
              "of a fork_island. Giving up now.";
        try {
            // Close the read descriptor, we don't need to read anything from the parent.
            p.close_r();
            // Run the evolution.
            auto algo = isl.get_algorithm();
            auto new_pop = algo.evolve(isl.get_population());
            // Pack in m and serialize the result of the evolution.
            // NOTE: m was def cted, which, for tuples, value-inits all members.
            // So the status flag is already zero and the error message empty.
            std::get<2>(m) = std::move(algo);
            std::get<3>(m) = std::move(new_pop);
            // Serialize the message into a stringstream.
            std::stringstream ss;
            serialize_message(ss, m);
            // NOTE: any error raised past this point may now result in incomplete/corrupted
            // data being sent back to the parent. We have no way of recovering from that,
            // so we will just bail out.
            try {
                // Send the evolved population/algorithm back to the parent.
                send_ss(ss);
                // Close the write descriptor.
                p.close_w();
                // All done, we can kill the child.
                std::exit(0);
            } catch (...) {
                std::cerr << "An unrecoverable error was raised while trying to send data back to the parent process "
                             "from the child process of a fork_island. Giving up now."
                          << std::endl;
                std::exit(1);
            }
        } catch (const std::exception &e) {
            // If we caught an std::exception try to set the error message in m before continuing.
            // We will try to send the error message back to the parent.
            try {
                std::get<1>(m) = e.what();
            } catch (...) {
                std::cerr << fatal_msg << std::endl;
                std::exit(1);
            }
        } catch (...) {
            // Not an std::exception, we won't have an error message.
        }
        // If we get here, it means that something went wrong above. We will try
        // to send an error message back to the parent. Failing that, we will bail.
        try {
            // Set the error flag.
            std::get<0>(m) = 1;
            // Make sure the algo/pop in m are set to serializable entities.
            std::get<2>(m) = algorithm{};
            std::get<3>(m) = population{};
            // Send the message.
            std::stringstream ss;
            serialize_message(ss, m);
            send_ss(ss);
            // Close the write descriptor.
            p.close_w();
            // All done, we can kill the child.
            std::exit(0);
        } catch (...) {
            std::cerr << fatal_msg << std::endl;
            std::exit(1);
        }
        // LCOV_EXCL_STOP
    }
}

#endif

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
 */
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
    /// Mutable iterator.
    /**
     * Dereferencing a mutable iterator will yield a reference to an island within the archipelago.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    Mutable iterators are provided solely in order to allow calling non-const methods
     *    on the islands. Assigning an island via a mutable iterator will be undefined behaviour.
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
     * The default constructor will initialise an empty archipelago.
     */
    archipelago() {}
    /// Copy constructor.
    /**
     * The islands of \p other will be copied into \p this via archipelago::push_back().
     *
     * @param other the archipelago that will be copied.
     *
     * @throws unspecified any exception thrown by archipelago::push_back().
     */
    archipelago(const archipelago &other)
    {
        for (const auto &iptr : other.m_islands) {
            // This will end up copying the island members,
            // and assign the archi pointer as well.
            push_back(*iptr);
        }
    }
    /// Move constructor.
    /**
     * The move constructor will wait for any ongoing evolution in \p other to finish
     * and it will then transfer the state of \p other into \p this. After the move,
     * \p other is left in an unspecified but valid state.
     *
     * @param other the archipelago that will be moved.
     */
    archipelago(archipelago &&other) noexcept
    {
        // NOTE: in move operations we have to wait, because the ongoing
        // island evolutions are interacting with their hosting archi 'other'.
        // We cannot just move in the vector of islands.
        other.wait_check_ignore();
        // Move in the islands.
        m_islands = std::move(other.m_islands);
        // Re-direct the archi pointers to point to this.
        for (const auto &iptr : m_islands) {
            iptr->m_ptr->archi_ptr = this;
        }
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
     * This constructor will forward \p n times the input arguments \p args to the
     * push_back() method. If, however, the parameter pack contains an argument which
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
            // Move in the islands.
            m_islands = std::move(other.m_islands);
            // Re-direct the archi pointers to point to this.
            for (const auto &iptr : m_islands) {
                iptr->m_ptr->archi_ptr = this;
            }
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
        assert(std::all_of(m_islands.begin(), m_islands.end(),
                           [this](const std::unique_ptr<island> &iptr) { return iptr->m_ptr->archi_ptr == this; }));
    }
    /// Mutable island access.
    /**
     * This subscript operator can be used to access the <tt>i</tt>-th island of the archipelago (that is,
     * the <tt>i</tt>-th island that was inserted via push_back()). References returned by this method are valid even
     * after a push_back() invocation. Assignment and destruction of the archipelago will invalidate island references
     * obtained via this method.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. note::
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
            pagmo_throw(std::out_of_range, "cannot access the island at index " + std::to_string(i)
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
            pagmo_throw(std::out_of_range, "cannot access the island at index " + std::to_string(i)
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
     * equal to the value of size() before the call to this method).
     *
     * @param args the arguments that will be used for the construction of the island.
     *
     * @throws unspecified any exception thrown by memory allocation errors or by the invoked constructor
     * of pagmo::island.
     */
    template <typename... Args, push_back_enabler<Args...> = 0>
    void push_back(Args &&... args)
    {
        m_islands.emplace_back(detail::make_unique<island>(std::forward<Args>(args)...));
        // NOTE: this is noexcept.
        m_islands.back()->m_ptr->archi_ptr = this;
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

        // The other error case.
        if (n_idle_error) {
            if (n_busy) {
                // At least one island is idle with error. If any other
                // island is busy, we return busy error.
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
     * .. note::
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
     * .. note::
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
    /// Save to archive.
    /**
     * This method will save to \p ar the islands of the archipelago.
     *
     * @param ar the output archive.
     *
     * @throws unspecified any exception thrown by the serialization of pagmo::island.
     */
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(m_islands);
    }
    /// Load from archive.
    /**
     * This method will load into \p this the content of \p ar, after any ongoing evolution
     * in \p this has finished.
     *
     * @param ar the input archive.
     *
     * @throws unspecified any exception thrown by the deserialization of pagmo::island.
     */
    template <typename Archive>
    void load(Archive &ar)
    {
        archipelago tmp;
        ar(tmp.m_islands);
        *this = std::move(tmp);
    }

private:
    container_t m_islands;
};
} // namespace pagmo

PAGMO_REGISTER_ISLAND(pagmo::thread_island)

#if defined(PAGMO_WITH_FORK_ISLAND)

PAGMO_REGISTER_ISLAND(pagmo::fork_island)

#endif

#endif
