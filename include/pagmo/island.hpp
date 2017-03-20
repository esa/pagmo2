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
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <typeinfo>
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

/// Detect \p run_evolve() method.
/**
 * This type trait will be \p true if \p T provides a method with
 * the following signature:
 * @code{.unparsed}
 * void run_evolve(std::island &) const;
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
    // The run_evolve() method.
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
// of island::wait(). Normally this object's constructor and destructor will not
// do anything, but in Python we need to override this getter so that it returns
// a RAII object that unlocks the GIL, otherwise we could run into deadlocks in Python
// if isl::wait() holds the GIL while waiting.
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
    // Algo and pop a mutex to regulate access.
    std::mutex algo_mutex;
    std::shared_ptr<algorithm> algo;
    std::mutex pop_mutex;
    std::shared_ptr<population> pop;
    std::vector<std::future<void>> futures;
    archipelago *archi_ptr = nullptr;
    task_queue queue;
};
}

/// Island class.
/**
 * \image html island.jpg
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
 * to the island::evolve() method. At any time the user can query the state of the island
 * and fetch its internal data members. The user can explicitly wait for pending evolutions
 * to conclude by calling the island::wait() method.
 *
 * Typically, pagmo users will employ an already-available UDI (such as pagmo::thread_island) in
 * conjunction with this class, but advanced users can implement their own UDI types. A user-defined
 * island must implement the following method:
 * @code{.unparsed}
 * void run_evolve(pagmo::algorithm &, std::mutex &, pagmo::population &, std::mutex &);
 * @endcode
 *
 * The <tt>run_evolve()</tt> method of
 * the UDI will use the input algorithm's algorithm::evolve() method to evolve the input population
 * and, once the evolution is finished, will replace the input population with the evolved population.
 * The two extra arguments are (unlocked) mutexes that regulate exclusive access to the input algorithm and population
 * respectively. Typically, a UDI's <tt>run_evolve()</tt> method will lock the mutexes, copy the input algorithm and
 * population, release the locks, evolve the copy of the population, re-acquire the population's lock and finally assign
 * the evolved population. In addition to providing the above method, a UDI must also be default, copy and move
 * constructible. Also, since internally the pagmo::island class uses a separate thread of execution to provide
 * asynchronous behaviour, a UDI needs to guarantee a certain degree of thread-safety: it must be possible to interact
 * with the UDI while evolution is ongoing (e.g., it must be possible to copy the UDI while evolution is undergoing, or
 * call the <tt>get_name()</tt>, <tt>get_extra_info()</tt> methods, etc.), otherwise he behaviour will be undefined.
 *
 * In addition to the mandatory <tt>run_evolve()</tt> method, a UDI might implement the following optional methods:
 * @code{.unparsed}
 * std::string get_name() const;
 * std::string get_extra_info() const;
 * @endcode
 *
 * See the documentation of the corresponding methods in this class for details on how the optional
 * methods in the UDI are used by pagmo::island.
 *
 * **NOTE**: a moved-from pagmo::island is destructible and assignable. Any other operation will result
 * in undefined behaviour.
 */
class island
{
    // Handy shortcut.
    using idata_t = detail::island_data;
    // archi needs access to the internal of island.
    friend class archipelago;

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
     * - threading primitives,
     * - memory allocation errors,
     * - the copy constructors of pagmo::algorithm and pagmo::population.
     */
    island(const island &other)
        : m_ptr(detail::make_unique<idata_t>(other.m_ptr->isl_ptr->clone(), other.get_algorithm(),
                                             other.get_population()))
    {
        // NOTE: the idata_t ctor will set the archi ptr to null. The archi ptr is never copied.
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
        other.wait();
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
     * **NOTE**: this constructor is enabled only if \p a can be used to construct a
     * pagmo::algorithm and \p p is an instance of pagmo::population.
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
     * **NOTE**: this constructor is enabled only if:
     * - \p Isl satisfies pagmo::is_udi,
     * - \p a can be used to construct a pagmo::algorithm,
     * - \p p is an instance of pagmo::population.
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
     * **NOTE**: this constructor is enabled only if \p a can be used to construct a
     * pagmo::algorithm, and \p p, \p size and \p seed can be used to construct a pagmo::population.
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
     * **NOTE**: this constructor is enabled only if \p Isl satisfies pagmo::is_udi, \p a can be used to construct a
     * pagmo::algorithm, and \p p, \p size and \p seed can be used to construct a pagmo::population.
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
     * If the island has not been moved-from, the destructor will call island::wait().
     */
    ~island()
    {
        // If the island has been moved from, don't do anything.
        if (m_ptr) {
            wait();
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
                wait();
            }
            other.wait();
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
    /// Launch evolution.
    /**
     * This method will evolve the island's pagmo::population using the
     * island's pagmo::algorithm. The evolution happens asynchronously:
     * a call to island::evolve() will create an evolution task that will be pushed
     * to a queue, and then return immediately.
     * The tasks in the queue are consumed
     * by a separate thread of execution managed by the pagmo::island object,
     * which will invoke the <tt>run_evolve()</tt>
     * method of the UDI to perform the actual evolution. The island's population will be updated
     * at the end of each evolution task. Exceptions raised insided the tasks are stored within
     * the island object, and can be re-raised by calling wait().
     *
     * It is possible to call this method multiple times to enqueue multiple evolution tasks, which
     * will be consumed in a FIFO (first-in first-out) fashion. The user may call island::wait() to block until
     * all tasks have been completed, and to fetch exceptions raised during the execution of the tasks.
     *
     * @throws unspecified any exception thrown by:
     * - threading primitives,
     * - memory allocation errors,
     * - the public interface of \p std::future.
     */
    void evolve()
    {
        // First add an empty future, so that if an exception is thrown
        // we will not have modified m_futures, nor we will have a future
        // in flight which we cannot wait upon.
        m_ptr->futures.emplace_back();
        try {
            // Move assign a new future provided by the enqueue() method.
            // NOTE: enqueue either returns a valid future, or throws without
            // having enqueued any task.
            m_ptr->futures.back() = m_ptr->queue.enqueue([this]() { this->m_ptr->isl_ptr->run_evolve(*this); });
        } catch (...) {
            // We end up here only if enqueue threw. In such a case, we need to cleanup
            // the empty future we added above before re-throwing and exiting.
            m_ptr->futures.pop_back();
            throw;
        }
    }
    /// Block until evolution ends.
    /**
     * This method will block until all the evolution tasks enqueued via island::evolve() have been completed.
     * The method will also raise the first exception raised by any task enqueued since the last time wait() was called.
     *
     * @throws unspecified any exception thrown by:
     * - evolution tasks,
     * - threading primitives.
     */
    void get()
    {
        auto iwr = detail::wait_raii<>::getter();
        (void)iwr;
        for (decltype(m_ptr->futures.size()) i = 0; i < m_ptr->futures.size(); ++i) {
            // NOTE: this has to be valid, as the only way to get the value of the futures is via
            // this method, and we clear the futures vector after we are done.
            assert(m_ptr->futures[i].valid());
            try {
                m_ptr->futures[i].get();
            } catch (...) {
                // If any of the futures stores an exception, we will re-raise it.
                // But first, we need to get all the other futures and erase the futures
                // vector.
                for (i = i + 1u; i < m_ptr->futures.size(); ++i) {
                    try {
                        m_ptr->futures[i].get();
                    } catch (...) {
                    }
                }
                m_ptr->futures.clear();
                throw;
            }
        }
        m_ptr->futures.clear();
    }
    // TODO docs, document rationale, document noexcept.
    // TODO rationale of clearing the futures.
    void wait() noexcept
    {
        auto iwr = detail::wait_raii<>::getter();
        (void)iwr;
        for (const auto &f : m_ptr->futures) {
            // NOTE: this has to be valid, as the only way to get the value of the futures is via
            // this method, and we clear the futures vector after we are done.
            assert(f.valid());
            f.wait();
        }
        m_ptr->futures.clear();
    }
    /// Check island status.
    /**
     * @return \p true if the island is evolving, \p false otherwise.
     *
     * @throws unspecified any exception thrown by threading primitives.
     */
    bool busy() const
    {
        for (const auto &f : m_ptr->futures) {
            assert(f.valid());
            if (f.wait_for(std::chrono::duration<int>::zero()) != std::future_status::ready) {
                return true;
            }
        }
        return false;
    }
    /// Get the algorithm.
    /**
     * It is safe to call this method while the island is evolving.
     *
     * @return a copy of the island's algorithm.
     *
     * @throws unspecified any exception thrown by threading primitives.
     */
    algorithm get_algorithm() const
    {
        std::unique_lock<std::mutex> lock(m_ptr->algo_mutex);
        auto new_algo_ptr = m_ptr->algo;
        lock.unlock();
        return *new_algo_ptr;
    }
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
     * @throws unspecified any exception thrown by threading primitives.
     */
    population get_population() const
    {
        std::unique_lock<std::mutex> lock(m_ptr->pop_mutex);
        auto new_pop_ptr = m_ptr->pop;
        lock.unlock();
        return *new_pop_ptr;
    }
    void set_population(population pop)
    {
        auto new_pop_ptr = std::make_shared<population>(std::move(pop));
        std::lock_guard<std::mutex> lock(m_ptr->pop_mutex);
        m_ptr->pop = new_pop_ptr;
    }
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
     * It is safe to call this operator while the island is evolving.
     *
     * @param os the target stream.
     * @param isl the island.
     *
     * @return a reference to \p os.
     *
     * @throws unspecified any exception thrown by the stream operators of fundamental types,
     * pagmo::algorithm and pagmo::population, or by pagmo::island::get_extra_info().
     */
    friend std::ostream &operator<<(std::ostream &os, const island &isl)
    {
        stream(os, "Island name: ", isl.get_name());
        stream(os, "\n\tEvolving: ", isl.busy(), "\n\n");
        const auto extra_str = isl.get_extra_info();
        if (!extra_str.empty()) {
            stream(os, "Extra info:\n", extra_str, "\n\n");
        }
        stream(os, isl.get_algorithm(), "\n\n");
        stream(os, isl.get_population(), "\n\n");
        return os;
    }
    /// Save to archive.
    /**
     * This method will save \p this to the archive \p ar.
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
        ar(tmp_island.m_ptr->algo);
        ar(tmp_island.m_ptr->pop);
        *this = std::move(tmp_island);
    }

private:
    std::unique_ptr<idata_t> m_ptr;
};

/// Run evolve.
/**
 * This method will invoke the <tt>evolve()</tt> method on a copy of \p algo, using a copy
 * of \p pop as argument, and it will then assign the result of the evolution back to \p pop.
 *
 * @param algo the algorithm that will be used for the evolution.
 * @param algo_mutex a mutex regulating exclusive access to \p algo.
 * @param pop the population that will be evolved by \p algo.
 * @param pop_mutex a mutex regulating exclusive access to \p pop.
 *
 * @throws std::invalid_argument if either \p algo or <tt>pop</tt>'s problem do not provide
 * at least the pagmo::thread_safety::basic thread safety guarantee.
 * @throws unspecified any exception thrown by:
 * - threading primitives,
 * - the copy constructors of \p pop and \p algo,
 * - the <tt>evolve()</tt> method of \p algo.
 */
inline void thread_island::run_evolve(island &isl) const
{
    // const auto i_ts = isl.get_thread_safety();
    // TODO check thread safety.
    // TODO doc.

    isl.set_population(isl.get_algorithm().evolve(isl.get_population()));
}

class archipelago
{
public:
    using size_type = std::vector<std::unique_ptr<island>>::size_type;
    archipelago()
    {
    }
    archipelago(const archipelago &other)
    {
        for (const auto &iptr : other.m_islands) {
            // This will end up copying the island members,
            // and assign the archi pointer as well.
            push_back(*iptr);
        }
    }
    archipelago(archipelago &&other) noexcept
    {
        // NOTE: in move operations we have to wait, because the ongoing
        // island evolutions are interacting with their hosting archi 'other'.
        // We cannot just move in the vector of islands.
        try {
            other.wait();
        } catch (const std::system_error &) {
            std::abort();
        } catch (...) {
        }
        // Move in the islands.
        m_islands = std::move(other.m_islands);
        // Re-direct the archi pointers to point to this.
        for (const auto &iptr : m_islands) {
            iptr->m_ptr->archi_ptr = this;
        }
    }

private:
    template <typename... Args>
    using n_ctor_enabler = enable_if_t<std::is_constructible<island, Args &&...>::value, int>;

public:
    template <typename... Args, n_ctor_enabler<Args...> = 0>
    explicit archipelago(size_type n, Args &&... args)
    {
        for (size_type i = 0; i < n; ++i) {
            push_back(std::forward<Args>(args)...);
        }
    }
    archipelago &operator=(const archipelago &other)
    {
        if (this != &other) {
            *this = archipelago(other);
        }
        return *this;
    }
    archipelago &operator=(archipelago &&other) noexcept
    {
        if (this != &other) {
            // NOTE: as in the move ctor, we need to wait on other.
            try {
                other.wait();
            } catch (const std::system_error &) {
                std::abort();
            } catch (...) {
            }
            // Move in the islands.
            m_islands = std::move(other.m_islands);
            // Re-direct the archi pointers to point to this.
            for (const auto &iptr : m_islands) {
                iptr->m_ptr->archi_ptr = this;
            }
        }
        return *this;
    }
    ~archipelago()
    {
        assert(std::all_of(m_islands.begin(), m_islands.end(),
                           [this](const std::unique_ptr<island> &iptr) { return iptr->m_ptr->archi_ptr == this; }));
    }

private:
    template <typename... Args>
    using push_back_enabler = n_ctor_enabler<Args...>;

public:
    island &operator[](size_type i)
    {
        if (i >= size()) {
            pagmo_throw(std::out_of_range, "cannot access the island at index " + std::to_string(i)
                                               + ": the archipelago has a size of only " + std::to_string(size()));
        }
        return *m_islands[i];
    }
    const island &operator[](size_type i) const
    {
        if (i >= size()) {
            pagmo_throw(std::out_of_range, "cannot access the island at index " + std::to_string(i)
                                               + ": the archipelago has a size of only " + std::to_string(size()));
        }
        return *m_islands[i];
    }
    size_type size() const
    {
        return m_islands.size();
    }
    template <typename... Args, push_back_enabler<Args...> = 0>
    void push_back(Args &&... args)
    {
        m_islands.emplace_back(detail::make_unique<island>(std::forward<Args>(args)...));
        m_islands.back()->m_ptr->archi_ptr = this;
    }
    void evolve()
    {
        for (auto &iptr : m_islands) {
            iptr->evolve();
        }
    }
    void wait()
    {
        for (auto it = m_islands.begin(); it != m_islands.end(); ++it) {
            try {
                (*it)->wait();
            } catch (...) {
                for (it = it + 1; it != m_islands.end(); ++it) {
                    try {
                        (*it)->wait();
                    } catch (...) {
                    }
                }
                throw;
            }
        }
    }
    bool busy() const
    {
        return std::any_of(m_islands.begin(), m_islands.end(),
                           [](const std::unique_ptr<island> &iptr) { return iptr->busy(); });
    }
    friend std::ostream &operator<<(std::ostream &os, const archipelago &archi)
    {
        stream(os, "Number of islands: ", archi.size(), "\n");
        stream(os, "Evolving: ", archi.busy(), "\n\n");
        stream(os, "Islands summaries:\n\n");
        detail::table t({"#", "Size", "Algo", "Prob", "Evolving"}, "\t");
        for (decltype(archi.size()) i = 0; i < archi.size(); ++i) {
            t.add_row({std::to_string(i), std::to_string(archi[i].get_population().size()),
                       archi[i].get_algorithm().get_name(), archi[i].get_population().get_problem().get_name(),
                       archi[i].busy() ? "True" : "False"});
        }
        stream(os, t);
        return os;
    }
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(m_islands);
    }
    template <typename Archive>
    void load(Archive &ar)
    {
        archipelago tmp;
        ar(tmp.m_islands);
        *this = std::move(tmp);
    }

private:
    std::vector<std::unique_ptr<island>> m_islands;
};
}

PAGMO_REGISTER_ISLAND(pagmo::thread_island)

#endif
