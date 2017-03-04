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

#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <pagmo/algorithm.hpp>
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

// Fwd decl.
class archipelago;

template <typename T>
class has_enqueue_evolution
{
    template <typename U>
    using enqueue_evolution_t = decltype(
        std::declval<U &>().enqueue_evolution(std::declval<const algorithm &>(), std::declval<archipelago *>()));
    static const bool implementation_defined = is_detected<enqueue_evolution_t, T>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_enqueue_evolution<T>::value;

template <typename T>
class has_wait
{
    template <typename U>
    using wait_t = decltype(std::declval<const U &>().wait());
    static const bool implementation_defined = is_detected<wait_t, T>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_wait<T>::value;

template <typename T>
class has_get_population
{
    template <typename U>
    using get_population_t = decltype(std::declval<const U &>().get_population());
    static const bool implementation_defined = std::is_same<population, detected_t<get_population_t, T>>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_population<T>::value;

namespace detail
{

// Specialise this to true in order to disable all the UDI checks and mark a type
// as a UDI regardless of the features provided by it.
template <typename>
struct disable_udi_checks : std::false_type {
};
}

template <typename T>
class is_udi
{
    static const bool implementation_defined
        = (std::is_same<T, uncvref_t<T>>::value && std::is_default_constructible<T>::value
           && std::is_copy_constructible<T>::value && std::is_move_constructible<T>::value
           && std::is_destructible<T>::value && has_enqueue_evolution<T>::value && has_wait<T>::value
           && has_get_population<T>::value)
          || detail::disable_udi_checks<T>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool is_udi<T>::value;

// TODO fill up.
struct null_island {
    void enqueue_evolution(const algorithm &, archipelago *)
    {
    }
    void wait() const
    {
    }
    population get_population() const
    {
        return population{};
    }
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

class thread_island
{
    template <typename T>
    static void check_thread_safety(const T &x)
    {
        if (static_cast<int>(x.get_thread_safety()) < static_cast<int>(thread_safety::basic)) {
            pagmo_throw(
                std::invalid_argument,
                "thread islands require objects which provide at least the basic thread safety level, but the object '"
                    + x.get_name() + "' does not provide any thread safety guarantee");
        }
    }

public:
    thread_island() = default;
    explicit thread_island(const population &pop) : m_pop(pop)
    {
        check_thread_safety(m_pop.get_problem());
    }
    explicit thread_island(population &&pop) : m_pop(std::move(pop))
    {
        check_thread_safety(m_pop.get_problem());
    }
    thread_island(const thread_island &other) : m_pop(other.get_population())
    {
        // NOTE: check here as well, in case copying the problem does something strange to its
        // thread safety.
        check_thread_safety(m_pop.get_problem());
    }
    // NOTE: the noexcept move semantics here means that if the pop mutex throws, the program
    // will abort. We can avoid this behaviour by adopting an atomic-based spinlock instead
    // of std::mutex. Let's wait and see if this becomes a problem in practice.
    // NOTE: no need to re-check thread safety when move operations are involved: in the problem class,
    // we have a pointer to the UDP and the move operation just moves the pointer without invoking
    // any operation on the UDP.
    thread_island(thread_island &&other) noexcept : m_pop(other.move_out_population())
    {
    }
    ~thread_island()
    {
        try {
            wait();
        } catch (...) {
        }
    }
    thread_island &operator=(const thread_island &other)
    {
        if (this != &other) {
            *this = thread_island(other);
        }
        return *this;
    }
    thread_island &operator=(thread_island &&other) noexcept
    {
        if (this != &other) {
            move_in_population(other.move_out_population());
        }
        return *this;
    }
    void enqueue_evolution(const algorithm &, archipelago *);
    void wait() const
    {
        std::lock_guard<std::mutex> lock(m_futures_mutex);
        for (decltype(m_futures.size()) i = 0; i < m_futures.size(); ++i) {
            // NOTE: this has to be valid, as the only way to get the value of the futures is via
            // this method, and we clear the futures vector after we are done.
            assert(m_futures[i].valid());
            try {
                m_futures[i].get();
            } catch (...) {
                // If any of the futures stores an exception, we will re-raise it.
                // But first, we need to get all the other futures and erase the futures
                // vector.
                for (i = i + 1u; i < m_futures.size(); ++i) {
                    try {
                        m_futures[i].get();
                    } catch (...) {
                    }
                }
                m_futures.clear();
                throw;
            }
        }
        m_futures.clear();
    }
    bool busy() const
    {
        std::lock_guard<std::mutex> lock(m_futures_mutex);
        for (const auto &f : m_futures) {
            assert(f.valid());
            if (f.wait_for(std::chrono::duration<int>::zero()) != std::future_status::ready) {
                return true;
            }
        }
        return false;
    }
    population get_population() const
    {
        std::lock_guard<std::mutex> lock(m_pop_mutex);
        return m_pop;
    }
    std::string get_name() const
    {
        return "Thread island";
    }
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(get_population());
    }
    template <typename Archive>
    void load(Archive &ar)
    {
        population tmp;
        ar(tmp);
        check_thread_safety(tmp.get_problem());
        move_in_population(std::move(tmp));
    }

private:
    void move_in_population(population &&pop)
    {
        std::lock_guard<std::mutex> lock(m_pop_mutex);
        m_pop = std::move(pop);
    }
    population move_out_population()
    {
        std::lock_guard<std::mutex> lock(m_pop_mutex);
        return std::move(m_pop);
    }

private:
    population m_pop;
    detail::task_queue m_queue;
    mutable std::mutex m_pop_mutex;
    mutable std::vector<std::future<void>> m_futures;
    mutable std::mutex m_futures_mutex;
};

namespace detail
{

struct isl_inner_base {
    virtual ~isl_inner_base()
    {
    }
    virtual isl_inner_base *clone() const = 0;
    virtual void enqueue_evolution(const algorithm &, archipelago *) = 0;
    virtual void wait() const = 0;
    virtual population get_population() const = 0;
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
    virtual isl_inner_base *clone() const override final
    {
        return ::new isl_inner(m_value);
    }
    // The enqueue_evolution() method.
    virtual void enqueue_evolution(const algorithm &algo, archipelago *archi) override final
    {
        m_value.enqueue_evolution(algo, archi);
    }
    // The wait() method.
    virtual void wait() const override final
    {
        m_value.wait();
    }
    // The get_population() method.
    virtual population get_population() const override final
    {
        return m_value.get_population();
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
    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<isl_inner_base>(this), m_value);
    }
    T m_value;
};
}

class island
{
    friend class archipelago;
    template <typename Algo, typename Isl>
    using generic_ctor_enabler
        = enable_if_t<std::is_constructible<algorithm, Algo &&>::value && !std::is_same<island, uncvref_t<Isl>>::value
                          && is_udi<uncvref_t<Isl>>::value,
                      int>;

public:
    island() : island(null_algorithm{}, null_island{})
    {
    }
    template <typename Algo, typename Isl, generic_ctor_enabler<Algo, Isl> = 0>
    explicit island(Algo &&a, Isl &&isl)
        : m_algo(std::forward<Algo>(a)), m_ptr(::new detail::isl_inner<uncvref_t<Isl>>(std::forward<Isl>(isl)))
    {
    }

private:
    template <typename Algo, typename Prob>
    using algo_prob_ctor_enabler = enable_if_t<std::is_constructible<algorithm, Algo &&>::value
                                                   && std::is_constructible<problem, Prob &&>::value,
                                               int>;

public:
    template <typename Algo, typename Prob, algo_prob_ctor_enabler<Algo, Prob> = 0>
    explicit island(Algo &&a, Prob &&p, population::size_type pop_size, unsigned seed = pagmo::random_device::next())
        : island(std::forward<Algo>(a), population(std::forward<Prob>(p), pop_size, seed))
    {
    }

private:
    template <typename Algo>
    using algo_pop_ctor_enabler = enable_if_t<std::is_constructible<algorithm, Algo &&>::value, int>;

public:
    template <typename Algo, algo_pop_ctor_enabler<Algo> = 0>
    explicit island(Algo &&a, const population &pop) : m_algo(std::forward<Algo>(a))
    {
        thread_island t_isl(pop);
        m_ptr.reset(::new detail::isl_inner<thread_island>(std::move(t_isl)));
    }
    template <typename Algo, algo_pop_ctor_enabler<Algo> = 0>
    explicit island(Algo &&a, population &&pop) : m_algo(std::forward<Algo>(a))
    {
        thread_island t_isl(std::move(pop));
        m_ptr.reset(::new detail::isl_inner<thread_island>(std::move(t_isl)));
    }
    island(const island &other) : m_algo(other.m_algo), m_ptr(other.m_ptr->clone())
    {
    }
    island(island &&other) noexcept : m_algo(std::move(other.m_algo)), m_ptr(std::move(other.m_ptr))
    {
    }
    island &operator=(const island &other)
    {
        if (this != &other) {
            *this = island(other);
        }
        return *this;
    }
    island &operator=(island &&other) noexcept
    {
        if (this != &other) {
            m_algo = std::move(other.m_algo);
            m_ptr = std::move(other.m_ptr);
        }
        return *this;
    }
    void evolve()
    {
        ptr()->enqueue_evolution(m_algo, m_archi);
    }
    void wait() const
    {
        ptr()->wait();
    }
    population get_population() const
    {
        return ptr()->get_population();
    }
    std::string get_name() const
    {
        return ptr()->get_name();
    }
    std::string get_extra_info() const
    {
        return ptr()->get_extra_info();
    }
    const algorithm &get_algorithm() const
    {
        return m_algo;
    }
    algorithm &get_algorithm()
    {
        return m_algo;
    }
    friend std::ostream &operator<<(std::ostream &os, const island &isl)
    {
        stream(os, "Island name: ", isl.get_name());
        stream(os, "\n\nAlgorithm:\n", isl.get_algorithm());
        stream(os, "\n\nPopulation:\n", isl.get_population());
        const auto extra_str = isl.get_extra_info();
        if (!extra_str.empty()) {
            stream(os, "\nExtra info:\n", extra_str);
        }
        return os;
    }
    template <typename Archive>
    void save(Archive &ar) const
    {
        ar(m_algo, m_ptr);
    }
    template <typename Archive>
    void load(Archive &ar)
    {
        island tmp_island;
        ar(tmp_island.m_algo, tmp_island.m_ptr);
        *this = std::move(tmp_island);
    }

private:
    // Two small helpers to make sure that whenever we require
    // access to the pointer it actually points to something.
    detail::isl_inner_base const *ptr() const
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }
    detail::isl_inner_base *ptr()
    {
        assert(m_ptr.get() != nullptr);
        return m_ptr.get();
    }

private:
    // The algorithm.
    algorithm m_algo;
    // Pointer to the inner base island.
    std::unique_ptr<detail::isl_inner_base> m_ptr;
    // Archipelago pointer.
    archipelago *m_archi = nullptr;
};

class archipelago
{
};

// NOTE: place it here, as we need the definition of archipelago.
inline void thread_island::enqueue_evolution(const algorithm &algo, archipelago *archi)
{
    check_thread_safety(algo);
    std::lock_guard<std::mutex> lock(m_futures_mutex);
    // First add an empty future, so that if an exception is thrown
    // we will not have modified m_futures, nor we will have a future
    // in flight which we cannot wait upon.
    m_futures.emplace_back();
    try {
        // Move assign a new future provided by the enqueue() method.
        // NOTE: enqueue either returns a valid future, or throws without
        // having enqueued any task.
        // NOTE: it is important to copy algo here, because if we stored a reference
        // to it, the reference might be modified from pagmo::island and we would have
        // a data race.
        m_futures.back() = m_queue.enqueue([this, algo, archi]() {
            // Check it again, in the remote case copying it creates an algo with no thread safety.
            check_thread_safety(algo);
            auto new_pop = algo.evolve(this->get_population());
            if (static_cast<int>(new_pop.get_problem().get_thread_safety()) < static_cast<int>(thread_safety::basic)) {
                pagmo_throw(std::runtime_error,
                            "An evolve() call in thread_island returned a population with a "
                            "problem providing no thread safety. This most likely indicates an error "
                            "in the implementation of the user-defined problem/algorithm.");
            }
            this->move_in_population(std::move(new_pop));
        });
    } catch (...) {
        // We end up here only if enqueue threw. In such a case, we need to cleanup
        // the empty future we added above before re-throwing and exiting.
        m_futures.pop_back();
        throw;
    }
}
}

PAGMO_REGISTER_ISLAND(pagmo::null_island)

PAGMO_REGISTER_ISLAND(pagmo::thread_island)

#endif
