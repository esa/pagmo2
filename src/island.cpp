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

#include <pagmo/config.hpp>

#include <cassert>
#include <chrono>
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
#include <tuple>
#include <unordered_map>
#include <utility>

#include <boost/any.hpp>
#include <boost/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/detail/gte_getter.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/thread_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#if defined(PAGMO_WITH_FORK_ISLAND)
#include <pagmo/islands/fork_island.hpp>
#endif

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

namespace detail
{

// NOTE: this is just a simple wrapper to force noexcept behaviour on std::future::wait().
// If f.wait() throws something, the program will terminate. A valid std::future should not
// throw, but technically the standard does not guarantee that. Having this noexcept wrapper
// simplifies reasoning about exception behaviour in wait(), wait_check(), etc.
void wait_f(const std::future<void> &f) noexcept
{
    assert(f.valid());
    f.wait();
}

// Small helper to determine if a future holds an exception.
// The noexcept reasoning is the same as above. Here we could fail
// because of memory errors, but there's not much we can do in such
// a case.
bool future_has_exception(std::future<void> &f) noexcept
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
bool future_running(const std::future<void> &f)
{
    return f.wait_for(std::chrono::duration<int>::zero()) != std::future_status::ready;
}

} // namespace detail

namespace detail
{

namespace
{

boost::any default_wait_raii_getter()
{
    return boost::any{};
}

} // namespace

// NOTE: the default implementation just returns a defcted boost::any, whose ctor and dtor
// will have no effect.
std::function<boost::any()> wait_raii_getter = &default_wait_raii_getter;

namespace
{

// This is the default UDI type selector. It will select the thread_island if both algorithm
// and population provide at least the basic thread safety guarantee. Otherwise, it will select
// the fork_island, if available.
void default_island_factory(const algorithm &algo, const population &pop, std::unique_ptr<detail::isl_inner_base> &ptr)
{
    (void)algo;
    (void)pop;
#if defined(PAGMO_WITH_FORK_ISLAND)
    if (algo.get_thread_safety() < thread_safety::basic
        || pop.get_problem().get_thread_safety() < thread_safety::basic) {
        ptr = detail::make_unique<isl_inner<fork_island>>();
        return;
    }
#endif
    ptr = detail::make_unique<isl_inner<thread_island>>();
}

} // namespace

// Static init.
std::function<void(const algorithm &, const population &, std::unique_ptr<detail::isl_inner_base> &)> island_factory
    = &default_island_factory;

// NOTE: thread_island is ok as default choice, as the null_prob/null_algo
// are both thread safe.
island_data::island_data()
    : isl_ptr(detail::make_unique<isl_inner<thread_island>>()), algo(std::make_shared<algorithm>()),
      pop(std::make_shared<population>())
{
}

const std::unordered_map<evolve_status, std::string, island_status_hasher> island_statuses
    = {{evolve_status::idle, "idle"},
       {evolve_status::busy, "busy"},
       {evolve_status::idle_error, "idle - **error occurred**"},
       {evolve_status::busy_error, "busy - **error occurred**"}};

} // namespace detail

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Provide the stream operator overload for evolve_status.
std::ostream &operator<<(std::ostream &os, evolve_status es)
{
    return os << detail::island_statuses.at(es);
}

#endif

// NOTE: the idea in the move members and the dtor is that
// we want to wait *and* erase any future in the island, before doing
// the move/destruction. Thus we use this small wrapper.
void island::wait_check_ignore()
{
    try {
        wait_check();
        // LCOV_EXCL_START
    } catch (...) {
    }
    // LCOV_EXCL_STOP
}

/// Default constructor.
/**
 * The default constructor will initialise an island containing a UDI of type pagmo::thread_island,
 * and default-constructed pagmo::algorithm and pagmo::population.
 *
 * @throws unspecified any exception thrown by any invoked constructor or by memory allocation failures.
 */
island::island() : m_ptr(detail::make_unique<idata_t>()) {}

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
island::island(const island &other)
    : m_ptr(detail::make_unique<idata_t>(other.m_ptr->isl_ptr->clone(), other.get_algorithm(), other.get_population()))
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
island::island(island &&other) noexcept
{
    other.wait_check_ignore();
    m_ptr = std::move(other.m_ptr);
}

/// Destructor.
/**
 * If the island has not been moved-from, the destructor will call island::wait_check(),
 * ignoring any exception that might be thrown.
 */
island::~island()
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
island &island::operator=(island &&other) noexcept
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
island &island::operator=(const island &other)
{
    if (this != &other) {
        *this = island(other);
    }
    return *this;
}

void island::evolve(unsigned n)
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
            // Random bits for use in the migration logic.
            // Wrap them in optionals so that, if we don't need
            // them, we don't waste cycles initialising them.
            boost::optional<std::mt19937> migr_eng;
            // Migration probability distribution.
            boost::optional<std::uniform_real_distribution<double>> pdist;
            // Distribution for selecting the connecting island.
            boost::optional<std::uniform_int_distribution<archipelago::size_type>> idist;

            // Cache the archi pointer.
            const auto aptr = this->m_ptr->archi_ptr;

            // Figure out what is the island's index in the archi, if we are
            // in an archi. Otherwise, this variable will be unused.
            const auto isl_idx = aptr ? aptr->get_island_idx(*this) : 0u;

            for (auto i = 0u; i < n; ++i) {
                if (aptr) {
                    // If the island is in an archi, before
                    // launching the evolution migrate the
                    // individuals from the connecting islands.
                    const auto connections = aptr->get_island_connections(isl_idx);
                    assert(connections.first.size() == connections.second.size());

                    // Do something only if we actually have connections.
                    if (connections.first.size()) {
                        // Init the rng bits, if necessary.
                        if (!migr_eng) {
                            assert(!pdist);
                            assert(!idist);

                            migr_eng.emplace(static_cast<std::mt19937::result_type>(random_device::next()));
                            pdist.emplace(0., 1.);
                            idist.emplace();
                        }

                        // Pick a random island index among the islands connecting to this.
                        const auto idx
                            = (*idist)(*migr_eng, std::uniform_int_distribution<archipelago::size_type>::param_type(
                                                      0, connections.first.size() - 1u));

                        // Throw the dice against the migration probability.
                        if ((*pdist)(*migr_eng)
                            < connections.second[static_cast<decltype(connections.second.size())>(idx)]) {
                            // Extract the candidate migrants from the archipelago.
                            const auto migrants = aptr->extract_migrants(idx);

                            // Extract the migration data from the island.
                            const auto mig_data = this->get_migration_data();
                            // auto new_inds = this->m_ptr->r_pol_ptr->replace(inds, migrants);
                            // Determine which individuals were migrated from migrants into new_inds,
                            // log them.
                            // this->set_individuals(new_inds);
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
void island::wait_check()
{
    auto iwr = detail::wait_raii_getter();
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
void island::wait()
{
    // NOTE: we use this function in move ops and in the dtor, which are all noexcept. In theory we could
    // end up aborting in case the wait_raii mechanism throws in such cases. We could also end up aborting
    // due to memory failures in future_has_exception().
    // NOTE: the idea here is that, after a wait() call, all the futures of successful tasks have been erased,
    // with at most 1 surviving future from the first throwing task. This way, wait() does some cleaning up
    // behind the scenes, without changing the behaviour of successive wait_check() and status() calls: wait_check()
    // will still re-throw the first exception, and status() will still return idle_error.
    auto iwr = detail::wait_raii_getter();
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
evolve_status island::status() const
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
algorithm island::get_algorithm() const
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

    // Create a new reference to the internal algo
    // (this involves only C++ operations).
    std::shared_ptr<algorithm> new_algo_ptr;
    {
        std::lock_guard<std::mutex> lock(m_ptr->algo_mutex);
        new_algo_ptr = m_ptr->algo;
    }

    // Return a copy.
    // NOTE: when exiting the function, the dtor of new_algo_ptr
    // will be called. This will result in the refcount
    // decreasing, and, if new_algo_ptr is the last existing reference,
    // in the call of the dtor of the internal algorithm. This could
    // be the case, for instance, if we are using set_algorithm() from
    // another thread.
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
void island::set_algorithm(const algorithm &algo)
{
    // Step 1: create a new shared ptr to a copy of algo.
    auto new_algo_ptr = std::make_shared<algorithm>(algo);

    // Step 2: init an empty algorithm pointer.
    std::shared_ptr<algorithm> old_ptr;

    // Step 3: store a reference to the old algo
    // in old_ptr, and assign a reference to the
    // new algo.
    {
        std::lock_guard<std::mutex> lock(m_ptr->algo_mutex);
        old_ptr = m_ptr->algo;
        // NOTE: this assignment will never invoke
        // the destructor of the object pointed-to
        // by m_ptr->algo, as we made sure
        // to create a new reference above.
        m_ptr->algo = new_algo_ptr;
    }

    // NOTE: upon exit, the refcount of old_ptr and
    // new_algo_ptr will be decreased, possibly invoking
    // the dtor of the contained objects.
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
population island::get_population() const
{
    // NOTE: same pattern as in get_algorithm().
    std::shared_ptr<population> new_pop_ptr;
    {
        std::lock_guard<std::mutex> lock(m_ptr->pop_mutex);
        new_pop_ptr = m_ptr->pop;
    }

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
void island::set_population(const population &pop)
{
    // Same pattern as in set_algorithm().
    auto new_pop_ptr = std::make_shared<population>(pop);

    std::shared_ptr<population> old_ptr;

    {
        std::lock_guard<std::mutex> lock(m_ptr->pop_mutex);
        old_ptr = m_ptr->pop;
        m_ptr->pop = new_pop_ptr;
    }
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
std::string island::get_name() const
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
std::string island::get_extra_info() const
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
std::ostream &operator<<(std::ostream &os, const island &isl)
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

/// Check if the island is in a valid state.
/**
 * @return ``false`` if ``this`` was moved from, ``true`` otherwise.
 */
bool island::is_valid() const
{
    return static_cast<bool>(m_ptr);
}

// Get the migration data.
island::migration_data_t island::get_migration_data() const
{
    migration_data_t retval;

    {
        // NOTE: this helper is called from the separate
        // thread of execution within pagmo::island. We need to protect
        // with a gte.
        auto gte = detail::gte_getter();
        (void)gte;

        // Get a copy of the population.
        auto tmp_pop(get_population());

        // Move out the individuals.
        std::get<0>(std::get<0>(retval)) = std::move(tmp_pop.m_ID);
        std::get<1>(std::get<0>(retval)) = std::move(tmp_pop.m_x);
        std::get<2>(std::get<0>(retval)) = std::move(tmp_pop.m_f);

        // nobj, nec, nic, nix.
        std::get<1>(retval) = tmp_pop.get_problem().get_nobj();
        std::get<2>(retval) = tmp_pop.get_problem().get_nec();
        std::get<3>(retval) = tmp_pop.get_problem().get_nic();
        std::get<4>(retval) = tmp_pop.get_problem().get_nix();

        // The vector of tolerances.
        std::get<5>(retval) = tmp_pop.get_problem().get_c_tol();
    }

    return retval;
}

// Set all the individuals in the population.
void island::set_individuals(const individuals_group_t &inds)
{
    auto tmp_inds(inds);

    {
        // NOTE: this helper is called from the separate
        // thread of execution within pagmo::island. We need to protect
        // with a gte.
        auto gte = detail::gte_getter();
        (void)gte;

        auto tmp_pop(get_population());

        tmp_pop.m_ID = std::move(std::get<0>(tmp_inds));
        tmp_pop.m_x = std::move(std::get<1>(tmp_inds));
        tmp_pop.m_f = std::move(std::get<2>(tmp_inds));

        set_population(tmp_pop);
    }
}

} // namespace pagmo
