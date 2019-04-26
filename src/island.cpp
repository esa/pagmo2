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

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <initializer_list>
#include <ios>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(PAGMO_WITH_FORK_ISLAND)

#include <sys/types.h>
#include <unistd.h>
#include <wait.h>

#endif

#include <boost/any.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

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
void thread_island::run_evolve(island &isl) const
{
    const auto i_ts = isl.get_thread_safety();
    if (i_ts[0] < thread_safety::basic) {
        pagmo_throw(std::invalid_argument,
                    "the 'thread_island' UDI requires an algorithm providing at least the 'basic' "
                    "thread safety guarantee");
    }
    if (i_ts[1] < thread_safety::basic) {
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

/// Serialization support.
/**
 * This class is stateless, no data will be saved to or loaded from the archive.
 */
template <typename Archive>
void thread_island::serialize(Archive &, unsigned)
{
}

#if defined(PAGMO_WITH_FORK_ISLAND)

namespace detail
{

namespace
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
                pagmo_throw(std::runtime_error,
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
                pagmo_throw(std::runtime_error,
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
            pagmo_throw(std::runtime_error, "Unable to read from a pipe with the read() function. The error code is "
                                                + std::to_string(errno) + " and the error message is: '"
                                                + std::strerror(errno) + "'");
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
            pagmo_throw(std::runtime_error, "Unable to write to a pipe with the write() function. The error code is "
                                                + std::to_string(errno) + " and the error message is: '"
                                                + std::strerror(errno) + "'");
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

} // namespace

} // namespace detail

void fork_island::run_evolve(island &isl) const
{
    // The structure we use to pass messages from the child to the parent:
    // - int, status flag,
    // - string, error message,
    // - the algorithm used for evolution,
    // - the evolved population.
    using message_t = std::tuple<int, std::string, algorithm, population>;
    // A message that will be used both by parent and child.
    message_t m;
    // The pipe.
    detail::pipe_t p;
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
            char buffer[100];
            std::stringstream ss;
            {
                boost::archive::binary_iarchive iarchive(ss);
                while (true) {
                    const auto read_bytes = p.read(static_cast<void *>(buffer), sizeof(buffer));
                    if (!read_bytes) {
                        // EOF, break out.
                        break;
                    }
                    ss.write(buffer, static_cast<std::streamsize>(read_bytes));
                }
                iarchive >> m;
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
            // Issue also a waitpid in order to clean up the zombie process.
            // Ignore the return value, as we are just trying to clean up here.
            ::waitpid(child_pid, nullptr, 0);
            // Re-raise.
            throw;
        }
        // Wait on the child.
        // NOTE: this is necessary because, if we don't do this,
        // the child process becomes a zombie and its entry in the process
        // table is not freed up. This will eventually lead to
        // failure in the creation of new child processes.
        if (::waitpid(child_pid, nullptr, 0) != child_pid) {
            // LCOV_EXCL_START
            pagmo_throw(std::runtime_error, "The waitpid() function returned an error while attempting to wait for the "
                                            "child process in fork_island");
            // LCOV_EXCL_STOP
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
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << ms;
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

// Extra info: report the child process' ID, if evolution
// is active.
std::string fork_island::get_extra_info() const
{
    const auto pid = m_pid.load();
    if (pid) {
        return "\tChild PID: " + std::to_string(pid);
    }
    return "\tNo active child";
}

template <typename Archive>
void fork_island::serialize(Archive &, unsigned)
{
}

#endif

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
void island::set_algorithm(algorithm algo)
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
population island::get_population() const
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
void island::set_population(population pop)
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
std::array<thread_safety, 2> island::get_thread_safety() const
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

// NOTE: same utility method as in pagmo::island, see there.
void archipelago::wait_check_ignore()
{
    try {
        wait_check();
    } catch (...) {
    }
}

/// Default constructor.
/**
 * The default constructor will initialise an empty archipelago.
 */
archipelago::archipelago() {}

/// Copy constructor.
/**
 * The islands of \p other will be copied into \p this via archipelago::push_back().
 *
 * @param other the archipelago that will be copied.
 *
 * @throws unspecified any exception thrown by archipelago::push_back().
 */
archipelago::archipelago(const archipelago &other)
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
archipelago::archipelago(archipelago &&other) noexcept
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
archipelago &archipelago::operator=(const archipelago &other)
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
archipelago &archipelago::operator=(archipelago &&other) noexcept
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
archipelago::~archipelago()
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
island &archipelago::operator[](size_type i)
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
const island &archipelago::operator[](size_type i) const
{
    if (i >= size()) {
        pagmo_throw(std::out_of_range, "cannot access the island at index " + std::to_string(i)
                                           + ": the archipelago has a size of only " + std::to_string(size()));
    }
    return *m_islands[i];
}

void archipelago::evolve(unsigned n)
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
void archipelago::wait() noexcept
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
void archipelago::wait_check()
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
evolve_status archipelago::status() const
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

/// Get the fitness vectors of the islands' champions.
/**
 * @return a collection of the fitness vectors of the islands' champions.
 *
 * @throws unspecified any exception thrown by population::champion_f() or
 * by memory errors in standard containers.
 */
std::vector<vector_double> archipelago::get_champions_f() const
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
std::vector<vector_double> archipelago::get_champions_x() const
{
    std::vector<vector_double> retval;
    for (const auto &isl_ptr : m_islands) {
        retval.emplace_back(isl_ptr->get_population().champion_x());
    }
    return retval;
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
std::ostream &operator<<(std::ostream &os, const archipelago &archi)
{
    stream(os, "Number of islands: ", archi.size(), "\n");
    stream(os, "Status: ", archi.status(), "\n\n");
    stream(os, "Islands summaries:\n\n");
    detail::table t({"#", "Type", "Algo", "Prob", "Size", "Status"}, "\t");
    for (decltype(archi.size()) i = 0; i < archi.size(); ++i) {
        const auto pop = archi[i].get_population();
        t.add_row(i, archi[i].get_name(), archi[i].get_algorithm().get_name(), pop.get_problem().get_name(), pop.size(),
                  archi[i].status());
    }
    stream(os, t);
    return os;
}

} // namespace pagmo

PAGMO_S11N_ISLAND_IMPLEMENT(pagmo::thread_island)

#if defined(PAGMO_WITH_FORK_ISLAND)

PAGMO_S11N_ISLAND_IMPLEMENT(pagmo::fork_island)

#endif
