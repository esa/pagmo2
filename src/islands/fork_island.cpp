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

#include <atomic>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <ios>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/fork_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>

namespace pagmo
{

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
            {
                // Prepare a local buffer and a stringstream, then read the data from the child.
                char buffer[100];
                std::stringstream ss;
                while (true) {
                    const auto read_bytes = p.read(static_cast<void *>(buffer), sizeof(buffer));
                    if (!read_bytes) {
                        // EOF, break out.
                        break;
                    }
                    ss.write(buffer, static_cast<std::streamsize>(read_bytes));
                }
                boost::archive::binary_iarchive iarchive(ss);
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
        isl.set_algorithm(std::get<2>(m));
        isl.set_population(std::get<3>(m));
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

} // namespace pagmo

PAGMO_S11N_ISLAND_IMPLEMENT(pagmo::fork_island)
