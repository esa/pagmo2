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

#ifndef PAGMO_FORK_ISLAND_HPP
#define PAGMO_FORK_ISLAND_HPP

#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ios>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include <sys/types.h>
#include <unistd.h>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/serialization.hpp>

namespace pagmo
{

class fork_island
{
    static void raise_errno(const std::string &fname)
    {
        pagmo_throw(std::runtime_error, "The call to the function " + fname
                                            + "() in the fork island failed with error code " + std::to_string(errno)
                                            + " and the following error message: '" + std::strerror(errno) + "'");
    }
    static void exit_unrecoverable()
    {
        std::cerr << "An unrecoverable error occurred in fork_island, giving up now :(" << std::endl;
        std::exit(1);
    }
    // Small RAII wrapper around a pipe.
    struct pipe_t {
        pipe_t()
        {
            int fd[2];
            if (pipe(fd) == -1) {
                raise_errno("pipe");
            }
            // The pipe was successfully opened, copy over
            // the r/w descriptors.
            read = fd[0];
            write = fd[1];
        }
        ~pipe_t()
        {
            // Attempt to close the pipe on destruction.
            // These calls could fail in theory.
            close(read);
            close(write);
        }
        int read, write;
    };
    using message_t = std::tuple<int, std::string, algorithm, population>;

public:
    std::string get_name() const
    {
        return "Fork island";
    }
    void run_evolve(island &isl) const
    {
        pipe_t p;
        auto child_pid = fork();
        if (child_pid == -1) {
            // Forking failed.
            raise_errno("fork");
        }
        if (child_pid) {
            // We are in the parent.
            // Init the message from the child.
            message_t m;
            try {
                // Close the write descriptor, we don't need to send anything to the child.
                if (close(p.write) == -1) {
                    raise_errno("close");
                }
                // Prepare a local buffer and a stringstream, then read the data from the child.
                char buffer[1024];
                std::stringstream ss;
                {
                    cereal::BinaryInputArchive iarchive(ss);
                    while (true) {
                        const auto read_bytes = read(p.read, static_cast<void *>(buffer), sizeof(buffer));
                        if (read_bytes == 0) {
                            // EOF, break out.
                            break;
                        }
                        if (read_bytes == -1) {
                            raise_errno("read");
                        }
                        ss.write(buffer, boost::numeric_cast<std::streamsize>(read_bytes));
                    }
                    iarchive(m);
                }
                // Close the read descriptor.
                if (close(p.read) == -1) {
                    raise_errno("close");
                }
            } catch (...) {
                // Something failed above. Try to kill the child
                // before re-raising the error.
                if (kill(child_pid, SIGTERM) == -1 && errno != ESRCH) {
                    // The signal delivery to the child failed, and not because
                    // the child does not exist any more (if the child did not exist,
                    // errno would be ESRCH).
                    exit_unrecoverable();
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
            isl.set_algorithm(std::get<2>(m));
            isl.set_population(std::get<3>(m));
            std::cout << "parent, everything ok!\n";
        } else {
            try {
                // We are in the child.
                // Close the read descriptor, we don't need to read anything from the parent.
                if (close(p.read) == -1) {
                    raise_errno("close");
                }
                // Run the evolution.
                auto algo = isl.get_algorithm();
                auto new_pop = algo.evolve(isl.get_population());
                // Pack in a tuple and serialize the result of the evolution.
                message_t m(0, "", std::move(algo), std::move(new_pop));
                std::stringstream ss;
                {
                    cereal::BinaryOutputArchive oarchive(ss);
                    oarchive(m);
                }
                // Now start the process of sending back the data to the parent.
                char buffer[1024];
                while (!ss.eof()) {
                    // Copy a chunk of data from the stream to the local buffer.
                    ss.read(buffer, boost::numeric_cast<std::streamsize>(sizeof(buffer)));
                    // Figure out how much we actually read.
                    const auto read_bytes = boost::numeric_cast<std::size_t>(ss.gcount());
                    assert(read_bytes <= sizeof(buffer));
                    // Now let's send the current content of the buffer to the parent.
                    if (write(p.write, static_cast<const void *>(buffer), read_bytes) == -1) {
                        raise_errno("write");
                    }
                }
                // Close the write descriptor.
                if (close(p.write) == -1) {
                    raise_errno("close");
                }
                std::cout << "child, everything ok!\n";
                // All done, we can kill the child.
                // std::exit(0);
            } catch (...) {
                // TODO
                std::exit(0);
            }
        }
    }
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

} // namespace pagmo

PAGMO_REGISTER_ISLAND(pagmo::fork_island)

#endif