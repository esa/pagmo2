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

#ifndef PAGMO_ISLANDS_FORK_ISLAND_HPP
#define PAGMO_ISLANDS_FORK_ISLAND_HPP

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_FORK_ISLAND)

#include <atomic>
#include <string>

#include <unistd.h>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/island.hpp>

namespace pagmo
{

// Fork island: will offload the evolution to a child process created with the fork() system call.
class PAGMO_DLL_PUBLIC fork_island
{
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
    std::string get_extra_info() const;
    // Get the PID of the child.
    pid_t get_child_pid() const
    {
        return m_pid.load();
    }
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    mutable std::atomic<pid_t> m_pid;
};

} // namespace pagmo

PAGMO_S11N_ISLAND_EXPORT_KEY(pagmo::fork_island)

#else

#error The fork_island.hpp header was included, but the fork island is not available on the current platform

#endif

#endif
