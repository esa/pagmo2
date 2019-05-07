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

#ifndef PAGMO_ISLANDS_THREAD_ISLAND_HPP
#define PAGMO_ISLANDS_THREAD_ISLAND_HPP

#include <string>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/island.hpp>

namespace pagmo
{

/// Thread island.
/**
 * This class is a user-defined island (UDI) that will run evolutions directly inside
 * the separate thread of execution within pagmo::island.
 *
 * thread_island is the UDI type automatically selected by the constructors of pagmo::island
 * on non-POSIX platforms or when both the island's problem and algorithm provide at least the
 * pagmo::thread_safety::basic thread safety guarantee.
 */
class PAGMO_DLL_PUBLIC thread_island
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
    // Serialization support.
    template <typename Archive>
    void serialize(Archive &, unsigned);
};

} // namespace pagmo

PAGMO_S11N_ISLAND_EXPORT_KEY(pagmo::thread_island)

#endif
