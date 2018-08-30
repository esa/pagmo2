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

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_FORK_ISLAND)

#include <pagmo/island.hpp>

#else

#error The fork_island.hpp header was included, but the fork island is not available on the current platform (this might mean that either the platform is not POSIX-compliant, or that the definition _POSIX_C_SOURCE is not active)

#endif

#endif
