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

#define BOOST_TEST_MODULE fork_island_test
#include <boost/test/included/unit_test.hpp>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/fork_island.hpp>
#include <pagmo/problems/rosenbrock.hpp>

#include <chrono>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(fork_island_test)
{
    island fi_0(fork_island{}, de{10000000}, rosenbrock{100}, 20);
    fi_0.evolve();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << fi_0 << '\n';
    fi_0.wait_check();
}
