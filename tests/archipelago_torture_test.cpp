/* Copyright 2017-2020 PaGMO development team

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

#define BOOST_TEST_MODULE archipelago_torture_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <sstream>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/rosenbrock.hpp>

using namespace pagmo;

// A small test for poking at an archipelago while it is evolving.
BOOST_AUTO_TEST_CASE(archipelago_torture_00)
{
    archipelago archi{10, de{50}, rosenbrock{100}, 100u};
    for (auto i = 0; i < 50; ++i) {
        archi.evolve();
    }
    auto nadd = 0;
    while (archi.status() == evolve_status::busy) {
        if (nadd < 50) {
            archi.push_back(de{50}, rosenbrock{100}, 100u);
            ++nadd;
        }
        for (auto &isl : archi) {
            auto algo = isl.get_algorithm();
            auto pop = isl.get_population();
            isl.set_algorithm(algo);
            isl.set_population(pop);
            auto isl_copy(isl);
            auto name = isl.get_name();
            auto einfo = isl.get_extra_info();
        }
        auto s = archi.size();
        (void)s;
        auto cf = archi.get_champions_f();
        auto cx = archi.get_champions_x();
        std::ostringstream oss;
        oss << archi;
        BOOST_CHECK(!oss.str().empty());
    }
    BOOST_CHECK_NO_THROW(archi.wait_check());
}
