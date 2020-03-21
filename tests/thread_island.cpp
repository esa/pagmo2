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

#define BOOST_TEST_MODULE thread_island_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <initializer_list>
#include <stdexcept>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/island.hpp>
#include <pagmo/islands/thread_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// Thread-unsafe UDA.
struct tu_uda {
    population evolve(const population &pop) const
    {
        return pop;
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

// Thread unsafe UDP.
struct tu_udp {
    vector_double fitness(const vector_double &) const
    {
        return {1};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

BOOST_AUTO_TEST_CASE(thread_island_test)
{
    {
        island isl(thread_island{}, tu_uda{}, problem(), 20u);
        isl.evolve();
        BOOST_CHECK_EXCEPTION(isl.wait_check(), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(),
                                   "the 'thread_island' UDI requires an algorithm providing at least the 'basic' "
                                   "thread safety guarantee");
        });
    }
    {
        island isl(thread_island{}, algorithm(), tu_udp{}, 20u);
        isl.evolve();
        BOOST_CHECK_EXCEPTION(isl.wait_check(), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(),
                                   "the 'thread_island' UDI requires a problem providing at least the 'basic' "
                                   "thread safety guarantee");
        });
    }
}
