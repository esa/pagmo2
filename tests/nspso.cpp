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

#define BOOST_TEST_MODULE nspso_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nspso.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nspso_algorithm_construction)
{
    nspso user_algo{1u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u};
    BOOST_CHECK_NO_THROW(nspso{});
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 24u);
    // Check the throws
    // Wrong max_w and min_w
    BOOST_CHECK_THROW((nspso{1u, 0.95, -10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 0.94, 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, -0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    // Wrong c1, c2 and chi
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., -0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, -0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, -0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    // Wrong v_coeff
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, -0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, 1.5, 2u, "crowding distance", 24u}), std::invalid_argument);
    // Wrong leader_selection_range
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 101u, "crowding distance", 24u}),
                      std::invalid_argument);
    // Wrong eta_m
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "something else", 24u}), std::invalid_argument);
}
