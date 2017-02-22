/* Copyright 2017 PaGMO development team

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

#define BOOST_TEST_MODULE nsga2_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nsga2_algorithm_construction)
{
    nsga2 user_algo{1u, 0.95, 10., 0.01, 50., 0u, 32u};
    BOOST_CHECK_NO_THROW(nsga2{});
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 32u);
    // BOOST_CHECK((user_algo.get_log() == moead::log_type{}));

    // Check the throws
    // Wrong cr
    BOOST_CHECK_THROW((nsga2{1u, 1., 10., 0.01, 50., 0u, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, -1., 10., 0.01, 50., 0u, 32u}), std::invalid_argument);
    // Wrong m
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., 1.1, 50., 0u, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., -1.1, 50., 0u, 32u}), std::invalid_argument);
    // Wrong eta_m
    BOOST_CHECK_THROW((nsga2{1u, .95, 100., 0.01, 50., 0u, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, .95, .98, 0.01, 50., 0u, 32u}), std::invalid_argument);
    // Wrong eta_m
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., 0.01, 100, 0u, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga2{1u, .95, 10., 0.01, .98, 0u, 32u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga2_evolve_test)
{
    // We check that the problem is checked to be suitable
    // empty pop
    BOOST_CHECK_THROW((nsga2{}.evolve(population{zdt{}})), std::invalid_argument);
    // stochastic
    BOOST_CHECK_THROW((nsga2{}.evolve(population{inventory{}, 5u, 23u})), std::invalid_argument);
    // constrained prob
    BOOST_CHECK_THROW((nsga2{}.evolve(population{hock_schittkowsky_71{}, 5u, 23u})), std::invalid_argument);
    // single objective prob
    BOOST_CHECK_THROW((nsga2{}.evolve(population{rosenbrock{}, 5u, 23u})), std::invalid_argument);
    // wrong integer dimension
    BOOST_CHECK_THROW((nsga2{1u, 0.95, 10., 0.01, 50., 100u, 32u}.evolve(population{zdt{}, 10u, 23u})),
                      std::invalid_argument);

    population pop{zdt{1u}, 100u};
    algorithm algo{nsga2{100u}};
    pop = algo.evolve(pop);
}

BOOST_AUTO_TEST_CASE(nsga2_setters_getters_test)
{
    nsga2 user_algo{1u, 0.95, 10., 0.01, 50., 0u, 32u};
    user_algo.set_verbosity(200u);
    BOOST_CHECK(user_algo.get_verbosity() == 200u);
    user_algo.set_seed(23456u);
    BOOST_CHECK(user_algo.get_seed() == 23456u);
    BOOST_CHECK(user_algo.get_name().find("NSGA-II") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Verbosity") != std::string::npos);
    // BOOST_CHECK_NO_THROW(user_algo.get_log());
}
