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

#define BOOST_TEST_MODULE pso_test
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <limits> //  std::numeric_limits<double>::infinity();
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(construction)
{
    BOOST_CHECK_NO_THROW(pso{});
    pso user_algo{100, 0.79, 2., 2., 0.1, 5u, 2u, 4u, false, 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    // BOOST_CHECK((user_algo.get_log() == cmaes::log_type{}));

    BOOST_CHECK_NO_THROW((pso{100, 0.79, 2., 2., 0.1, 5u, 2u, 4u, false, 23u}));

    BOOST_CHECK_THROW((pso{100, -0.79, 2., 2., 0.1, 5u, 2u, 4u, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((pso{100, 2.3, 2., 2., 0.1, 5u, 2u, 4u, false, 23u}), std::invalid_argument);

    BOOST_CHECK_THROW((pso{100, 0.79, -1., 2., 0.1, 5u, 2u, 4u, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((pso{100, 0.79, 2., -1., 0.1, 5u, 2u, 4u, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((pso{100, 0.79, 5., 2., 0.1, 5u, 2u, 4u, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((pso{100, 0.79, 2., 5., 0.1, 5u, 2u, 4u, false, 23u}), std::invalid_argument);

    BOOST_CHECK_THROW((pso{100, 0.79, 2., 2., -2.3, 5u, 2u, 4u, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((pso{100, 0.79, 2., 2., 1.1, 5u, 2u, 4u, 23u}), std::invalid_argument);

    BOOST_CHECK_THROW((pso{100, -0.79, 2., 2., 0.1, 8u, 2u, 4u, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((pso{100, -0.79, 2., 2., 0.1, 0u, 2u, 4u, false, 23u}), std::invalid_argument);

    BOOST_CHECK_THROW((pso{100, 0.79, 2., 2., 0.1, 5u, 6u, 4u, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((pso{100, 0.79, 2., 2., 0.1, 5u, 0u, 4u, false, 23u}), std::invalid_argument);

    BOOST_CHECK_THROW((pso{100, 0.79, 2., 2., 0.1, 5u, 2u, 0u, false, 23u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(evolve_test)
{
    // We then check that the evolve throws if called on unsuitable problems
    BOOST_CHECK_THROW(pso{10u}.evolve(population{problem{rosenbrock{}}}), std::invalid_argument);
    BOOST_CHECK_THROW(pso{10u}.evolve(population{problem{zdt{}}, 15u}), std::invalid_argument);
    BOOST_CHECK_THROW(pso{10u}.evolve(population{problem{hock_schittkowsky_71{}}, 15u}), std::invalid_argument);
    BOOST_CHECK_THROW(pso{10u}.evolve(population{problem{inventory{}}, 15u}), std::invalid_argument);
    // And a clean exit for 0 generations
    population pop{rosenbrock{25u}, 10u};
    BOOST_CHECK(pso{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
}
