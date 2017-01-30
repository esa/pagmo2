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

#define BOOST_TEST_MODULE simulated_annealing_test
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <limits> //  std::numeric_limits<double>::infinity();
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(simulated_annealing_construction)
{
    BOOST_CHECK_NO_THROW(simulated_annealing{});
    simulated_annealing user_algo{10, 0.1, 10u, 10u, 10u, 1., 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    // BOOST_CHECK((user_algo.get_log() == cmaes::log_type{}));

    BOOST_CHECK_THROW((simulated_annealing{-1., .1, 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{std::nan(""), 0.1, 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, -1., 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, std::nan(""), 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, .1, 10u, 10u, 10u, 1.1, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, .1, 10u, 10u, 10u, -1.1, 23u}), std::invalid_argument);

    // We check that the problem is checked to be suitable
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{problem{zdt{}}, 5u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{problem{inventory{}}, 5u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{problem{hock_schittkowsky_71{}}, 5u, 23u})),
                      std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{problem{rosenbrock{}}})), std::invalid_argument);
}
