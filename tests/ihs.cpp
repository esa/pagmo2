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

#define BOOST_TEST_MODULE ihs_problem_test
#include <boost/test/included/unit_test.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>

using namespace pagmo;
using namespace std;

BOOST_AUTO_TEST_CASE(ihs_algorithm_construction)
{
    {
        // Here we construct a valid ihs uda
        ihs user_algo{1u, 0.85, 0.35, 0.99, 1e-5, 1., 42u};
        BOOST_CHECK(user_algo.get_verbosity() == 0u);
        BOOST_CHECK(user_algo.get_seed() == 42u);
        BOOST_CHECK((user_algo.get_log() == ihs::log_type{}));
    }

    // Here we construct invalid ihs udas and test that construction fails
    BOOST_CHECK_THROW((ihs{1u, 1.2, 0.35, 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, -0.2, 0.35, 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 23., 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, -22.4, 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 12., 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, -0.2, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 0.34, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 0.99, -0.43, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 0.99, 0.4, 0.3, 42u}), std::invalid_argument);
}
BOOST_AUTO_TEST_CASE(ihs_evolve_test)
{
    // We test for unsuitable populations
    {
        population pop{rosenbrock{25u}};
        BOOST_CHECK_THROW(ihs{15u}.evolve(pop), std::invalid_argument);
        population pop2{null_problem{2u, 3u, 4u}};
        BOOST_CHECK_THROW(ihs{15u}.evolve(pop2), std::invalid_argument);
    }
    // And a clean exit for 0 generations
    {
        population pop1{rosenbrock{25u}, 10u};
        BOOST_CHECK(ihs{0u}.evolve(pop1).get_x()[0] == pop1.get_x()[0]);
    }
    {
        population pop{rosenbrock{10u}, 20u};
        algorithm algo(ihs{1000000u, 0.85, 0.35, 0.99, 1e-5, 1.});
        algo.set_verbosity(1000u);
        pop = algo.evolve(pop);
        print("Best: ", pop.champion_f()[0], "\n");
    }
}

BOOST_AUTO_TEST_CASE(ihs_setters_getters_test)
{
    ihs user_algo{1u, 0.85, 0.35, 0.99, 1e-5, 1., 42u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("Improved Harmony Search") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Maximum distance bandwidth") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}