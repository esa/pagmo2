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

#define BOOST_TEST_MODULE mbh_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(mbh_algorithm_construction)
{
    compass_search inner_algo{100u, 0.1, 0.001, 0.7};
    {
        mbh user_algo{inner_algo, 5, 1e-3};
        BOOST_CHECK((user_algo.get_perturb() == vector_double{1e-3}));
        BOOST_CHECK(user_algo.get_verbosity() == 0u);
        BOOST_CHECK((user_algo.get_log() == mbh::log_type{}));
    }
    {
        mbh user_algo{inner_algo, 5, {1e-3, 1e-2, 1e-3, 1e-2}};
        BOOST_CHECK((user_algo.get_perturb() == vector_double{1e-3, 1e-2, 1e-3, 1e-2}));
        BOOST_CHECK(user_algo.get_verbosity() == 0u);
        BOOST_CHECK((user_algo.get_log() == mbh::log_type{}));
    }
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, -2.1}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, 3.2}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, {0.2, 0.1, 0.}}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, {0.2, 0.1, -0.12}}), std::invalid_argument);
    BOOST_CHECK_THROW((mbh{inner_algo, 5u, {0.2, 1.1, 0.12}}), std::invalid_argument);
    BOOST_CHECK_NO_THROW(mbh{});
}

BOOST_AUTO_TEST_CASE(mbh_evolve_test)
{
    // Here we only test that evolution is deterministic if the
    // seed is controlled
    {
        problem prob{hock_schittkowsky_71{}};
        prob.set_c_tol({1e-3, 1e-3});
        population pop1{prob, 5u, 23u};
        population pop2{prob, 5u, 23u};

        mbh user_algo1{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);

        mbh user_algo2{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        user_algo2.set_verbosity(1u);
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log().size() > 0u);
        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }
    // We then check that the evolve throws if called on unsuitable problems
    {
        mbh user_algo{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        BOOST_CHECK_THROW(user_algo.evolve(population{problem{zdt{}}, 15u}), std::invalid_argument);
    }
    {
        mbh user_algo{compass_search{100u, 0.1, 0.001, 0.7}, 5u, 0.1, 23u};
        BOOST_CHECK_THROW(user_algo.evolve(population{problem{inventory{}}, 15u}), std::invalid_argument);
    }
}
