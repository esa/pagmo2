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

#define BOOST_TEST_MODULE rosenbrock_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(rosenbrock_test)
{
    // Problem construction
    rosenbrock ros2{2u};
    rosenbrock ros5{5u};
    BOOST_CHECK_THROW(rosenbrock{0u}, std::invalid_argument);
    BOOST_CHECK_THROW(rosenbrock{1u}, std::invalid_argument);
    BOOST_CHECK_NO_THROW(problem{rosenbrock{2u}});
    // Pick a few reference points
    vector_double x2 = {1., 1.};
    vector_double x5 = {1., 1., 1., 1., 1.};
    // Fitness test
    BOOST_CHECK((ros2.fitness({1., 1.}) == vector_double{0.}));
    BOOST_CHECK((ros5.fitness({1., 1., 1., 1., 1.}) == vector_double{0.}));
    // Bounds Test
    BOOST_CHECK((ros2.get_bounds() == std::pair<vector_double, vector_double>{{-5., -5.}, {10., 10.}}));
    // Name and extra info tests
    BOOST_CHECK(ros5.get_name().find("Rosenbrock") != std::string::npos);
    // Best known test
    auto x_best = ros2.best_known();
    BOOST_CHECK((x_best == vector_double{1., 1.}));
    // Gradient test.
    auto g2 = ros2.gradient({.1, .2});
    BOOST_CHECK(std::abs(g2[0] + 9.4) < 1E-8);
    BOOST_CHECK(std::abs(g2[1] - 38.) < 1E-8);
    auto g5 = ros5.gradient({.1, .2, .3, .4, .5});
    BOOST_CHECK(std::abs(g5[0] + 9.4) < 1E-8);
    BOOST_CHECK(std::abs(g5[1] - 15.6) < 1E-8);
    BOOST_CHECK(std::abs(g5[2] - 13.4) < 1E-8);
    BOOST_CHECK(std::abs(g5[3] - 6.4) < 1E-8);
    BOOST_CHECK(std::abs(g5[4] - 68.) < 1E-8);
}

BOOST_AUTO_TEST_CASE(rosenbrock_serialization_test)
{
    problem p{rosenbrock{4u}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1., 1., 1.});
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(p);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(p);
    }
    // Change the content of p before deserializing.
    p = problem{null_problem{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(p);
    }
    auto after = boost::lexical_cast<std::string>(p);
    BOOST_CHECK_EQUAL(before, after);
}
