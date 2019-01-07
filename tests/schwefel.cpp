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

#define BOOST_TEST_MODULE schwefel_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(schwefel_test)
{
    // Problem construction
    schwefel sch1{1u};
    schwefel sch3{3u};
    BOOST_CHECK_THROW(schwefel{0u}, std::invalid_argument);
    BOOST_CHECK_NO_THROW(problem{sch3});
    // Pick a few reference points
    vector_double x1 = {1.12};
    vector_double x3 = {-23.45, 12.34, 111.12};
    // Fitness test
    BOOST_CHECK_CLOSE(sch1.fitness(x1)[0], 418.0067810680098, 1e-13);
    BOOST_CHECK_CLOSE(sch1.fitness(x3)[0], 1338.0260195323838, 1e-13);
    // Bounds Test
    BOOST_CHECK((sch3.get_bounds() == std::pair<vector_double, vector_double>{{-500, -500, -500}, {500, 500, 500}}));
    // Name and extra info tests
    BOOST_CHECK(sch3.get_name().find("Schwefel") != std::string::npos);
    // Best known test
    auto x_best = sch3.best_known();
    BOOST_CHECK((x_best == vector_double{420.9687, 420.9687, 420.9687}));
}

BOOST_AUTO_TEST_CASE(schwefel_serialization_test)
{
    problem p{schwefel{4u}};
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
