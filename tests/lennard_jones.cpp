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

#define BOOST_TEST_MODULE lennard_jones_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/lennard_jones.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(lennard_jones_test)
{
    // Problem construction
    BOOST_CHECK_THROW(lennard_jones{0u}, std::invalid_argument);
    BOOST_CHECK_THROW(lennard_jones{1u}, std::invalid_argument);
    BOOST_CHECK_THROW(lennard_jones{2u}, std::invalid_argument);
    BOOST_CHECK_THROW(lennard_jones{std::numeric_limits<unsigned>::max() / 2}, std::overflow_error);

    lennard_jones lj{3u};
    BOOST_CHECK_NO_THROW(problem{lj});
    // Pick a few reference points
    vector_double x1 = {1.12, -0.33, 2.34};
    vector_double x2 = {1.23, -1.23, 0.33};
    // Fitness test
    BOOST_CHECK_CLOSE(lj.fitness(x1)[0], -1.7633355813175688, 1e-13);
    BOOST_CHECK_CLOSE(lj.fitness(x2)[0], -1.833100934753864, 1e-13);
    // Bounds Test
    BOOST_CHECK((lj.get_bounds() == std::pair<vector_double, vector_double>{{-3, -3, -3}, {3, 3, 3}}));
    // Name and extra info tests
    BOOST_CHECK(lj.get_name().find("Jones") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(lennard_jones_serialization_test)
{
    problem p{lennard_jones{30u}};
    // Call objfun to increase the internal counters.
    p.fitness(vector_double(30u * 3 - 6u, 0.1));
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(p);
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << p;
    }
    // Change the content of p before deserializing.
    p = problem{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> p;
    }
    auto after = boost::lexical_cast<std::string>(p);
    BOOST_CHECK_EQUAL(before, after);
}
