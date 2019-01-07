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

#define BOOST_TEST_MODULE inventory_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(inventory_test)
{
    // We check construction, seed setter and fitness determinism
    inventory prob(5, 5, 24);
    auto f1 = prob.fitness({3, 6, 7, 2, 2});
    prob.set_seed(23u);
    auto f2 = prob.fitness({3, 6, 7, 2, 2});
    prob.set_seed(24u);
    auto f3 = prob.fitness({3, 6, 7, 2, 2});
    BOOST_CHECK(f1 == f3);
    BOOST_CHECK(f1 != f2);

    // Checking bounds
    BOOST_CHECK((prob.get_bounds()
                 == std::pair<vector_double, vector_double>({{0., 0., 0., 0., 0}, {200., 200., 200., 200., 200.}})));

    // Checking the name and extra info methods
    BOOST_CHECK(prob.get_name().find("Inventory") != std::string::npos);
    BOOST_CHECK(prob.get_extra_info().find("Weeks") != std::string::npos);
    BOOST_CHECK(prob.get_extra_info().find("Sample size") != std::string::npos);
    BOOST_CHECK(prob.get_extra_info().find("Seed") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(inventory_serialization_test)
{
    problem p{inventory{5u, 5u, 32u}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1., 1., 1., 1.});
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
