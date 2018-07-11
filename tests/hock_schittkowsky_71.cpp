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

#define BOOST_TEST_MODULE hock_schittkowsky_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(hock_schittkowsky_71_test)
{
    // Problem instantiation
    problem p{hock_schittkowsky_71{}};
    // Pick a few reference points
    vector_double x1 = {1., 1., 1., 1.};
    vector_double x2 = {2., 2., 2., 2.};
    // Fitness test
    BOOST_CHECK((p.fitness(x1) == vector_double{4, -36, 24}));
    BOOST_CHECK((p.fitness(x2) == vector_double{26, -24, 9}));
    // Gradient test
    BOOST_CHECK((p.gradient(x1) == vector_double{4, 1, 2, 3, 2, 2, 2, 2, -1, -1, -1, -1}));
    BOOST_CHECK((p.gradient(x2) == vector_double{16, 4, 5, 12, 4, 4, 4, 4, -8, -8, -8, -8}));
    // Hessians test
    auto hess1 = p.hessians(x1);
    BOOST_CHECK(hess1.size() == 3);
    BOOST_CHECK((hess1[0] == vector_double{2, 1, 1, 4, 1, 1}));
    BOOST_CHECK((hess1[1] == vector_double{2, 2, 2, 2}));
    BOOST_CHECK((hess1[2] == vector_double{-1, -1, -1, -1, -1, -1}));
    // Hessians sparsity test
    auto sp = p.hessians_sparsity();
    BOOST_CHECK(sp.size() == 3);
    BOOST_CHECK((sp[0] == sparsity_pattern{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 1}, {3, 2}}));
    BOOST_CHECK((sp[1] == sparsity_pattern{{0, 0}, {1, 1}, {2, 2}, {3, 3}}));
    BOOST_CHECK((sp[2] == sparsity_pattern{{1, 0}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {3, 2}}));
    // Name and extra info tests
    BOOST_CHECK(p.get_name().find("Schittkowsky") != std::string::npos);
    BOOST_CHECK(p.get_extra_info().find("Schittkowsky") != std::string::npos);
    // Best known test
    auto x_best = p.extract<hock_schittkowsky_71>()->best_known();
    BOOST_CHECK_CLOSE(x_best[0], 1, 1e-13);
    BOOST_CHECK_CLOSE(x_best[1], 4.74299963, 1e-13);
    BOOST_CHECK_CLOSE(x_best[2], 3.82114998, 1e-13);
    BOOST_CHECK_CLOSE(x_best[3], 1.37940829, 1e-13);
}

BOOST_AUTO_TEST_CASE(hock_schittkowsky_71_serialization_test)
{
    problem p{hock_schittkowsky_71{}};
    // Call objfun, grad and hess to increase
    // the internal counters.
    p.fitness({1., 1., 1., 1.});
    p.gradient({1., 1., 1., 1.});
    p.hessians({1., 1., 1., 1.});
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
