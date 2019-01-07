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

#define BOOST_TEST_MODULE zdt_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(zdt_construction_test)
{
    zdt zdt_default{};
    zdt zdt5{5, 11};

    BOOST_CHECK_THROW((zdt{7, 23}), std::invalid_argument);
    BOOST_CHECK_THROW((zdt{2, 1}), std::invalid_argument);
    BOOST_CHECK_NO_THROW(problem{zdt_default});
    BOOST_CHECK_NO_THROW(problem{zdt5});
    // We also test get_nobj() here as not to add one more small test
    BOOST_CHECK(zdt_default.get_nobj() == 2u);
    // We also test get_name()
    BOOST_CHECK(zdt5.get_name().find("ZDT5") != std::string::npos);
    // And the integer dimansion
    BOOST_CHECK(problem(zdt5).get_nix() == (11u - 1u) * 5u + 30u);
    BOOST_CHECK(problem(zdt5).get_ncx() == 0u);
    BOOST_CHECK(problem(zdt_default).get_nix() == 0u);
}

BOOST_AUTO_TEST_CASE(zdt1_fitness_test)
{
    {
        zdt zdt1{1, 30};
        vector_double x(30, 0.25);
        BOOST_CHECK_CLOSE(zdt1.fitness(x)[0], 0.25, 1e-13);
        BOOST_CHECK_CLOSE(zdt1.fitness(x)[1], 2.3486121811340026, 1e-13);
    }
    zdt zdt1{1, 13};
    {
        vector_double x(13, 0.33);
        BOOST_CHECK_CLOSE(zdt1.fitness(x)[0], 0.33, 1e-13);
        BOOST_CHECK_CLOSE(zdt1.fitness(x)[1], 2.825404001404863, 1e-13);
    }
}

BOOST_AUTO_TEST_CASE(zdt2_fitness_test)
{
    {
        zdt zdt2{2, 30};
        vector_double x(30, 0.25);
        BOOST_CHECK_CLOSE(zdt2.fitness(x)[0], 0.25, 1e-13);
        BOOST_CHECK_CLOSE(zdt2.fitness(x)[1], 3.230769230769231, 1e-13);
    }
    zdt zdt2{2, 13};
    {
        vector_double x(13, 0.33);
        BOOST_CHECK_CLOSE(zdt2.fitness(x)[0], 0.33, 1e-13);
        BOOST_CHECK_CLOSE(zdt2.fitness(x)[1], 3.9425692695214107, 1e-13);
    }
}

BOOST_AUTO_TEST_CASE(zdt3_fitness_test)
{
    {
        zdt zdt3{3, 30};
        vector_double x(30, 0.25);
        BOOST_CHECK_CLOSE(zdt3.fitness(x)[0], 0.25, 1e-13);
        BOOST_CHECK_CLOSE(zdt3.fitness(x)[1], 2.0986121811340026, 1e-13);
    }
    zdt zdt3{3, 13};
    {
        vector_double x(13, 0.33);
        BOOST_CHECK_CLOSE(zdt3.fitness(x)[0], 0.33, 1e-13);
        BOOST_CHECK_CLOSE(zdt3.fitness(x)[1], 3.092379609548596, 1e-13);
    }
}

BOOST_AUTO_TEST_CASE(zdt4_fitness_test)
{
    {
        zdt zdt4{4, 30};
        vector_double x(30, 0.25);
        BOOST_CHECK_CLOSE(zdt4.fitness(x)[0], 0.25, 1e-13);
        BOOST_CHECK_CLOSE(zdt4.fitness(x)[1], 570.7417450526075, 1e-13);
    }
    zdt zdt4{4, 13};
    {
        vector_double x(13, 0.33);
        BOOST_CHECK_CLOSE(zdt4.fitness(x)[0], 0.33, 1e-13);
        BOOST_CHECK_CLOSE(zdt4.fitness(x)[1], 178.75872382132619, 1e-13);
    }
}

BOOST_AUTO_TEST_CASE(zdt5_fitness_test)
{
    {
        zdt zdt5{5, 30};
        vector_double x(175, 1.);
        std::fill(x.begin() + 100, x.end(), 0.);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[0], 31., 1e-13);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[1], 1.4193548387096775, 1e-13);
    }
    {
        zdt zdt5{5, 13};
        vector_double x(90, 1.);
        std::fill(x.begin() + 45, x.end(), 0.);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[0], 31., 1e-13);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[1], 0.6774193548387096, 1e-13);
    }
    // Test with double relaxation
    {
        zdt zdt5{5, 30};
        vector_double x(175, 1.35422);
        std::fill(x.begin() + 100, x.end(), 0.1534567);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[0], 31., 1e-13);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[1], 1.4193548387096775, 1e-13);
    }
    {
        zdt zdt5{5, 13};
        vector_double x(90, 1.34677824);
        std::fill(x.begin() + 45, x.end(), 0.345345);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[0], 31., 1e-13);
        BOOST_CHECK_CLOSE(zdt5.fitness(x)[1], 0.6774193548387096, 1e-13);
    }
}

BOOST_AUTO_TEST_CASE(zdt6_fitness_test)
{
    {
        zdt zdt6{6, 30};
        vector_double x(30, 0.25);
        BOOST_CHECK_CLOSE(zdt6.fitness(x)[0], 0.6321205588285577, 1e-13);
        BOOST_CHECK_CLOSE(zdt6.fitness(x)[1], 7.309699961231513, 1e-13);
    }
    {
        zdt zdt6{6, 13};
        vector_double x(13, 0.33);
        BOOST_CHECK_CLOSE(zdt6.fitness(x)[0], 0.999999983628226, 1e-13);
        BOOST_CHECK_CLOSE(zdt6.fitness(x)[1], 7.693505388431892, 1e-13);
    }
}

BOOST_AUTO_TEST_CASE(zdt_p_distance_test)
{
    zdt zdt1{1, 30};
    zdt zdt2{2, 30};
    zdt zdt3{3, 30};
    zdt zdt4{4, 10};
    zdt zdt5{5, 11};
    zdt zdt6{6, 10};
    vector_double x(30, 0.143);
    vector_double xi(175, 1.);
    std::fill(xi.begin() + 100, xi.end(), 0.);
    BOOST_CHECK_CLOSE(zdt1.p_distance(x), 1.2869999999999997, 1e-13);
    BOOST_CHECK_CLOSE(zdt2.p_distance(x), 1.2869999999999997, 1e-13);
    BOOST_CHECK_CLOSE(zdt3.p_distance(x), 1.2869999999999997, 1e-13);
    BOOST_CHECK_CLOSE(zdt4.p_distance(x), 355.63154167532053, 1e-13);
    BOOST_CHECK_CLOSE(zdt5.p_distance(xi), 15., 1e-13);
    BOOST_CHECK_CLOSE(zdt6.p_distance(x), 5.534476131480399, 1e-13);
}

BOOST_AUTO_TEST_CASE(zdt_get_bounds_test)
{
    zdt zdt1{1, 30};
    zdt zdt2{2, 30};
    zdt zdt3{3, 30};
    zdt zdt4{4, 10};
    zdt zdt5{5, 11};
    zdt zdt6{6, 10};
    std::pair<vector_double, vector_double> bounds123({vector_double(30, 0.), vector_double(30, 1.)});
    std::pair<vector_double, vector_double> bounds4({vector_double(10, -5.), vector_double(10, 5.)});
    std::pair<vector_double, vector_double> bounds5({vector_double(80, 0.), vector_double(80, 1.)});
    std::pair<vector_double, vector_double> bounds6({vector_double(10, 0.), vector_double(10, 1.)});
    bounds4.first[0] = 0.;
    bounds4.second[0] = 1.;

    BOOST_CHECK(zdt1.get_bounds() == bounds123);
    BOOST_CHECK(zdt2.get_bounds() == bounds123);
    BOOST_CHECK(zdt3.get_bounds() == bounds123);
    BOOST_CHECK(zdt4.get_bounds() == bounds4);
    BOOST_CHECK(zdt5.get_bounds() == bounds5);
    BOOST_CHECK(zdt6.get_bounds() == bounds6);
}

BOOST_AUTO_TEST_CASE(zdt_serialization_test)
{
    problem p{zdt{4, 4}};
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
