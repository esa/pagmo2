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

#define BOOST_TEST_MODULE generic_utilities_test
#include <boost/test/included/unit_test.hpp>

#include <limits>
#include <stdexcept>
#include <tuple>

#include <pagmo/io.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(uniform_real_from_range_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    auto nan = std::numeric_limits<double>::quiet_NaN();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_THROW(uniform_real_from_range(1, 0, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-big, big, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-3, inf, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-nan, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(nan, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-nan, 3, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-3, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(inf, inf, r_engine), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(random_decision_vector_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    auto nan = std::numeric_limits<double>::quiet_NaN();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_THROW(random_decision_vector({{1, 2}, {0, 3}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{1, -big}, {2, big}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{1, -inf}, {2, 32}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{1, 2, 3}, {2, 3}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, 2, 3}, {1, 4, nan}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, 2, nan}, {1, 4, 4}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, nan, 3}, {1, nan, 4}}, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, 2, 3}, {1, 4, 5}}, r_engine, 4u), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, 2, 3.1}, {1, 4, 5}}, r_engine, 1u), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, 2, 3}, {1, 4, 5.2}}, r_engine, 1u), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, -1.1, 3}, {1, 2, 5}}, r_engine, 2u), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, -1.1, big}, {1, 2, big}}, r_engine, 2u), std::invalid_argument);
    BOOST_CHECK_NO_THROW(random_decision_vector({{0, -1.1, big}, {1, 2, big}}, r_engine));
    BOOST_CHECK_THROW(random_decision_vector({{0, -1.1, -inf}, {1, 2, inf}}, r_engine, 2u), std::invalid_argument);
    BOOST_CHECK_THROW(random_decision_vector({{0, -1.1, inf}, {1, 2, inf}}, r_engine, 2u), std::invalid_argument);

    // Test the results
    BOOST_CHECK((random_decision_vector({{3, 4}, {3, 4}}, r_engine) == vector_double{3, 4}));
    BOOST_CHECK(random_decision_vector({{0, 0}, {1, 1}}, r_engine)[0] >= 0);
    BOOST_CHECK(random_decision_vector({{0, 0}, {1, 1}}, r_engine)[1] < 1);
    BOOST_CHECK(random_decision_vector({{0, 0}, {1, 0}}, r_engine, 1u)[1] == 0);
    for (auto i = 0u; i < 100; ++i) {
        auto res = random_decision_vector({{0, -20}, {1, 20}}, r_engine, 1u);
        BOOST_CHECK(res[1] == std::floor(res[1]));
    }

    // Test the overload
    BOOST_CHECK((random_decision_vector({3, 4}, {3, 4}, r_engine) == vector_double{3, 4}));
    BOOST_CHECK(random_decision_vector({0, 0}, {1, 1}, r_engine)[0] >= 0);
    BOOST_CHECK(random_decision_vector({0, 0}, {1, 1}, r_engine)[1] < 1);
    BOOST_CHECK(random_decision_vector({0, 0}, {1, 0}, r_engine, 1u)[1] == 0);

    for (auto i = 0u; i < 100; ++i) {
        auto res = random_decision_vector({0, -20}, {1, 20}, r_engine, 1u);
        BOOST_CHECK(res[1] == std::floor(res[1]));
    }
}

BOOST_AUTO_TEST_CASE(force_bounds_test)
{
    detail::random_engine_type r_engine(32u);
    // force_bounds_random
    {
        vector_double x{1., 2., 3.};
        vector_double x_fix = x;
        detail::force_bounds_random(x_fix, {0., 0., 0.}, {3., 3., 3.}, r_engine);
        BOOST_CHECK(x == x_fix);
        detail::force_bounds_random(x_fix, {0., 0., 0.}, {1., 1., 1.}, r_engine);
        BOOST_CHECK(x != x_fix);
        BOOST_CHECK_EQUAL(x_fix[0], 1.);
        BOOST_CHECK(x_fix[1] <= 1. && x_fix[1] >= 0.);
        BOOST_CHECK(x_fix[2] <= 1. && x_fix[2] >= 0.);
    }
    // force_bounds_reflection
    {
        vector_double x{1., 2., 5.};
        vector_double x_fix = x;
        detail::force_bounds_reflection(x_fix, {0., 0., 0.}, {3., 3., 5.});
        BOOST_CHECK(x == x_fix);
        detail::force_bounds_reflection(x_fix, {0., 0., 0.}, {1., 1.9, 2.1});
        BOOST_CHECK(x != x_fix);
        BOOST_CHECK_EQUAL(x_fix[0], 1.);
        BOOST_CHECK_CLOSE(x_fix[1], 1.8, 1e-8);
        BOOST_CHECK_CLOSE(x_fix[2], 0.8, 1e-8);
    }
    // force_bounds_stick
    {
        vector_double x{1., 2., 5.};
        vector_double x_fix = x;
        detail::force_bounds_stick(x_fix, {0., 0., 0.}, {3., 3., 5.});
        BOOST_CHECK(x == x_fix);
        // ub
        detail::force_bounds_stick(x_fix, {0., 0., 0.}, {1., 1.9, 2.1});
        BOOST_CHECK(x != x_fix);
        BOOST_CHECK_EQUAL(x_fix[0], 1.);
        BOOST_CHECK_EQUAL(x_fix[1], 1.9);
        BOOST_CHECK_EQUAL(x_fix[2], 2.1);
        // lb
        detail::force_bounds_stick(x_fix, {2., 2., 2.}, {3., 3., 3.});
        BOOST_CHECK_EQUAL(x_fix[0], 2.);
        BOOST_CHECK_EQUAL(x_fix[1], 2.);
        BOOST_CHECK_EQUAL(x_fix[2], 2.1);
    }
}

BOOST_AUTO_TEST_CASE(binomial_coefficient_test)
{
    BOOST_CHECK_EQUAL(binomial_coefficient(0u, 0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(1u, 0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(1u, 1u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u, 0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u, 1u), 2u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u, 2u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(13u, 5u), 1287u);
    BOOST_CHECK_EQUAL(binomial_coefficient(21u, 10u), 352716u);
    BOOST_CHECK_THROW(binomial_coefficient(10u, 21u), std::invalid_argument);
    BOOST_CHECK_THROW(binomial_coefficient(0u, 1u), std::invalid_argument);
    BOOST_CHECK_THROW(binomial_coefficient(4u, 7u), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(kNN_test)
{
    // Corner cases
    {
        std::vector<vector_double> points = {};
        std::vector<std::vector<vector_double::size_type>> res = {};
        BOOST_CHECK(kNN(points, 2u) == res);
    }
    {
        std::vector<vector_double> points = {{1.2, 1.2}};
        std::vector<std::vector<vector_double::size_type>> res = {{}};
        BOOST_CHECK(kNN(points, 2u) == res);
    }
    {
        std::vector<vector_double> points = {{1.2, 1.2}, {1.2, 1.2}};
        std::vector<std::vector<vector_double::size_type>> res = {{1u}, {0u}};
        BOOST_CHECK(kNN(points, 2u) == res);
    }
    {
        std::vector<vector_double> points = {{1, 1}, {2, 2}, {3.1, 3.1}};
        std::vector<std::vector<vector_double::size_type>> res = {{1u, 2u}, {0u, 2u}, {1u, 0u}};
        BOOST_CHECK(kNN(points, 2u) == res);
    }

    // Some test cases
    {
        std::vector<vector_double> points = {{1, 1}, {2, 2}, {3.1, 3.1}, {4.2, 4.2}, {5.4, 5.4}};
        std::vector<std::vector<vector_double::size_type>> res
            = {{1u, 2u, 3u}, {0u, 2u, 3u}, {1u, 3u, 0u}, {2u, 4u, 1u}, {3u, 2u, 1u}};
        BOOST_CHECK(kNN(points, 3u) == res);
    }
    // throws
    {
        std::vector<vector_double> points = {{1, 1}, {2, 2}, {2, 3, 4}};
        BOOST_CHECK_THROW(kNN(points, 3u), std::invalid_argument);
    }
}
