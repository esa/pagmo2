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

#define BOOST_TEST_MODULE constrained_utilities_test
#include <boost/test/included/unit_test.hpp>

#include <stdexcept>

#include <pagmo/io.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(compare_fc_test)
{
    vector_double f1 = {2., 1.};
    vector_double f2 = {1., 2.};
    vector_double f3 = {1., -2.};
    vector_double f4 = {1., -3., -5.};
    vector_double f5 = {0.2, 1., 2.};
    vector_double tol = {0., 0.};
    vector_double empty = {};
    BOOST_CHECK(compare_fc(f1, f2, 1u, 0.) == true);
    BOOST_CHECK(compare_fc(f2, f1, 1u, 0.) == false);
    BOOST_CHECK(compare_fc(f1, f3, 0u, 0.) == false);
    BOOST_CHECK(compare_fc(f4, f5, 2u, 0.) == false);
    BOOST_CHECK(compare_fc(f4, f5, 1u, 0.) == true);
    BOOST_CHECK(compare_fc(f4, f5, 2u, tol) == false);
    BOOST_CHECK(compare_fc(f4, f5, 1u, tol) == true);

    BOOST_CHECK_THROW(compare_fc(f1, f5, 1u, 0.), std::invalid_argument);
    BOOST_CHECK_THROW(compare_fc(f1, f2, 3u, 0.), std::invalid_argument);
    BOOST_CHECK_THROW(compare_fc(f1, f2, 1u, tol), std::invalid_argument);
    BOOST_CHECK_THROW(compare_fc(empty, empty, 1u, 0.), std::invalid_argument);
    BOOST_CHECK_THROW(compare_fc(empty, empty, 1u, tol), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(sort_population_con_test)
{
    std::vector<vector_double> example;
    vector_double::size_type neq;
    vector_double tol;
    std::vector<vector_double::size_type> result;
    // Test 1 - check on known cases
    example = {{0, 0, 0}, {1, 1, 0}, {2, 0, 0}};
    neq = 1;
    result = {0, 2, 1};
    tol = {0., 0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
    neq = 1;
    result = {0, 1, 2};
    tol = {0., 0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1, 0, -20}, {0, 0, -1}, {1, 0, -2}};
    neq = 1;
    result = {0, 1, 2};
    tol = {0., 0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1, 0, -20}, {0, 0, -1}, {1, 0, -2}};
    neq = 2;
    result = {1, 2, 0};
    tol = {0., 0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1, 0, 0}, {0, 0, -1}, {1, 0, 0}};
    neq = 2;
    result = {0, 1, 2};
    tol = {0., 1.};
    BOOST_CHECK(sort_population_con(example, neq) != result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) != result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{-1, 0, -20}, {0, 0, -1}, {1, 0, -2}};
    neq = 0;
    result = {0, 1, 2};
    tol = {0., 0.};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, 0.) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    example = {{1}, {0}, {2}, {3}};
    neq = 0;
    result = {1, 0, 2, 3};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    // Test corner cases
    example = {};
    neq = 0;
    result = {};
    tol = {2., 3., 4.};
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    BOOST_CHECK(sort_population_con(example, neq) == result);
    example = {{1}};
    neq = 0;
    result = {0};
    BOOST_CHECK(sort_population_con(example, neq) == result);
    BOOST_CHECK(sort_population_con(example, neq, tol) == result);
    // Test throws
    example = {{1, 2, 3}, {1, 2}};
    BOOST_CHECK_THROW(sort_population_con(example, neq), std::invalid_argument);
    example = {{-1, 0, 0}, {0, 0, -1}, {1, 0, 0}};
    BOOST_CHECK_THROW(sort_population_con(example, 3), std::invalid_argument);
    BOOST_CHECK_THROW(sort_population_con(example, 4), std::invalid_argument);
    tol = {2, 3, 4};
    BOOST_CHECK_THROW(sort_population_con(example, 0, tol), std::invalid_argument);
    tol = {2};
    BOOST_CHECK_THROW(sort_population_con(example, 0, tol), std::invalid_argument);
    example = {{}, {}};
    BOOST_CHECK_THROW(sort_population_con(example, 0), std::invalid_argument);
    BOOST_CHECK_THROW(sort_population_con(example, 0, tol), std::invalid_argument);
}
