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

#define BOOST_TEST_MODULE sga_problem_test
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(sga_algorithm_construction)
{
    // sga uda{1u, 0.5, 10., .02, .5, 1u, 2u, "gaussian", "truncated", "sbx", 0u};
    // We check the default constructor, a correct call and the possibility to build a pagmo::algorithm
    BOOST_CHECK_NO_THROW(sga{});
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "gaussian", "tournament", "exponential", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "uniform", "tournament", "exponential", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, 20., 1u, 5u, "polynomial", "tournament", "exponential", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "gaussian", "truncated", "exponential", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "gaussian", "tournament", "binomial", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "gaussian", "tournament", "sbx", 0u, 32u}));
    BOOST_CHECK_NO_THROW(algorithm(sga{}));
    // We check incorrect calls to the constructor
    BOOST_CHECK_THROW((sga{1u, 12., 10., .02, .5, 1u, 5u, "gaussian", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, -1.1, 10., .02, .5, 1u, 5u, "gaussian", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 0.1, .02, .5, 1u, 5u, "gaussian", "truncated", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 101., .02, .5, 1u, 5u, "gaussian", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., -0.2, .5, 1u, 5u, "gaussian", "truncated", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., 1.3, .5, 1u, 5u, "gaussian", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "unknown_method", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "gaussian", "unknown_method", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "gaussian", "truncated", "unknown_method", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 1u, 5u, "polynomial", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, 101, 1u, 5u, "polynomial", "truncated", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, -3, 1u, 5u, "uniform", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, 1.1, 1u, 5u, "uniform", "tournament", "exponential", 0u, 32u}),
                      std::invalid_argument);
}
