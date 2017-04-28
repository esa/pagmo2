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
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(sga_algorithm_construction)
{
    // sga uda{1u, 0.5, 10., .02, .5, 1u, 2u, "gaussian", "truncated", "sbx", 0u};
    // We check the default constructor, a correct call and the possibility to build a pagmo::algorithm
    BOOST_CHECK_NO_THROW(sga{});
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "gaussian", "tournament", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "uniform", "tournament", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, 20., 5u, "exponential", "polynomial", "tournament", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "gaussian", "truncated", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "binomial", "gaussian", "tournament", 0u, 32u}));
    BOOST_CHECK_NO_THROW((sga{1u, .95, 10., .02, .5, 5u, "sbx", "gaussian", "tournament", 0u, 32u}));
    BOOST_CHECK_NO_THROW(algorithm(sga{}));
    // We check incorrect calls to the constructor
    BOOST_CHECK_THROW((sga{1u, 12., 10., .02, .5, 5u, "exponential", "gaussian", "tournament", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, -1.1, 10., .02, .5, 5u, "exponential", "gaussian", "tournament", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 0.1, .02, .5, 5u, "exponential", "gaussian", "truncated", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 101., .02, .5, 5u, "exponential", "gaussian", "tournament", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., -0.2, .5, 5u, "exponential", "gaussian", "truncated", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., 1.3, .5, 5u, "exponential", "gaussian", "tournament", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "unknown_method", "tournament", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "gaussian", "unknown_method", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "unknown_method", "gaussian", "truncated", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, .5, 5u, "exponential", "polynomial", "tournament", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, 101, 5u, "exponential", "polynomial", "truncated", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, -3, 5u, "exponential", "uniform", "tournament", 0u, 32u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((sga{1u, .95, 10., .02, 1.1, 5u, "exponential", "uniform", "tournament", 0u, 32u}),
                      std::invalid_argument);
    sga uda{1000u, .90, 1., 1. / 20., 1., 4u, "sbx", "polynomial", "tournament", 0u};
    // sea uda{10000u};
    uda.set_verbosity(1u);
    schwefel udp{20u};
    population pop{udp, 20u};
    print(pop.champion_f(), "\n");
    pop = uda.evolve(pop);
    print(pop.champion_f(), "\n");
}
