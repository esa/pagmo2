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

#define BOOST_TEST_MODULE golomb_ruler_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/golomb_ruler.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(golomb_ruler_test)
{
    // Problem construction
    BOOST_CHECK_THROW((golomb_ruler{0u, 10u}), std::invalid_argument);
    BOOST_CHECK_THROW((golomb_ruler{10u, 0u}), std::invalid_argument);
    BOOST_CHECK_THROW((golomb_ruler{400u, std::numeric_limits<unsigned>::max() / 300u}), std::overflow_error);

    golomb_ruler gr{4u, 4u};
    BOOST_CHECK_NO_THROW(problem{gr});

    // Pick a few reference points
    vector_double x1 = {1, 3, 2}; // 0 1 4 6 -> 1,4,6,3,5,2
    vector_double x2 = {3, 4, 1}; // 0 3 7 8 -> 3,7,8,4,5,1
    vector_double x3 = {1, 3, 1}; // 0 1 4 5 -> 1,4,5,3,4,1
    // Fitness test
    BOOST_CHECK_EQUAL(gr.fitness(x1)[0], 6);
    BOOST_CHECK_EQUAL(gr.fitness(x1)[1], 0);

    BOOST_CHECK_EQUAL(gr.fitness(x2)[0], 8);
    BOOST_CHECK_EQUAL(gr.fitness(x2)[1], 0);

    BOOST_CHECK_EQUAL(gr.fitness(x3)[0], 5);
    BOOST_CHECK_EQUAL(gr.fitness(x3)[1], 2);

    // Problem dimension test
    BOOST_CHECK_EQUAL(gr.get_nix(), 3u);
    BOOST_CHECK_EQUAL(gr.get_nec(), 1u);

    // Bounds Test
    BOOST_CHECK((gr.get_bounds() == std::pair<vector_double, vector_double>{{1, 1, 1}, {4, 4, 4}}));
    // Name and extra info tests
    BOOST_CHECK(gr.get_name().find("Golomb") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(golomb_ruler_solve_order_4)
{
    // This tests always succeeds and its here only to show a solution strategy
    // that uses death penalty to deal with the constraint
    golomb_ruler udp{4u, 10u};
    problem prob{unconstrain{udp}};
    print(prob);
    algorithm algo{sga{1000, 0.2, 1., 0.5}};
    algo.set_verbosity(1);
    population pop{prob, 10};
    pop = algo.evolve(pop);
    print(pop.champion_x(), " ", pop.champion_f());
}

BOOST_AUTO_TEST_CASE(golomb_ruler_serialization_test)
{
    problem p{golomb_ruler{30u}};
    // Call objfun to increase the internal counters.
    p.fitness(vector_double(29u, 1));
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
