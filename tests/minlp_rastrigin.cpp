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

#define BOOST_TEST_MODULE pagmo_minlp_rastrigin_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/detail/constants.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rastrigin.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(min_lp_rastrigin_test)
{
    // Problem construction
    BOOST_CHECK_THROW((minlp_rastrigin{0u, 0u}), std::invalid_argument);
    BOOST_CHECK_NO_THROW((problem{minlp_rastrigin{1u, 1u}}));
    BOOST_CHECK_NO_THROW((problem{minlp_rastrigin{0u, 1u}}));
    BOOST_CHECK_NO_THROW((problem{minlp_rastrigin{1u, 0u}}));
    BOOST_CHECK_NO_THROW((problem{minlp_rastrigin{2u, 3u}}));

    // Fitness test
    detail::random_engine_type r_engine(pagmo::random_device::next());
    for (auto i = 0u; i < 100; ++i) {
        auto x = random_decision_vector({-5.12, -10}, {5.12, -5}, r_engine, 0u);
        BOOST_CHECK((minlp_rastrigin{2u, 0u}.fitness(x)) == rastrigin{2u}.fitness(x));
        BOOST_CHECK((minlp_rastrigin{2u, 0u}.gradient(x)) == rastrigin{2u}.gradient(x));
        BOOST_CHECK((minlp_rastrigin{2u, 0u}.hessians(x)) == rastrigin{2u}.hessians(x));
        x = random_decision_vector({-5.12, -10}, {5.12, -5}, r_engine, 1u);
        BOOST_CHECK((minlp_rastrigin{1u, 1u}.fitness(x)) == rastrigin{2u}.fitness(x));
        BOOST_CHECK((minlp_rastrigin{1u, 1u}.gradient(x)) == rastrigin{2u}.gradient(x));
        // BOOST_CHECK((minlp_rastrigin{1u, 1u}.hessians(x)) == rastrigin{2u}.hessians(x));
        x = random_decision_vector({-5, -10}, {-4, -5}, r_engine, 2u);
        BOOST_CHECK((minlp_rastrigin{0u, 2u}.fitness(x)) == rastrigin{2u}.fitness(x));
        BOOST_CHECK((minlp_rastrigin{0u, 2u}.gradient(x)) == rastrigin{2u}.gradient(x));
        BOOST_CHECK((minlp_rastrigin{0u, 2u}.hessians(x)) == rastrigin{2u}.hessians(x));
    }

    // Bounds Test
    BOOST_CHECK((minlp_rastrigin{1u, 0u}.get_bounds() == std::pair<vector_double, vector_double>{{-5.12}, {5.12}}));
    BOOST_CHECK((minlp_rastrigin{0u, 1u}.get_bounds() == std::pair<vector_double, vector_double>{{-10}, {-5}}));
    BOOST_CHECK(
        (minlp_rastrigin{1u, 1u}.get_bounds() == std::pair<vector_double, vector_double>{{-5.12, -10}, {5.12, -5}}));

    // Name and extra info tests
    BOOST_CHECK((minlp_rastrigin{0u, 1u}.get_name().find("MINLP Rastrigin Function") != std::string::npos));
    BOOST_CHECK(
        (problem{minlp_rastrigin{1u, 1u}}.get_extra_info().find("MINLP continuous dimension") != std::string::npos));
}

BOOST_AUTO_TEST_CASE(rastrigin_serialization_test)
{
    problem p{minlp_rastrigin{2u, 2u}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1., 1, 1});
    p.gradient({1., 1., 1, 1});
    p.hessians({1., 1., 1, 1});
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
