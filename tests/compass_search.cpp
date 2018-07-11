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

#define BOOST_TEST_MODULE compass_search_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(compass_search_algorithm_construction)
{
    compass_search user_algo{100u, 0.1, 0.001, 0.7};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK((user_algo.get_log() == compass_search::log_type{}));

    BOOST_CHECK_THROW((compass_search{1234u, 1.1}), std::invalid_argument);
    BOOST_CHECK_THROW((compass_search{1234u, -0.3}), std::invalid_argument);
    BOOST_CHECK_THROW((compass_search{1234u, 0.7, 1.1}), std::invalid_argument);
    BOOST_CHECK_THROW((compass_search{1234u, 0.7, 0.8}), std::invalid_argument);
    BOOST_CHECK_THROW((compass_search{1234u, 0.7, 0.1, 1.3}), std::invalid_argument);
    BOOST_CHECK_THROW((compass_search{1234u, 0.7, 0.1, -0.3}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(compass_search_evolve_test)
{
    double stop_range = 1e-5;

    // Here we only test that evolution is deterministic (stop criteria will be range)
    problem prob1{hock_schittkowsky_71{}};
    prob1.set_c_tol({1e-3, 1e-3});
    population pop1{prob1, 5u, 23u};

    problem prob2{hock_schittkowsky_71{}};
    prob2.set_c_tol({1e-3, 1e-3});
    population pop2{prob2, 5u, 23u};

    compass_search user_algo1{10000u, 0.5, stop_range, 0.5};
    user_algo1.set_verbosity(1u);
    pop1 = user_algo1.evolve(pop1);

    compass_search user_algo2{10000u, 0.5, stop_range, 0.5};
    user_algo2.set_verbosity(1u);
    pop2 = user_algo2.evolve(pop2);

    BOOST_CHECK(user_algo1.get_log().size() > 0u);
    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    BOOST_CHECK(static_cast<double>(std::get<2>(user_algo1.get_log()[user_algo1.get_log().size() - 1])) <= stop_range);

    // We test the max_fevals stopping criteria
    auto max_fevals = 10u;
    compass_search user_algo3{max_fevals, 0.5, stop_range, 0.5};
    user_algo3.set_verbosity(1u);
    population pop3{prob2, 5u, 23u};
    pop3 = user_algo3.evolve(pop3);
    BOOST_CHECK(std::get<0>(user_algo3.get_log()[user_algo3.get_log().size() - 1]) > max_fevals);

    // We then check that the evolve throws if called on unsuitable problems
    BOOST_CHECK_THROW(compass_search{10u}.evolve(population{problem{rosenbrock{}}, 0u}), std::invalid_argument);
    BOOST_CHECK_THROW(compass_search{10u}.evolve(population{problem{zdt{}}, 15u}), std::invalid_argument);
    BOOST_CHECK_THROW(compass_search{10u}.evolve(population{problem{inventory{}}, 15u}), std::invalid_argument);
    // And a clean exit for 0 generations
    population pop{rosenbrock{25u}, 10u};
    BOOST_CHECK(compass_search{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
}

BOOST_AUTO_TEST_CASE(compass_search_setters_getters_test)
{
    compass_search user_algo{10000u, 0.5, 0.1, 0.5};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    BOOST_CHECK(user_algo.get_max_fevals() == 10000u);
    BOOST_CHECK(user_algo.get_start_range() == 0.5);
    BOOST_CHECK(user_algo.get_stop_range() == 0.1);
    BOOST_CHECK(user_algo.get_reduction_coeff() == 0.5);
    BOOST_CHECK(user_algo.get_name().find("Compass") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Reduction coefficient") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(compass_search_serialization_test)
{
    // We test the serialization of a pagmo algorithm when constructed with compass_search
    // Make one evolution
    problem prob{rosenbrock{25u}};
    population pop{prob, 10u, 23u};
    algorithm algo{compass_search{10000u, 0.5, 0.1, 0.5}};
    algo.set_verbosity(1u); // allows the log to be filled
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<compass_search>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(algo);
    }
    // Change the content of p before deserializing.
    algo = algorithm{null_algorithm{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(algo);
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<compass_search>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<1>(before_log[i]), std::get<1>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<2>(before_log[i]), std::get<2>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<4>(before_log[i]), std::get<4>(after_log[i]), 1e-8);
    }
}
