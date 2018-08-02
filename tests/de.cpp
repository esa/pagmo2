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

#define BOOST_TEST_MODULE de_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(de_algorithm_construction)
{
    de user_algo{1234u, 0.7, 0.5, 2u, 1e-6, 1e-6, 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK((user_algo.get_log() == de::log_type{}));

    BOOST_CHECK_THROW((de{1234u, 1.2, 0.5, 0u, 1e-6, 1e-6, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((de{1234u, 1.2, 0.5, 10u, 1e-6, 1e-6, 23u}), std::invalid_argument);

    BOOST_CHECK_THROW((de{1234u, 1.2, 0.5, 2u, 1e-6, 1e-6, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((de{1234u, -0.7, 0.5, 2u, 1e-6, 1e-6, 23u}), std::invalid_argument);

    BOOST_CHECK_THROW((de{1234u, 0.7, 1.5, 2u, 1e-6, 1e-6, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((de{1234u, 0.7, -0.5, 2u, 1e-6, 1e-6, 23u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(de_evolve_test)
{
    // Here we only test that evolution is deterministic if the
    // seed is controlled for all variants
    {
        problem prob{rosenbrock{25u}};
        population pop1{prob, 5u, 23u};
        population pop2{prob, 5u, 23u};
        population pop3{prob, 5u, 23u};

        for (unsigned int i = 1u; i <= 10u; ++i) {
            de user_algo1{10u, 0.7, 0.5, i, 1e-6, 1e-6, 23u};
            user_algo1.set_verbosity(1u);
            pop1 = user_algo1.evolve(pop1);

            BOOST_CHECK(user_algo1.get_log().size() > 0u);

            de user_algo2{10u, 0.7, 0.5, i, 1e-6, 1e-6, 23u};
            user_algo2.set_verbosity(1u);
            pop2 = user_algo2.evolve(pop2);

            BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

            user_algo2.set_seed(23u);
            pop3 = user_algo2.evolve(pop3);

            BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
        }
    }
    // Here we check that the exit condition of ftol and xtol actually provoke an exit within 5000 gen (rosenbrock{2} is
    // used)
    {
        de user_algo{1000000u, 0.7, 0.5, 2u, 1e-3, 1e-50, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 20u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 5000u);
    }
    {
        de user_algo{1000000u, 0.7, 0.5, 2u, 1e-50, 1e-3, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 20u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 5000u);
    }

    // We then check that the evolve throws if called on unsuitable problems
    BOOST_CHECK_THROW(de{10u}.evolve(population{problem{rosenbrock{}}, 4u}), std::invalid_argument);
    BOOST_CHECK_THROW(de{10u}.evolve(population{problem{zdt{}}, 15u}), std::invalid_argument);
    BOOST_CHECK_THROW(de{10u}.evolve(population{problem{hock_schittkowsky_71{}}, 15u}), std::invalid_argument);
    BOOST_CHECK_THROW(de{10u}.evolve(population{problem{inventory{}}, 15u}), std::invalid_argument);
    // And a clean exit for 0 generations
    population pop{rosenbrock{25u}, 10u};
    BOOST_CHECK(de{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
}

BOOST_AUTO_TEST_CASE(de_setters_getters_test)
{
    de user_algo{10u, 0.7, 0.5, 2u, 1e-6, 1e-6, 23u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("Differential") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Parameter F") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(de_serialization_test)
{
    // Make one evolution
    problem prob{rosenbrock{25u}};
    population pop{prob, 10u, 23u};
    algorithm algo{de{10u, 0.7, 0.5, 2u, 1e-6, 1e-6, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<de>()->get_log();
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
    auto after_log = algo.extract<de>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<4>(before_log[i]), std::get<4>(after_log[i]), 1e-8);
    }
}
