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

#define BOOST_TEST_MODULE simulated_annealing_test
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <limits> //  std::numeric_limits<double>::infinity();
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(simulated_annealing_construction)
{
    BOOST_CHECK_NO_THROW(simulated_annealing{});
    simulated_annealing user_algo{10, 0.1, 10u, 10u, 10u, 1., 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK((user_algo.get_log() == simulated_annealing::log_type{}));

    BOOST_CHECK_THROW((simulated_annealing{-1., .1, 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{std::nan(""), 0.1, 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, -1., 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, std::nan(""), 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, .1, 10u, 10u, 10u, 1.1, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, .1, 10u, 10u, 10u, -1.1, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, .1, 0u, 10u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{10, .1, 10u, 0u, 10u, 1., 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{.1, 10, 10u, 10u, 10u, 1., 23u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(simulated_annealing_evolve_test)
{
    // We check that evolution is deterministic if the
    // seed is controlled
    problem prob{rosenbrock{10u}};
    population pop1{prob, 5u, 23u};
    simulated_annealing user_algo1{10., 1e-5, 100u, 10u, 10u, 1., 23u};
    user_algo1.set_verbosity(200u);
    pop1 = user_algo1.evolve(pop1);

    population pop2{prob, 5u, 23u};
    simulated_annealing user_algo2{10., 1e-5, 100u, 10u, 10u, 1., 23u};
    user_algo2.set_verbosity(200u);
    pop2 = user_algo2.evolve(pop2);
    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    population pop3{prob, 5u, 23u};
    user_algo2.set_seed(23u);
    pop3 = user_algo2.evolve(pop3);
    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    // We check that the problem is checked to be suitable
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{zdt{}, 5u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{inventory{}, 5u, 23u})), std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{hock_schittkowsky_71{}, 5u, 23u})),
                      std::invalid_argument);
    BOOST_CHECK_THROW((simulated_annealing{}.evolve(population{rosenbrock{}})), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(sea_setters_getters_test)
{
    simulated_annealing user_algo{10., 1e-5, 100u, 10u, 10u, 1., 123u};
    user_algo.set_verbosity(200u);
    BOOST_CHECK(user_algo.get_verbosity() == 200u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("Simulated Annealing (Corana's)") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Verbosity") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(simulated_annealing_serialization_test)
{
    // Make one evolution
    problem prob{rosenbrock{25u}};
    population pop{prob, 5u, 23u};
    algorithm algo{simulated_annealing{10., 1e-5, 100u, 10u, 10u, 1., 23u}};
    algo.set_verbosity(200u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<simulated_annealing>()->get_log();
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
    auto after_log = algo.extract<simulated_annealing>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<1>(before_log[i]), std::get<1>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<4>(before_log[i]), std::get<4>(after_log[i]), 1e-8);
    }
}
