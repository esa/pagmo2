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

#define BOOST_TEST_MODULE nspso_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nspso.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/wfg.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nspso_algorithm_construction)
{
    nspso user_algo{1u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u};
    BOOST_CHECK_NO_THROW(nspso{});
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 24u);
    // Check the throws
    // Wrong max_w and min_w
    BOOST_CHECK_THROW((nspso{1u, 0.95, -10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 0.94, 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, -0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    // Wrong c1, c2 and chi
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., -0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, -0.5, 0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, -0.5, 0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    // Wrong v_coeff
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, -0.5, 2u, "crowding distance", 24u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, 1.5, 2u, "crowding distance", 24u}), std::invalid_argument);
    // Wrong leader_selection_range
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 101u, "crowding distance", 24u}),
                      std::invalid_argument);
    // Wrong eta_m
    BOOST_CHECK_THROW((nspso{1u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "something else", 24u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nspso_evolve_test)
{
    // We check that the problem is checked to be suitable
    // stochastic
    BOOST_CHECK_THROW((nspso{}.evolve(population{inventory{}, 5u, 23u})), std::invalid_argument);
    // constrained prob
    BOOST_CHECK_THROW((nspso{}.evolve(population{hock_schittkowsky_71{}, 5u, 23u})), std::invalid_argument);
    // single objective prob
    BOOST_CHECK_THROW((nspso{}.evolve(population{rosenbrock{}, 5u, 23u})), std::invalid_argument);
    // and a clean exit for 0 generation
    population pop{zdt{2u}, 10u};
    BOOST_CHECK(nspso{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
    // We check for deterministic behaviour if the seed is controlled
    // we treat the last three components of the decision vector as integers
    // to trigger all cases
    dtlz udp{1u, 10u, 3u};
    population pop1{udp, 50u, 23u};
    population pop2{udp, 50u, 23u};
    population pop3{udp, 50u, 23u};

    nspso user_algo1{10u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u};
    user_algo1.set_verbosity(1u);
    pop1 = user_algo1.evolve(pop1);

    BOOST_CHECK(user_algo1.get_log().size() > 0u);

    nspso user_algo2{10u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u};
    user_algo2.set_verbosity(1u);
    pop2 = user_algo2.evolve(pop2);
    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    user_algo2.set_seed(24u);
    pop3 = user_algo2.evolve(pop3);

    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    // We evolve for many-objectives
    wfg udp_2{4u, 16u, 15u, 14u};
    population pop4{udp_2, 52u, 23u};
    pop4 = user_algo2.evolve(pop4);
}

BOOST_AUTO_TEST_CASE(nspso_setters_getters_test)
{
    nspso user_algo{10u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "crowding distance", 24u};
    user_algo.set_verbosity(200u);
    BOOST_CHECK(user_algo.get_verbosity() == 200u);
    user_algo.set_seed(23456u);
    BOOST_CHECK(user_algo.get_seed() == 23456u);
    BOOST_CHECK(user_algo.get_name().find("NSPSO") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Verbosity") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(nspso_zdt5_test)
{
    algorithm algo{nspso(10u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "max min", 24u)};
    algo.set_verbosity(1u);
    algo.set_seed(23456u);
    population pop{zdt(5u, 10u), 20u, 24u};
    pop = algo.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        auto x = pop.get_x()[i];
        std::all_of(x.begin(), x.end(), [](double el) { return (el == std::floor(el)); });
    }
}

BOOST_AUTO_TEST_CASE(nspso_serialization_test)
{
    // Make one evolution
    problem prob{zdt{1u, 30u}};
    population pop{prob, 40u, 23u};
    algorithm algo{nspso{10u, 0.95, 10., 0.01, 0.5, 0.5, 0.5, 2u, "niche count", 24u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<nspso>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    // Change the content of p before deserializing.
    algo = algorithm{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> algo;
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<nspso>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        for (auto j = 0u; j < 2u; ++j) {
            BOOST_CHECK_CLOSE(std::get<2>(before_log[i])[j], std::get<2>(after_log[i])[j], 1e-8);
        }
    }
}
