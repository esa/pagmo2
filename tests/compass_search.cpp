/* Copyright 2017-2026 PaGMO development team

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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowski_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
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
    BOOST_CHECK_THROW((compass_search{1234u, std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(compass_search_evolve_test)
{
    double stop_range = 1e-5;

    // Here we only test that evolution is deterministic (stop criteria will be range)
    problem prob1{hock_schittkowski_71{}};
    prob1.set_c_tol({1e-3, 1e-3});
    population pop1{prob1, 5u, 23u};

    problem prob2{hock_schittkowski_71{}};
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
    algorithm algo{compass_search{10000u, 0.5, 0.1, 0.5, 1e6}};
    algo.set_verbosity(1u); // allows the log to be filled
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<compass_search>()->get_log();
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
    auto after_log = algo.extract<compass_search>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
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

BOOST_AUTO_TEST_CASE(compass_search_time_limit_test)
{
    // The time limit defaults to zero (no limit) and must be a non-negative finite number of seconds.
    BOOST_CHECK_EQUAL(compass_search{}.get_max_time(), 0.);
    BOOST_CHECK_EQUAL((compass_search{10u, 0.5, 0.01, 0.5, 1.5}).get_max_time(), 1.5);
    for (auto max_time : {-1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        BOOST_CHECK_THROW((compass_search{10u, 0.5, 0.01, 0.5, max_time}), std::invalid_argument);
    }
    compass_search uda{};
    uda.set_max_time(2.5);
    BOOST_CHECK_EQUAL(uda.get_max_time(), 2.5);
    BOOST_CHECK_THROW(uda.set_max_time(-1.), std::invalid_argument);
    BOOST_CHECK_THROW(uda.set_max_time(std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_EQUAL(uda.get_max_time(), 2.5);
    BOOST_CHECK(uda.get_extra_info().find("Max time") != std::string::npos);

    population pop{rosenbrock{10u}, 20u, 23u};
    // A time limit that is not reached leaves the evolution unchanged.
    {
        auto uda1 = compass_search{1000u, 0.5, 0.01, 0.5, 0.};
        auto uda2 = compass_search{1000u, 0.5, 0.01, 0.5, 1e6};
        uda1.set_verbosity(1u);
        uda2.set_verbosity(1u);
        const auto pop1 = uda1.evolve(pop);
        const auto pop2 = uda2.evolve(pop);
        BOOST_CHECK(pop1.get_x() == pop2.get_x());
        BOOST_CHECK(pop1.get_f() == pop2.get_f());
        BOOST_CHECK(uda1.get_log() == uda2.get_log());
    }
    // The time limit stops the evolution long before the generation limit, and each call to evolve() has
    // its own time budget.
    {
        const double max_time = 0.1;
        auto uda1 = compass_search{std::numeric_limits<unsigned>::max(), 0.5, 0., 1. - 1e-9, max_time};
        auto pop1 = pop;
        for (auto i = 0; i < 2; ++i) {
            const auto start = std::chrono::steady_clock::now();
            pop1 = uda1.evolve(pop1);
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            BOOST_CHECK_GE(elapsed, max_time);
            BOOST_CHECK_LT(elapsed, max_time + 5.);
        }
    }
    // However small the time limit, the first iteration is always run.
    {
        auto uda1 = compass_search{std::numeric_limits<unsigned>::max(), 0.5, 0.01, 0.5, 1e-9};
        uda1.set_verbosity(1u);
        uda1.evolve(pop);
        BOOST_CHECK_EQUAL(uda1.get_log().size(), 1u);
    }
}
