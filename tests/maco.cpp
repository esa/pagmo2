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

#define BOOST_TEST_MODULE maco_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/maco.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/lennard_jones.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/wfg.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(maco_algorithm_construction)
{
    maco user_algo{1u, 63u, 1.0, 1u, 7u, 10000u, 0., false, 23u};
    BOOST_CHECK_NO_THROW(maco{});
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);

    // Check the throws
    // Wrong focus
    BOOST_CHECK_THROW((maco{1u, 63u, 1.0, 1u, 7u, 10000u, -1.0, false, 23u}), std::invalid_argument);
    // Wrong threshold
    BOOST_CHECK_THROW((maco{1u, 63u, 1.0, 3u, 7u, 10000u, 0., false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((maco{1u, 63u, 1.0, 0, 7u, 10000u, 0., true, 23u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(maco_evolve_test)
{
    // We check that the problem is checked to be suitable
    // stochastic
    BOOST_CHECK_THROW((maco{}.evolve(population{inventory{}, 63u, 23u})), std::invalid_argument);
    // Empty population.
    BOOST_CHECK_THROW(maco{10u}.evolve(population{problem{rosenbrock{}}, 0u}), std::invalid_argument);
    // constrained prob
    BOOST_CHECK_THROW((maco{}.evolve(population{hock_schittkowsky_71{}, 63u, 23u})), std::invalid_argument);
    // single objective prob
    BOOST_CHECK_THROW((maco{}.evolve(population{rosenbrock{}, 63u, 23u})), std::invalid_argument);
    // population size smaller than ker size
    BOOST_CHECK_THROW((maco{}.evolve(population{zdt{}, 3u, 23u})), std::invalid_argument);
    // and a clean exit for 0 generation
    population pop{zdt{}, 63u};
    BOOST_CHECK(maco{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);

    // We check for deterministic behaviour if the seed is controlled
    // we treat the last three components of the decision vector as integers
    // to trigger all cases
    dtlz udp{1u, 10u, 3u};

    population pop1{udp, 52u, 23u};
    population pop2{udp, 52u, 23u};
    population pop3{udp, 52u, 23u};

    maco user_algo1{10u, 50u, 1.0, 1u, 7u, 10000u, 0., false, 23u};
    user_algo1.set_verbosity(1u);
    pop1 = user_algo1.evolve(pop1);

    BOOST_CHECK(user_algo1.get_log().size() > 0u);

    maco user_algo2{10u, 50u, 1.0, 1u, 7u, 10000u, 0., false, 23u};
    user_algo2.set_verbosity(1u);
    pop2 = user_algo2.evolve(pop2);

    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    user_algo2.set_seed(23u);
    pop3 = user_algo2.evolve(pop3);

    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    // We evolve for many-objectives and trigger the output with the ellipses
    udp = dtlz{1u, 12u, 7u};
    population pop4{udp, 50u, 23u};
    pop4 = user_algo2.evolve(pop4);
}

BOOST_AUTO_TEST_CASE(maco_setters_getters_test)
{
    maco user_algo{1u, 63u, 1.0, 1u, 7u, 10000u, 0., false, 23u};
    user_algo.set_verbosity(200u);
    BOOST_CHECK(user_algo.get_verbosity() == 200u);
    user_algo.set_seed(23456u);
    BOOST_CHECK(user_algo.get_seed() == 23456u);
    BOOST_CHECK(user_algo.get_name().find("MHACO: Multi-objective Hypervolume-based Ant Colony Optimization")
                != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Verbosity") != std::string::npos);
}

// Integer test
BOOST_AUTO_TEST_CASE(maco_zdt5_test)
{
    algorithm algo{maco(100u, 13u, 1.0, 1u, 7u, 10000u, 0., false, 23u)};
    algo.set_verbosity(10u);
    algo.set_seed(23456u);
    population pop{zdt(5u, 10u), 20u, 32u};
    pop = algo.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        auto x = pop.get_x()[i];
        BOOST_CHECK(std::all_of(x.begin(), x.end(), [](double el) { return (el == std::floor(el)); }));
    }
}

BOOST_AUTO_TEST_CASE(maco_serialization_test)
{
    // Make one evolution
    problem prob{zdt{1u, 30u}};
    population pop{prob, 40u, 23u};
    algorithm algo{maco{10u, 13u, 1.0, 1u, 7u, 10000u, 0., false, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<maco>()->get_log();
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
    auto after_log = algo.extract<maco>()->get_log();
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

BOOST_AUTO_TEST_CASE(bfe_usage_test)
{
    // 1 - Algorithm with bfe disabled
    problem prob{dtlz(1, 10, 2)};
    maco uda1{maco{10, 43u}};
    uda1.set_verbosity(1u);
    uda1.set_seed(23u);
    // 2 - Instantiate
    algorithm algo1{uda1};

    // 3 - Instantiate populations
    population pop{prob, 43};
    population pop1{prob, 43};
    population pop2{prob, 43};

    // 4 - Evolve the population
    pop1 = algo1.evolve(pop);

    // 5 new algorithm that is bfe enabled
    maco uda2{maco{10, 43u}};
    uda2.set_verbosity(1u);
    uda2.set_seed(23u);
    uda2.set_bfe(bfe{}); // This will use the default bfe.
    // 6 - Instantiate a pagmo algorithm
    algorithm algo2{uda2};

    // 7 - Evolve the population
    pop2 = algo2.evolve(pop);
    BOOST_CHECK(algo1.extract<maco>()->get_log() == algo2.extract<maco>()->get_log());
}

// We now introduce some tests for coverage purposes:
BOOST_AUTO_TEST_CASE(miscellanea_tests)
{
    // 1 - Algorithm definition
    problem prob{dtlz(1, 10, 2)};
    maco uda{maco{10u, 43u, 1.0, 1u, 7u, 1u, 0., false, 23u}};
    maco uda2{maco{10u, 5u, 1.0, 1u, 7u, 10000u, 2.0, false, 23u}};
    maco uda3{maco{5u, 3u, 1.0, 5u, 3u, 10000u, 0.0, false, 23u}};
    maco uda4{maco{1u, 5u, 1.0, 1u, 7u, 10000u, 2.0, true, 23u}};

    uda.set_verbosity(1u);

    // 2 - Instantiate
    algorithm algo{uda};
    algorithm algo2{uda2};
    algorithm algo3{uda3};
    algorithm algo4{uda4};

    // 3 - Instantiate populations
    population pop{prob, 43};
    population pop2{prob, 6};
    population pop3{prob, 50};
    population pop4{prob, 6};

    // 4 - Evolve the populations
    pop = algo.evolve(pop);
    pop2 = algo2.evolve(pop2);
    pop3 = algo3.evolve(pop3);
    pop4 = algo4.evolve(pop4);
    for (int iter = 0u; iter < 10; ++iter) {
        pop4 = uda4.evolve(pop4);
    }
}

BOOST_AUTO_TEST_CASE(memory_test)
{
    maco uda{1u, 20u, 1.0, 8u, 7u, 10000u, 0., true, 23u};
    maco uda_2{10u, 20u, 1.0, 8u, 7u, 10000u, 0., false, 23u};
    uda.set_seed(23u);
    uda_2.set_seed(23u);
    uda.set_verbosity(1u);
    uda_2.set_verbosity(1u);
    problem prob{wfg{5u, 16u, 15u, 14u}};
    population pop_1{prob, 20u, 23u};
    population pop_2{prob, 20u, 23u};
    for (int iter = 0u; iter < 10; ++iter) {
        pop_1 = uda.evolve(pop_1);
    }
    pop_2 = uda_2.evolve(pop_2);
    BOOST_CHECK(pop_1.get_f() == pop_2.get_f());
}
