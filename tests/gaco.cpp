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

#define BOOST_TEST_MODULE gaco_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/cec2006.hpp>
#include <pagmo/problems/golomb_ruler.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/lennard_jones.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// Construction tests: we verify that the algorithm throws an error for wrong input parameters
BOOST_AUTO_TEST_CASE(construction_test)
{
    gaco user_algo{2u, 13u, 1.0, 0.0, 0.01, 1u, 7u, 1000u, 1000u, 0.0, false, 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK((user_algo.get_log() == gaco::log_type{}));
    BOOST_CHECK_THROW((gaco{2u, 13u, 1.0, 0.0, -0.01, 1u, 7u, 1000u, 1000u, 0.1, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((gaco{2u, 13u, 1.0, 0.0, 0.01, 1u, 7u, 1000u, 1000u, -0.1, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((gaco{2u, 13u, -1.0, 0.0, 0.01, 1u, 7u, 1000u, 1000u, 0.0, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((gaco{2u, 13u, 1.0, 0.0, 0.01, 3u, 7u, 1000u, 1000u, 0.0, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((gaco{2u, 13u, 1.0, 0.0, 0.01, 0u, 7u, 1000u, 1000u, 0.0, false, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW((gaco{2u, 13u, 1.0, 0.0, 0.01, 0u, 7u, 1000u, 1000u, 0.0, true, 23u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(evolve_test)
{
    // Here we only test that evolution is deterministic if the
    // seed is controlled for all variants
    {
        problem prob{rosenbrock{2u}};
        population pop1{prob, 10u, 23u};
        population pop2{prob, 10u, 23u};
        population pop3{prob, 10u, 23u};
        for (unsigned i = 1u; i < 3u; ++i) {
            gaco user_algo1{3u, 5u, 1.0, 1e9, 0.01, i, 7u, 1000u, 1000u, 0.0, false, 23u};
            user_algo1.set_verbosity(1u);
            pop1 = user_algo1.evolve(pop1);
            BOOST_CHECK(user_algo1.get_log().size() > 0u);
            gaco user_algo2{3u, 5u, 1.0, 1e9, 0.01, i, 7u, 1000u, 1000u, 0.0, false, 23u};
            user_algo2.set_verbosity(1u);
            pop2 = user_algo2.evolve(pop2);

            BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

            gaco user_algo3{3u, 5u, 1.0, 1e9, 0.01, i, 7u, 1000u, 1000u, 0.0, false, 23u};
            user_algo3.set_verbosity(1u);
            pop3 = user_algo3.evolve(pop3);

            BOOST_CHECK(user_algo2.get_log() == user_algo3.get_log());
        }
    }
    // Here we check that the exit condition of impstop and evalstop actually provoke an exit within 300u gen
    // (rosenbrock{10} and rosenbrock{2} are used)
    {
        gaco user_algo{200u, 15u, 1.0, 0.0, 0.01, 150u, 7u, 1u, 1000u, 0.0, false, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 20u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 200u);
    }
    {
        gaco user_algo{200u, 15u, 1.0, 0.0, 0.01, 150u, 7u, 1000u, 1u, 0.0, false, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 20u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 200u);
    }

    // We then check that the evolve throws if called on unsuitable problems
    // Integer variables problem
    //    BOOST_CHECK_THROW(gaco{2u}.evolve(population{problem{minlp_rastrigin{}}, 64u}), std::invalid_argument);
    // Multi-objective problem
    BOOST_CHECK_THROW(gaco{2u}.evolve(population{problem{zdt{}}, 64u}), std::invalid_argument);
    // Population size smaller than ker size
    BOOST_CHECK_THROW(gaco{2u}.evolve(population{problem{rosenbrock{}}, 60u}), std::invalid_argument);
    // Stochastic problem
    BOOST_CHECK_THROW((gaco{}.evolve(population{inventory{}, 65u, 23u})), std::invalid_argument);
    // and a clean exit for 0 generation
    population pop{rosenbrock{2u}, 10u};
    BOOST_CHECK(gaco{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
}

BOOST_AUTO_TEST_CASE(setters_getters_test)
{
    gaco user_algo{10u, 13u, 1.0, 0.0, 0.01, 9u, 7u, 1000u, 1000u, 0.0, false, 23u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("GACO: Ant Colony Optimization") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Oracle parameter") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(serialization_test)
{
    // Make one evolution
    problem prob{rosenbrock{2u}};
    population pop{prob, 15u, 23u};
    algorithm algo{gaco{10u, 13u, 1.0, 100.0, 0.01, 9u, 7u, 1000u, 1000u, 0.0, false, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<gaco>()->get_log();
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
    auto after_log = algo.extract<gaco>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<3>(before_log[i]), std::get<3>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<4>(before_log[i]), std::get<4>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<5>(before_log[i]), std::get<5>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<6>(before_log[i]), std::get<6>(after_log[i]), 1e-8);
    }
}

// Coverage tests: we make sure that the algorithm is tested in all the lines
BOOST_AUTO_TEST_CASE(miscellaneous_tests)
{
    problem prob{hock_schittkowsky_71{}};
    population population1{prob, 15u, 23u};
    prob.set_c_tol(1.0);
    gaco user_algo1{100u, 13u, 1.0, 1500.0, 0.01, 90u, 1u, 1000u, 1000u, 1000.0, false, 23u};
    user_algo1.set_verbosity(1u);
    population1 = user_algo1.evolve(population1);
    population population2{prob, 15u, 23u};
    gaco user_algo2{100u, 13u, 1.0, 2700.0, 0.01, 90u, 7u, 1000u, 1000u, 0.0, false, 23u};
    user_algo2.set_verbosity(1u);
    population2 = user_algo2.evolve(population2);
    population population3{prob, 15u, 23u};
    gaco user_algo3{100u, 13u, 1.0, 1500.0, 0.01, 90u, 7u, 1000u, 1000u, 0.0, false, 23u};
    user_algo3.set_verbosity(1u);
    population3 = user_algo3.evolve(population3);
    population population4{prob, 150u, 23u};
    gaco user_algo4{10u, 130u, 1.5, 1500.0, 0.01, 9u, 7u, 1000u, 1000u, 0.0, false, 23u}; // 1
    user_algo4.set_verbosity(1u);
    population4 = user_algo4.evolve(population4);
    population population5{prob, 150u, 23u};
    gaco user_algo5{10u, 130u, 1.5, 1500.0, 0.01, 9u, 7u, 1000u, 1000u, 0.0, false, 23u}; // 3
    user_algo5.set_verbosity(1u);
    population5 = user_algo5.evolve(population5);
    problem prob_ros{rosenbrock{10u}};
    population population6{prob_ros, 200u, 23u};
    gaco user_algo6{20u, 150u, 1.0, 0.0, 0.0, 9u, 7u, 10000u, 10000u, 0.0, false, 23u};
    user_algo6.set_verbosity(1u);
    population6 = user_algo6.evolve(population6);
    problem prob_cec{cec2006{1u}};
    population population7{prob_cec, 20u, 23u};
    gaco user_algo7{20u, 15u, 1.0, 1e9, 0.0, 9u, 7u, 10000u, 10000u, 0.0, false, 23u}; // 3
    user_algo7.set_verbosity(1u);
    population7 = user_algo7.evolve(population7);
    BOOST_CHECK(user_algo1.get_log().size() > 0u);
    BOOST_CHECK(user_algo2.get_log().size() > 0u);
    BOOST_CHECK(user_algo3.get_log().size() > 0u);
    BOOST_CHECK(user_algo4.get_log().size() > 0u);
    BOOST_CHECK(user_algo5.get_log().size() > 0u);
    BOOST_CHECK(user_algo6.get_log().size() > 0u);
    BOOST_CHECK(user_algo7.get_log().size() > 0u);
}

BOOST_AUTO_TEST_CASE(construction_test_2)
{
    cec2006 udp(1u);
    problem prob(udp);
    cstrs_self_adaptive uda(4, gaco(10, 10, 1., -15., 0., 7, 7));
    algorithm algo(uda);
    population pop(prob, 10);
    algo.set_verbosity(1);
    pop = algo.evolve(pop);
}

// Inf and NaN tests: we verify that the algo can handle NaN and inf has objectives w/o returning errors
struct udp_inf {

    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        double inf = std::numeric_limits<double>::infinity();
        return {-inf};
    }
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }

    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0., 0.}, {1., 1.}};
    }
};

struct udp_nan {

    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan};
    }
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }

    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0., 0.}, {1., 1.}};
    }
};

BOOST_AUTO_TEST_CASE(test_for_inf_and_nan)
{
    gaco{10, 10, 1., -15., 0., 7}.evolve(population{problem{udp_inf{}}, 15u});
    gaco{10, 10, 1., -15., 0., 7}.evolve(population{problem{udp_nan{}}, 15u});
}

// Memory test: we verify that the algorithm has implemented memory correctly
BOOST_AUTO_TEST_CASE(memory_test)
{
    gaco uda{1u, 20u, 1.0, 1e9, 0.01, 1u, 7u, 1000u, 1000u, 100.0, true, 23u};
    gaco uda_2{10u, 20u, 1.0, 1e9, 0.01, 1u, 7u, 1000u, 1000u, 100.0, false, 23u};
    // for coverage purposes we introduce also a third one:
    gaco uda_3{1u, 2u, 1.0, 1e9, 0.01, 1u, 7u, 1000u, 1000u, 100.0, false, 23u};
    uda.set_seed(23u);
    uda_2.set_seed(23u);
    uda_3.set_seed(23u);
    uda.set_verbosity(1u);
    uda_2.set_verbosity(1u);
    uda_3.set_verbosity(1u);
    problem prob{rosenbrock{2u}};
    population pop_1{prob, 20u, 23u};
    population pop_2{prob, 20u, 23u};
    population pop_3{prob, 2u, 23u};
    for (int iter = 0u; iter < 10; ++iter) {
        pop_1 = uda.evolve(pop_1);
    }
    pop_2 = uda_2.evolve(pop_2);
    pop_3 = uda_3.evolve(pop_3);
    BOOST_CHECK_EQUAL(pop_1.champion_f()[0], pop_2.champion_f()[0]);
}

// Integer tests: we verify that the returned population is actually made by integers
BOOST_AUTO_TEST_CASE(integer_test_1)
{
    population pop{minlp_rastrigin{3u, 3u}, 10u, 23u};
    gaco uda{10u, 10u, 1.0, 1e9, 0.0, 5u, 7u, 1000u, 1000u, 0.0, false, 23u};
    uda.set_verbosity(1u);
    pop = uda.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        auto x = pop.get_x()[i];
        BOOST_CHECK(std::all_of(x.begin() + 3, x.end(), [](double el) { return (el == std::floor(el)); }));
    }
}

BOOST_AUTO_TEST_CASE(integer_test_2)
{
    population pop{golomb_ruler{7, 10}, 10u, 23u};
    gaco uda{10u, 10u, 1.0, 25.0, 0.01, 5u, 7u, 1000u, 1000u, 0.0, false, 23u};
    uda.set_verbosity(1u);
    pop = uda.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        auto x = pop.get_x()[i];
        BOOST_CHECK(std::all_of(x.begin(), x.end(), [](double el) { return (el == std::floor(el)); }));
    }
}

BOOST_AUTO_TEST_CASE(bfe_usage_test)
{
    population pop{rosenbrock{10u}, 200u, 23u};
    gaco uda{40u, 10u, 1.0, 25.0, 0.01, 5u, 7u, 1000u, 1000u, 0.0, false, 23u};
    uda.set_verbosity(1u);
    uda.set_seed(23u);
    uda.set_bfe(bfe{}); // This will use the default bfe.
    pop = uda.evolve(pop);

    population pop_2{rosenbrock{10u}, 200u, 23u};
    gaco uda_2{40u, 10u, 1.0, 25.0, 0.01, 5u, 7u, 1000u, 1000u, 0.0, false, 23u};
    uda_2.set_verbosity(1u);
    uda_2.set_seed(23u);
    pop_2 = uda_2.evolve(pop_2);

    BOOST_CHECK_EQUAL(pop.champion_f()[0], pop_2.champion_f()[0]);
}
