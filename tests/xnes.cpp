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

#define BOOST_TEST_MODULE cmaes_test
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <limits> //  std::numeric_limits<double>::infinity();
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/xnes.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(xnes_algorithm_construction)
{
    // We test the construction of xnes with standard parameters
    // And that the UDA can be used to construct an algorithm
    BOOST_CHECK_NO_THROW((xnes{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 32u}));
    BOOST_CHECK_NO_THROW((algorithm{xnes{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 32u}}));
    // We test that wrong parameters will result in an error
    BOOST_CHECK_THROW(xnes(10u, -0.6, -1, -1, -1, 1e-6, 1e-6, false, false, 32u), std::invalid_argument);
    BOOST_CHECK_THROW((xnes{10u, 1.6, -1, -1, -1, 1e-6, 1e-6, false, false, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((xnes{10u, -1, -0.6, -1, -1, 1e-6, 1e-6, false, false, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((xnes{10u, -1, 1.6, -1, -1, 1e-6, 1e-6, false, false, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((xnes{10u, -1, -1, -0.6, -1, 1e-6, 1e-6, false, false, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((xnes{10u, -1, -1, 1.6, -1, 1e-6, 1e-6, false, false, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((xnes{10u, -1, -1, -1, -0.6, 1e-6, 1e-6, false, false, 32u}), std::invalid_argument);
    BOOST_CHECK_THROW((xnes{10u, -1, -1, 1, 1.6, 1e-6, 1e-6, false, false, 32u}), std::invalid_argument);
    // We test the defaults are correctly
    xnes user_algo{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 32u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 32u);
    BOOST_CHECK((user_algo.get_log() == xnes::log_type{}));
}

struct unbounded_lb {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0.};
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-std::numeric_limits<double>::infinity()}, {0.}};
    }
};

struct unbounded_ub {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0.};
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {std::numeric_limits<double>::infinity()}};
    }
};

BOOST_AUTO_TEST_CASE(xnes_evolve_test)
{
    {
        // Here we only test that evolution is deterministic if the
        // seed is controlled
        problem prob{rosenbrock{25u}};
        population pop1{prob, 5u, 23u};
        population pop2{prob, 5u, 23u};
        population pop3{prob, 5u, 23u};

        xnes user_algo1{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 23u};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);

        BOOST_CHECK(user_algo1.get_log().size() > 0u);

        xnes user_algo2{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 23u};
        user_algo2.set_verbosity(1u);
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

        user_algo2.set_seed(23u);
        pop3 = user_algo2.evolve(pop3);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }

    {
        // Here we only test that evolution is deterministic if the
        // seed is controlled and force bounds is active
        problem prob{rosenbrock{25u}};
        population pop1{prob, 5u, 23u};
        population pop2{prob, 5u, 23u};
        population pop3{prob, 5u, 23u};

        xnes user_algo1{10u, -1, -1, -1, 1.0, 1e-6, 1e-6, false, true, 23u};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);

        BOOST_CHECK(user_algo1.get_log().size() > 0u);

        xnes user_algo2{10u, -1, -1, -1, 1.0, 1e-6, 1e-6, false, true, 23u};
        user_algo2.set_verbosity(1u);
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

        user_algo2.set_seed(23u);
        pop3 = user_algo2.evolve(pop3);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }

    {
        // Here we only test that evolution is deterministic if the
        // seed is controlled and the problem is stochastic
        problem prob{inventory{4u, 10u, 23u}};
        population pop1{prob, 5u, 23u};
        population pop2{prob, 5u, 23u};

        xnes user_algo1{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 23u};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);

        xnes user_algo2{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 23u};
        user_algo2.set_verbosity(1u);
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log().size() > 0u);
        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }

    // Here we check that the exit condition of ftol and xtol actually provoke an exit within 5000 gen (rosenbrock{2} is
    // used)
    {
        xnes user_algo{5000u, -1, -1, -1, -1, 1e-6, 1e-16, false, false, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 20u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 5000u);
    }
    {
        xnes user_algo{5000u, -1, -1, -1, -1, 1e-16, 1e-6, false, false, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 20u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 5000u);
    }

    // We then check that the evolve throws if called on unsuitable problems
    BOOST_CHECK_THROW(xnes{10u}.evolve(population{problem{rosenbrock{}}, 3u}), std::invalid_argument);
    BOOST_CHECK_THROW(xnes{10u}.evolve(population{problem{zdt{}}, 15u}), std::invalid_argument);
    BOOST_CHECK_THROW(xnes{10u}.evolve(population{problem{hock_schittkowsky_71{}}, 15u}), std::invalid_argument);

    detail::random_engine_type r_engine(32u);
    population pop_lb{problem{unbounded_lb{}}};
    population pop_ub{problem{unbounded_ub{}}};
    for (auto i = 0u; i < 20u; ++i) {
        pop_lb.push_back(pagmo::random_decision_vector({0.}, {1.}, r_engine));
        pop_ub.push_back(pagmo::random_decision_vector({0.}, {1.}, r_engine));
    }
    BOOST_CHECK_THROW(xnes{10u}.evolve(pop_lb), std::invalid_argument);
    BOOST_CHECK_THROW(xnes{10u}.evolve(pop_ub), std::invalid_argument);
    // And a clean exit for 0 generations
    population pop{rosenbrock{25u}, 10u};
    BOOST_CHECK(xnes{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);

    // and we call evolve on the stochastic problem
    BOOST_CHECK_NO_THROW(xnes{10u}.evolve(population{problem{inventory{}}, 15u}));
}

BOOST_AUTO_TEST_CASE(xnes_setters_getters_test)
{
    xnes user_algo{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 23u};
    xnes user_algo2{10u, .5, .5, .5, .5, 1e-6, 1e-6, false, false, 23u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("xNES") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("eta_mu") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("auto") != std::string::npos);
    BOOST_CHECK(user_algo2.get_extra_info().find("auto") == std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(xnes_serialization_test)
{
    // Make one evolution
    problem prob{rosenbrock{25u}};
    population pop{prob, 10u, 23u};
    algorithm algo{xnes{10u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<xnes>()->get_log();
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
    auto after_log = algo.extract<xnes>()->get_log();
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
        BOOST_CHECK_CLOSE(std::get<5>(before_log[i]), std::get<5>(after_log[i]), 1e-8);
    }
}

BOOST_AUTO_TEST_CASE(xnes_memory_test)
{
    // We check here that when memory is true calling evolve(pop) two times on 1 gen
    // is the same as calling 1 time evolve with 2 gens
    xnes user_algo{1u, -1, -1, -1, -1, 1e-6, 1e-6, true, false, 23u};
    user_algo.set_verbosity(1u);
    problem prob{rosenbrock{25u}};
    population pop{prob, 10u, 23u};
    pop = user_algo.evolve(pop);
    pop = user_algo.evolve(pop);

    xnes user_algo2{2u, -1, -1, -1, -1, 1e-6, 1e-6, false, false, 23u};
    user_algo2.set_verbosity(1u);
    problem prob2{rosenbrock{25u}};
    population pop2{prob2, 10u, 23u};
    pop2 = user_algo2.evolve(pop2);

    auto log = user_algo.get_log();
    auto log2 = user_algo2.get_log();
    BOOST_CHECK_CLOSE(std::get<5>(log[0]), std::get<5>(log2[1]), 1e-8);
    BOOST_CHECK_CLOSE(std::get<4>(log[0]), std::get<4>(log2[1]), 1e-8);
    BOOST_CHECK_CLOSE(std::get<3>(log[0]), std::get<3>(log2[1]), 1e-8);
    BOOST_CHECK_CLOSE(std::get<2>(log[0]), std::get<2>(log2[1]), 1e-8);
    // the 1 and 0 will be different as fevals is reset at each evolve
}
