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

#define BOOST_TEST_MODULE moead_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/moead.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(moead_algorithm_construction)
{
    moead uda{10u, "grid", "tchebycheff", 20u, 1., 0.5, 20., 0.9, 2u, true, 23u};
    BOOST_CHECK(uda.get_verbosity() == 0u);
    BOOST_CHECK(uda.get_seed() == 23u);
    BOOST_CHECK((uda.get_log() == moead::log_type{}));

    // Check the throws
    // Wrong weight generation type
    BOOST_CHECK_THROW((moead{10u, "typo", "tchebycheff", 20u, 1., 0.5, 20., 0.9, 2u, true, 23u}),
                      std::invalid_argument);
    // Wrong decomposition method
    BOOST_CHECK_THROW((moead{10u, "grid", "typo", 20u, 1., 0.5, 20., 0.9, 2u, true, 23u}), std::invalid_argument);
    // Wrong CR
    BOOST_CHECK_THROW((moead{10u, "grid", "tchebycheff", 20u, 1.1, 0.5, 20., 0.9, 2u, true, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((moead{10u, "grid", "tchebycheff", 20u, -0.3, 0.5, 20., 0.9, 2u, true, 23u}),
                      std::invalid_argument);
    // Wrong F
    BOOST_CHECK_THROW((moead{10u, "grid", "tchebycheff", 20u, 1., 1.1, 20., 0.9, 2u, true, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((moead{10u, "grid", "tchebycheff", 20u, 1., -0.3, 20., 0.9, 2u, true, 23u}),
                      std::invalid_argument);
    // Wrong eta_m
    BOOST_CHECK_THROW((moead{10u, "grid", "tchebycheff", 20u, 1., 0.5, -20., 0.9, 2u, true, 23u}),
                      std::invalid_argument);
    // Wrong realb
    BOOST_CHECK_THROW((moead{10u, "grid", "tchebycheff", 20u, 1., 0.5, 20., 1.1, 2u, true, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((moead{10u, "grid", "tchebycheff", 20u, 1., 0.5, 20., -0.34, 2u, true, 23u}),
                      std::invalid_argument);
}

struct mo_con {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0., 0., 0.};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    vector_double::size_type get_nec() const
    {
        return 1u;
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0., 0.}, {1., 1.}};
    }
};

struct mo_sto {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0., 0.};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0., 0.}, {1., 1.}};
    }
    void set_seed(unsigned int) {}
};

struct mo_many {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0., 0., 0., 0., 0., 0.};
    }
    vector_double::size_type get_nobj() const
    {
        return 6u;
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0., 0.}, {1., 1.}};
    }
};

BOOST_AUTO_TEST_CASE(moead_evolve_test)
{
    // Here we only test that evolution is deterministic if the
    // seed is controlled
    problem prob{zdt{1u, 30u}};
    population pop1{prob, 40u, 23u};
    population pop2{prob, 40u, 23u};
    population pop3{prob, 40u, 23u};

    moead user_algo1{10u, "grid", "tchebycheff", 20u, 1., 0.5, 20., 0.9, 2u, true, 23u};
    user_algo1.set_verbosity(1u);
    pop1 = user_algo1.evolve(pop1);

    BOOST_CHECK(user_algo1.get_log().size() > 0u);

    moead user_algo2{10u, "grid", "tchebycheff", 20u, 1., 0.5, 20., 0.9, 2u, true, 23u};
    user_algo2.set_verbosity(1u);
    pop2 = user_algo2.evolve(pop2);

    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    user_algo2.set_seed(23u);
    pop3 = user_algo2.evolve(pop3);

    BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

    // We then check that the method evolve fails when called on unsuitable problems (populations)
    // Empty population.
    BOOST_CHECK_THROW(moead{10u}.evolve(population{problem{rosenbrock{}}, 0u}), std::invalid_argument);
    // Single objective problem
    BOOST_CHECK_THROW(moead{10u}.evolve(population{problem{rosenbrock{}}, 20u}), std::invalid_argument);
    // Multi-objective problem with constraints
    BOOST_CHECK_THROW(moead{10u}.evolve(population{problem{mo_con{}}, 20u}), std::invalid_argument);
    // Stochastic problem
    BOOST_CHECK_THROW(moead{10u}.evolve(population{problem{mo_sto{}}, 15u}), std::invalid_argument);
    // Population size is too small for the neighbourhood specified
    BOOST_CHECK_THROW(moead(10u, "grid", "tchebycheff", 20u).evolve(population{problem{zdt{}}, 15u}),
                      std::invalid_argument);

    // And a clean exit for 0 generations
    population pop{zdt{}, 40u};
    BOOST_CHECK(moead{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);

    // We test a call on many objectives (>5) to trigger the relative lines cropping the screen output
    population pop4{problem{mo_many{}}, 56u, 23u};
    user_algo1.evolve(pop4);
    BOOST_CHECK(std::get<3>(user_algo1.get_log()[0]).size() == 6u);
}

BOOST_AUTO_TEST_CASE(moead_setters_getters_test)
{
    moead user_algo{10u, "grid", "tchebycheff", 20u, 1., 0.5, 20., 0.9, 2u, true, 23u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(15u);
    BOOST_CHECK(user_algo.get_seed() == 15u);
    BOOST_CHECK(user_algo.get_name().find("MOEA/D") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Distribution index") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(moead_serialization_test)
{
    // Make one evolution
    problem prob{zdt{1u, 30u}};
    population pop{prob, 40u, 23u};
    algorithm algo{moead{10u, "grid", "tchebycheff", 10u, 0.9, 0.5, 20., 0.9, 2u, true, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<moead>()->get_log();
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
    auto after_log = algo.extract<moead>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        for (decltype(2u) j = 0u; j < 2u; ++j) {
            BOOST_CHECK_CLOSE(std::get<3>(before_log[i])[j], std::get<3>(after_log[i])[j], 1e-8);
        }
    }
}
