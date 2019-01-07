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

#define BOOST_TEST_MODULE ihs_problem_test
#include <boost/lexical_cast.hpp>
#include <boost/test/included/unit_test.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>

using namespace pagmo;
using namespace std;

BOOST_AUTO_TEST_CASE(ihs_algorithm_construction)
{
    {
        // Here we construct a valid ihs uda
        ihs user_algo{1u, 0.85, 0.35, 0.99, 1e-5, 1., 42u};
        BOOST_CHECK(user_algo.get_verbosity() == 0u);
        BOOST_CHECK(user_algo.get_seed() == 42u);
        BOOST_CHECK((user_algo.get_log() == ihs::log_type{}));
    }

    // Here we construct invalid ihs udas and test that construction fails
    BOOST_CHECK_THROW((ihs{1u, 1.2, 0.35, 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, -0.2, 0.35, 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 23., 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, -22.4, 0.99, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 12., 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, -0.2, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 0.34, 1e-5, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 0.99, -0.43, 1., 42u}), std::invalid_argument);
    BOOST_CHECK_THROW((ihs{1u, 0.85, 0.35, 0.99, 0.4, 0.3, 42u}), std::invalid_argument);
}

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

BOOST_AUTO_TEST_CASE(ihs_evolve_test)
{
    // We test for unsuitable populations
    {
        population pop{rosenbrock{25u}};
        BOOST_CHECK_THROW(ihs{15u}.evolve(pop), std::invalid_argument);
        population pop2{null_problem{2u, 3u, 4u}, 20u};
        BOOST_CHECK_THROW(ihs{15u}.evolve(pop2), std::invalid_argument);
        population pop3{inventory{}, 20u};
        BOOST_CHECK_THROW(ihs{15u}.evolve(pop3), std::invalid_argument);
    }
    // And a clean exit for 0 generations
    {
        population pop1{rosenbrock{25u}, 10u};
        BOOST_CHECK(ihs{0u}.evolve(pop1).get_x()[0] == pop1.get_x()[0]);
    }
    // Here we only test that evolution is deterministic if the
    // seed is controlled
    std::vector<problem> prob_list;
    prob_list.push_back(problem(rosenbrock{10u}));
    prob_list.push_back(problem(zdt{1u}));
    prob_list.push_back(problem(hock_schittkowsky_71{}));
    prob_list.push_back(problem(minlp_rastrigin{}));
    for (auto &prob : prob_list) {
        prob.set_c_tol(1e-4);
        population pop1{prob, 20u, 42u};
        population pop2{prob, 20u, 42u};
        population pop3{prob, 20u, 42u};
        ihs uda1{1000u, 0.85, 0.35, 0.99, 1e-5, 1., 42u};
        ihs uda2{1000u, 0.85, 0.35, 0.99, 1e-5, 1., 42u};
        uda1.set_verbosity(100u);
        uda2.set_verbosity(100u);
        pop1 = uda1.evolve(pop1);
        BOOST_CHECK(uda1.get_log().size() > 0u);
        pop2 = uda2.evolve(pop2);
        BOOST_CHECK(uda1.get_log() == uda2.get_log());
        uda2.set_seed(42u);
        pop3 = uda2.evolve(pop3);
        BOOST_CHECK(uda1.get_log() == uda2.get_log());
    }
    // We test a call on many objectives (>5) to trigger the relative lines cropping the screen output
    ihs uda1{100u, 0.85, 0.35, 0.99, 1e-5, 1., 42u};
    uda1.set_verbosity(10u);
    population pop{problem{mo_many{}}, 56u, 23u};
    uda1.evolve(pop);
    BOOST_CHECK(std::get<7>(uda1.get_log()[0]).size() == 6u);
}

BOOST_AUTO_TEST_CASE(ihs_setters_getters_test)
{
    ihs user_algo{1u, 0.85, 0.35, 0.99, 1e-5, 1., 42u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("Improved Harmony Search") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Maximum distance bandwidth") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(ihs_serialization_test)
{
    // Make one evolution
    problem prob{rosenbrock{25u}};
    population pop{prob, 10u, 23u};
    algorithm algo{ihs{100u, 0.85, 0.35, 0.99, 1e-5, 1., 42u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<ihs>()->get_log();
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
    auto after_log = algo.extract<ihs>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<1>(before_log[i]), std::get<1>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<4>(before_log[i]), std::get<4>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<5>(before_log[i]), std::get<5>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<6>(before_log[i]), std::get<6>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<7>(before_log[i])[0], std::get<7>(after_log[i])[0], 1e-8);
    }
}
