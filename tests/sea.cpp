#define BOOST_TEST_MODULE sea_problem_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/algorithm.hpp"
#include "../include/algorithms/null_algorithm.hpp"
#include "../include/algorithms/sea.hpp"
#include "../include/io.hpp"
#include "../include/population.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/problems/inventory.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/serialization.hpp"
#include "../include/types.hpp"

using namespace pagmo;
using namespace std;

BOOST_AUTO_TEST_CASE(sea_algorithm_construction)
{
    sea user_algo{1234u, 42u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 42u);
    BOOST_CHECK((user_algo.get_log() == sea::log_type{}));
}

BOOST_AUTO_TEST_CASE(sea_evolve_test)
{
    // Here we only test that evolution is deterministic if the
    // seed is controlled.
    //
    // On a single objective deterministic problem
    {
        problem prob1{rosenbrock{25u}};
        population pop1{prob1, 5u, 23u};
        sea user_algo1{10u, 23u};
        user_algo1.set_verbosity(1u);
        pop1 = user_algo1.evolve(pop1);

        problem prob2{rosenbrock{25u}};
        population pop2{prob2, 5u, 23u};
        sea user_algo2{10u, 23u};
        user_algo2.set_verbosity(1u);
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }
    // On a single objective stochastic problem
    {
        problem prob1{inventory{25u, 5u, 1432u}};
        population pop1{prob1, 5u, 23u};
        sea user_algo1{10u, 23u};
        user_algo1.set_verbosity(2u);
        pop1 = user_algo1.evolve(pop1);

        problem prob2{inventory{25u, 5u, 1432u}};
        population pop2{prob2, 5u, 23u};
        sea user_algo2{10u, 23u};
        user_algo2.set_verbosity(2u); // more verbosity here to also cover the relative code lines
        pop2 = user_algo2.evolve(pop2);

        BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());
    }
    // We then check that the evolve throws if called on unsuitable problems
    BOOST_CHECK_THROW(sea{10u}.evolve(population{problem{zdt{}}, 5u, 23u}), std::invalid_argument);
    BOOST_CHECK_THROW(sea{10u}.evolve(population{problem{hock_schittkowsky_71{}}, 5u, 23u}), std::invalid_argument);
    // Or with not enough individuals
    BOOST_CHECK_THROW(sea{10u}.evolve(population{problem{rosenbrock{}}, 0u}), std::invalid_argument);
    // And a clean exit for 0 generations
    population pop{rosenbrock{25u}, 10u};
    BOOST_CHECK(sea{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
}

BOOST_AUTO_TEST_CASE(sea_setters_getters_test)
{
    sea user_algo{10u, 23u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("Simple Evolutionary Algorithm") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Verbosity") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(sea_serialization_test)
{
    // Make one evolution
    problem prob{rosenbrock{25u}};
    population pop{prob, 5u, 23u};
    algorithm algo{sea{10u, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<sea>()->get_log();
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
    auto after_log = algo.extract<sea>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<4>(before_log[i]), std::get<4>(after_log[i]));
    }
}
