#define BOOST_TEST_MODULE null_algo_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/null_algorithm.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(algorithm_construction_and_evolve)
{
    // Trivial checks
    null_algorithm user_algo{};
    BOOST_CHECK(user_algo.get_a() == 42.1);
    BOOST_CHECK(user_algo.get_extra_info().find("Useless parameter") != std::string::npos);
    BOOST_CHECK(user_algo.get_name().find("Null") != std::string::npos);
    // Evolve check (population does not change)
    rosenbrock user_prob{};
    population pop(user_prob, 20u);
    auto evolved_pop = user_algo.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        BOOST_CHECK(pop.get_x()[i] == evolved_pop.get_x()[i]);
        BOOST_CHECK(pop.get_f()[i] == evolved_pop.get_f()[i]);
        BOOST_CHECK(pop.get_ID()[i] == evolved_pop.get_ID()[i]);
    }
}

BOOST_AUTO_TEST_CASE(serialization_test)
{
    algorithm algo{null_algorithm{}};
    auto a = algo.extract<null_algorithm>()->get_a();
    // Now serialize, deserialize and compare the result.
    std::stringstream ss;
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(algo);
    }
    // Change the content of p before deserializing.
    algo = algorithm{de{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(algo);
    }
    auto after_a = algo.extract<null_algorithm>()->get_a();
    BOOST_CHECK_EQUAL(a, after_a);
}
