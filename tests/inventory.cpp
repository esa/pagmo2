#define BOOST_TEST_MODULE inventory_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <exception>
#include <iostream>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(inventory_test)
{
    // We check construction, seed setter and fitness determinism
    inventory prob(5, 5, 24);
    auto f1 = prob.fitness({3, 6, 7, 2, 2});
    prob.set_seed(23u);
    auto f2 = prob.fitness({3, 6, 7, 2, 2});
    prob.set_seed(24u);
    auto f3 = prob.fitness({3, 6, 7, 2, 2});
    BOOST_CHECK(f1 == f3);
    BOOST_CHECK(f1 != f2);

    // Checking bounds and objective dimensions
    BOOST_CHECK(prob.get_nobj() == 1u);
    BOOST_CHECK((prob.get_bounds()
                 == std::pair<vector_double, vector_double>({{0., 0., 0., 0., 0}, {200., 200., 200., 200., 200.}})));

    // Checking the name and extra info methods
    BOOST_CHECK(prob.get_name().find("Inventory") != std::string::npos);
    BOOST_CHECK(prob.get_extra_info().find("Weeks") != std::string::npos);
    BOOST_CHECK(prob.get_extra_info().find("Sample size") != std::string::npos);
    BOOST_CHECK(prob.get_extra_info().find("Seed") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(inventory_serialization_test)
{
    problem p{inventory{5u, 5u, 32u}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1., 1., 1., 1.});
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(p);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(p);
    }
    // Change the content of p before deserializing.
    p = problem{null_problem{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(p);
    }
    auto after = boost::lexical_cast<std::string>(p);
    BOOST_CHECK_EQUAL(before, after);
}
