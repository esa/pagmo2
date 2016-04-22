#define BOOST_TEST_MODULE pagmo_null_problem_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include "../include/types.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/problems/null_problem.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(null_problem_test)
{
    // Problem instantiation
    problem p{null_problem{}};
    // Pick a few reference points
    vector_double x1 = {1};
    vector_double x2 = {2};
    // Fitness test
    BOOST_CHECK((p.fitness(x1) == vector_double{0,0,0}));
    BOOST_CHECK((p.fitness(x2) == vector_double{0,0,0}));
    // Gradient test
    BOOST_CHECK((p.gradient(x1) == vector_double{}));
    BOOST_CHECK((p.gradient(x2) == vector_double{}));
    // Gradient sparsity test
    auto gp = p.gradient_sparsity();
    BOOST_CHECK((gp == sparsity_pattern{}));
    // Hessians test
    auto hess1 = p.hessians(x1);
    BOOST_CHECK(hess1.size() == 3);
    BOOST_CHECK((hess1[0] == vector_double{}));
    BOOST_CHECK((hess1[1] == vector_double{}));
    BOOST_CHECK((hess1[2] == vector_double{}));
    // Hessians sparsity test
    auto sp = p.hessians_sparsity();
    BOOST_CHECK(sp.size() == 3);
    BOOST_CHECK((sp[0] == sparsity_pattern{}));
    BOOST_CHECK((sp[1] == sparsity_pattern{}));
    BOOST_CHECK((sp[2] == sparsity_pattern{}));
    // Name and extra info tests
    BOOST_CHECK(p.get_name().find("Null") != std::string::npos);
    BOOST_CHECK(p.get_extra_info().find("ficticious problem") != std::string::npos);
    // Best known test
    auto x_best = p.extract<null_problem>()->best_known();
    BOOST_CHECK(x_best[0] == 0);
}

BOOST_AUTO_TEST_CASE(null_problem_serialization_test)
{
    problem p{null_problem{}};
    // Call objfun, grad and hess to increase
    // the internal counters.
    p.fitness({1});
    p.gradient({1});
    p.hessians({1});
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