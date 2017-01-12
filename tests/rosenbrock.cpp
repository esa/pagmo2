#define BOOST_TEST_MODULE rosenbrock_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <exception>
#include <iostream>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(rosenbrock_test)
{
    // Problem construction
    rosenbrock ros2{2u};
    rosenbrock ros5{5u};
    BOOST_CHECK_THROW(rosenbrock{0u}, std::invalid_argument);
    BOOST_CHECK_THROW(rosenbrock{1u}, std::invalid_argument);
    BOOST_CHECK_NO_THROW(problem{rosenbrock{2u}});
    // Pick a few reference points
    vector_double x2 = {1., 1.};
    vector_double x5 = {1., 1., 1., 1., 1.};
    // Fitness test
    BOOST_CHECK((ros2.fitness({1., 1.}) == vector_double{0.}));
    BOOST_CHECK((ros5.fitness({1., 1., 1., 1., 1.}) == vector_double{0.}));
    // Bounds Test
    BOOST_CHECK((ros2.get_bounds() == std::pair<vector_double, vector_double>{{-5., -5.}, {10., 10.}}));
    // Name and extra info tests
    BOOST_CHECK(ros5.get_name().find("Rosenbrock") != std::string::npos);
    // Best known test
    auto x_best = ros2.best_known();
    BOOST_CHECK((x_best == vector_double{1., 1.}));
}

BOOST_AUTO_TEST_CASE(rosenbrock_serialization_test)
{
    problem p{rosenbrock{4u}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1., 1., 1.});
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
