#define BOOST_TEST_MODULE translate_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <stdexcept>
#include <string>

#include "../include/types.hpp"
#include "../include/io.hpp"
#include "../include/problems/translate.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(translate_construction_test)
{
    // First we check directly the two constructors
    problem p0{translate{}};
    problem p1{translate{null_problem{},{1}}};

    auto p0_string = boost::lexical_cast<std::string>(p0);
    auto p1_string = boost::lexical_cast<std::string>(p1);

    // We check that the default constructor constructs a problem
    // which has an identical representation to the problem
    // built by the explicit constructor.
    BOOST_CHECK(p0_string==p1_string);

    // We check that wrong size for translation results in an invalid_argument
    // exception
    BOOST_CHECK_THROW((translate{null_problem{},{1,2}}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(translate_functional_test)
{
    // Then we check that the hock_schittkowsky_71 problem is actually translated
    {
        problem p0{hock_schittkowsky_71{}};
        problem p1{translate{p0,{ 0.1,-0.2, 0.3, 0.4}}};
        problem p2{translate{p1,{-0.1, 0.2,-0.3,-0.4}}};
        vector_double x{3.,3.,3.,3.};
        // Fitness gradients and hessians are the same if the translation  is zero
        BOOST_CHECK(p0.fitness(x) == p2.fitness(x));
        BOOST_CHECK(p0.gradient(x) == p2.gradient(x));
        BOOST_CHECK(p0.hessians(x) == p2.hessians(x));
        // Bounds are unchanged if the translation is zero
        BOOST_CHECK(p0.get_bounds().first != p1.get_bounds().first);
        BOOST_CHECK(p0.get_bounds().first != p1.get_bounds().second);
        auto bounds0 = p0.get_bounds();
        auto bounds2 = p2.get_bounds();
        for (auto i = 0u; i < 4u; ++i) {
            BOOST_CHECK_CLOSE(bounds0.first[i], bounds2.first[i], 1e-13);
            BOOST_CHECK_CLOSE(bounds0.second[i], bounds2.second[i], 1e-13);
        }
        // We check that the problem's name has [translated] appended
        BOOST_CHECK(p1.get_name().find("[translated]") != std::string::npos);
        // We check that extra info has "Translation Vector:" somewhere"
        BOOST_CHECK(p1.get_extra_info().find("Translation Vector:") != std::string::npos);
        // We check we recover the translation vector
        auto translationvector = p1.extract<translate>()->get_translation();
        BOOST_CHECK((translationvector == vector_double{ 0.1,-0.2, 0.3, 0.4}));
    }
}

BOOST_AUTO_TEST_CASE(translate_serialization_test)
{
    // Do the checking with the full problem.
    hock_schittkowsky_71 p0{};
    problem p{translate{p0,{ 0.1,-0.2, 0.3, 0.4}}};
    // Call objfun, grad and hess to increase
    // the internal counters.
    p.fitness({1.,1.,1.,1.});
    p.gradient({1.,1.,1.,1.});
    p.hessians({1.,1.,1.,1.});
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

BOOST_AUTO_TEST_CASE(translate_stochastic_test)
{
    hock_schittkowsky_71 p0{};
    problem p{translate{p0,{ 0.1,-0.2, 0.3, 0.4}}};
    BOOST_CHECK(!p.is_stochastic());
}
