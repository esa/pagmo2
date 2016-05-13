#define BOOST_TEST_MODULE pagmo_population_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <exception>
#include <iostream>
#include <string>

#include "../include/population.hpp"
#include "../include/problem.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/types.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(population_construction_test)
{
    unsigned int seed = 123;
    population pop1{};
    population pop2{problem{zdt{1,5}}, 2, seed};
    population pop3{problem{zdt{2,5}}, 2, seed};

    // We check that the number of individuals is as expected
    BOOST_CHECK(pop1.size() == 0u);
    BOOST_CHECK(pop2.size() == 2u);
    BOOST_CHECK(pop3.size() == 2u);
    // We check population's individual chromosomes and IDs are the same
    // as the random seed was (and the problem dimension), while
    // fitness vectors were different as the problem is
    BOOST_CHECK(pop2.get_ID() == pop3.get_ID());
    BOOST_CHECK(pop2.get_x() == pop3.get_x());
    BOOST_CHECK(pop2.get_f() != pop3.get_f());
    // We check that the seed has been set correctly
    BOOST_CHECK(pop2.get_seed() == seed);

    // We test the generic constructor
    population pop4{zdt{2,5}, 2, seed};
    BOOST_CHECK(pop4.get_ID() == pop3.get_ID());
    BOOST_CHECK(pop4.get_x() == pop3.get_x());
    BOOST_CHECK(pop4.get_f() == pop3.get_f());
    population pop5{zdt{1,5}, 2, seed};
    BOOST_CHECK(pop2.get_ID() == pop5.get_ID());
    BOOST_CHECK(pop2.get_x() == pop5.get_x());
    BOOST_CHECK(pop2.get_f() == pop5.get_f());
}

BOOST_AUTO_TEST_CASE(population_copy_constructor_test)
{
    population pop1{problem{rosenbrock{5}}, 10u};
    population pop2(pop1);
    BOOST_CHECK(pop2.get_ID() == pop1.get_ID());
    BOOST_CHECK(pop2.get_x() == pop1.get_x());
    BOOST_CHECK(pop2.get_f() == pop1.get_f());
}

BOOST_AUTO_TEST_CASE(population_push_back_test)
{
    // Create an empty population
    population pop{problem{zdt{1}}};
    // We fill it with a few individuals and check the size growth
    for (unsigned int i = 0u; i < 5u; ++i) {
        BOOST_CHECK(pop.size() == i);
        BOOST_CHECK(pop.get_f().size() == i);
        BOOST_CHECK(pop.get_x().size() == i);
        BOOST_CHECK(pop.get_ID().size() == i);
        pop.push_back(vector_double(30, 0.5));
    }
    // We check the fitness counter
    BOOST_CHECK(pop.get_problem().get_fevals() == 5u);
}

BOOST_AUTO_TEST_CASE(population_decision_vector_test)
{
    // Create an empty population
    population pop{problem{null_problem{}}};
    auto bounds = pop.get_problem().get_bounds();
    // Generate a random decision_vector
    auto x = pop.decision_vector();
    // Check that the decision_vector is indeed within bounds
    for (decltype(x.size()) i = 0u; i < x.size(); ++i) {
        BOOST_CHECK(x[i] < bounds.second[i]);
        BOOST_CHECK(x[i] >= bounds.first[i]);
    }
}

BOOST_AUTO_TEST_CASE(population_champion_test)
{
    // Test throw
    {
        population pop{problem{zdt{}}, 2};
        BOOST_CHECK_THROW(pop.champion(), std::invalid_argument);
    }
    // Test on single objective
    {
        population pop{problem{rosenbrock{2}}};
        pop.push_back({0.5,0.5});
        pop.push_back({0.3,0.1});
        pop.push_back(pop.get_problem().extract<rosenbrock>()->best_known());
        pop.push_back({-0.5,0.5});
        BOOST_CHECK(pop.champion() == 2u);
    }
    // Test on constrained
    {
        population pop{problem{hock_schittkowsky_71{}}};
        pop.push_back({1.5,1.5,1.5,1.5});
        pop.push_back({1.3,1.1,2.3,3.4});
        pop.push_back(pop.get_problem().extract<hock_schittkowsky_71>()->best_known());
        pop.push_back({2.5,1.5,3.4,3.3});
        BOOST_CHECK(pop.champion(1e-5) == 2u); // tolerance matters here!!
    }
}

BOOST_AUTO_TEST_CASE(population_setters_test)
{
    population pop{problem{null_problem{}}, 2};
    // Test throw
    BOOST_CHECK_THROW(pop.set_xf(2, {3}, {1,2,3}), std::invalid_argument);// index invalid
    BOOST_CHECK_THROW(pop.set_xf(2, {3}, {1,2,3}), std::invalid_argument);// chromosome invalid
    BOOST_CHECK_THROW(pop.set_xf(2, {3}, {1,2}), std::invalid_argument);  // fitness invalid
    // Test set_xf
    pop.set_xf(0,{3},{1,2,3});
    BOOST_CHECK((pop.get_x()[0] == vector_double{3}));
    BOOST_CHECK((pop.get_f()[0] == vector_double{1,2,3}));
    // Test set_x
    pop.set_x(0,{1.2});
    BOOST_CHECK((pop.get_x()[0] == vector_double{1.2}));
    BOOST_CHECK(pop.get_f()[0] == pop.get_problem().fitness({1.2})); // works as counters are marked mutable
}

BOOST_AUTO_TEST_CASE(population_getters_test)
{
    population pop{problem{null_problem{}}, 1, 1234u};
    pop.set_xf(0,{3},{1,2,3});
    // Test
    BOOST_CHECK(pop.get_problem().get_name() == "Null problem");
    BOOST_CHECK((pop.get_f()[0] == vector_double{1,2,3}));
    BOOST_CHECK(pop.get_seed() == 1234u);
    BOOST_CHECK_NO_THROW(pop.get_ID());
    // Streaming operator is tested to contain the problem stream
    auto pop_string = boost::lexical_cast<std::string>(pop);
    auto prob_string = boost::lexical_cast<std::string>(pop.get_problem());
    BOOST_CHECK(pop_string.find(prob_string) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(population_serialization_test)
{
    population pop{problem{null_problem{}}, 30, 1234u};
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(pop);
    // Now serialize, deserialize and compare the result.
    {
    cereal::JSONOutputArchive oarchive(ss);
    oarchive(pop);
    }
    // Change the content of p before deserializing.
    pop = population{problem{zdt{5, 20u}}, 30};
    {
    cereal::JSONInputArchive iarchive(ss);
    iarchive(pop);
    }
    auto after = boost::lexical_cast<std::string>(pop);
    BOOST_CHECK_EQUAL(before, after);
}
