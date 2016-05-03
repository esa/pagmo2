#define BOOST_TEST_MODULE pagmo_population_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <exception>
#include <iostream>
#include <string>

#include "../include/population.hpp"
#include "../include/problem.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/problems/null_problem.hpp"
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
}

BOOST_AUTO_TEST_CASE(population_push_back_test)
{
    // Create an empty population
    population pop{problem{zdt{1}}};
    // We fill it with a few individuals and check the size growth
    for (unsigned int i = 0u; i < 5u; ++i) {
        BOOST_CHECK(pop.size() == i);
        pop.push_back(vector_double(30, 0.5));
    }
    // We check the fitness counter
    BOOST_CHECK(pop.get_problem().get_fevals() == 5u);
}

BOOST_AUTO_TEST_CASE(population_random_decision_vector_test)
{
    // Create an empty population
    population pop{problem{zdt{1}}};
    // We fill it with a few individuals and check the size growth
    for (unsigned int i = 0u; i < 5u; ++i) {
        BOOST_CHECK(pop.size() == i);
        pop.push_back(vector_double(30, 0.5));
    }
    // We check the fitness counter
    BOOST_CHECK(pop.get_problem().get_fevals() == 5u);
}