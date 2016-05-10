#define BOOST_TEST_MODULE pagmo_inventory_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <exception>
#include <iostream>
#include <string>

#include "../include/problem.hpp"
#include "../include/problems/inventory.hpp"
#include "../include/types.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(inventory_test)
{
    inventory prob;
    print(prob.fitness({20,21,22,23}));
    prob.set_seed(23u);
    print(prob.fitness({20,21,22,23}));
    print(problem{prob});
}
