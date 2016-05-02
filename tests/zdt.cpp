#define BOOST_TEST_MODULE pagmo_zdt_test
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <exception>
#include <iostream>
#include <string>

#include "../include/problem.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/types.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(zdt1_test)
{
    zdt zdt1{5,2};
    vector_double x(35,1);
    print(zdt1.fitness(x));
}