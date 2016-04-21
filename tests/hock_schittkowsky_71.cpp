#define BOOST_TEST_MODULE pagmo_hs_test
#include <boost/test/unit_test.hpp>

#include "../include/types.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(hock_schittkowsky_71_test)
{
    // Problem instantiation
    problem p{hock_schittkowsky_71{}};
    // Pick a few reference points
    vector_double x1 = {1., 1., 1., 1.};
    vector_double x2 = {2., 2., 2., 2.};
    vector_double x3 = {3., 3., 3., 3.};
    //Fitness test
    BOOST_CHECK((p.fitness(x1) == vector_double{4, -36, 24}));
    BOOST_CHECK((p.fitness(x2) == vector_double{26, -24, 9}));
}
