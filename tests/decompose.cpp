#define BOOST_TEST_MODULE pagmo_translate_test
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <stdexcept>
#include <string>

#include "../include/types.hpp"
#include "../include/io.hpp"
#include "../include/problems/decompose.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/problems/zdt.hpp"


using namespace pagmo;

BOOST_AUTO_TEST_CASE(decompose_construction_test)
{
    // First we check directly the two constructors
    decompose p0{};
    problem p{decompose(zdt{1u,5u}, {0.5,0.5}, {0., 0.})};
    print(p);
}
