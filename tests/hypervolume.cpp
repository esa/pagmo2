#define BOOST_TEST_MODULE hypervolume_utilities_test
#include <boost/test/unit_test.hpp>
#include <exception>
#include <tuple>

#include "../include/utils/hypervolume.hpp"
#include "../include/population.hpp"
#include "../include/problem.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/problems/rosenbrock.hpp"
//#include "../include/utils/hv_algorithms/hv_algorithm.hpp"
//#include "../include/utils/hv_algorithms/hv2d.hpp"

using namespace pagmo;

BOOST_AUTO_TEST_CASE(hypervolume_construction_test)
{
	// test standard construction


/*




# good point definition with extra arg (hypervolume(ps, "something
# extra")
self.assertRaises(
TypeError, hypervolume, [[1, 2, 3], [2, 3, 4], [2, 3, 5]],
"extra arg")
self.assertRaises(
TypeError, hypervolume, [[1, 2, 3], [2, 3, 4], "foo"])  # bad point
# bad point value
self.assertRaises(
TypeError, hypervolume, [[1, 2, 3], [2, 3, 4], [2, 3, "bar"]])
# skipping argument: hypervolume() should raise TypeError
self.assertRaises(TypeError, hypervolume)

self.assertRaises(
TypeError, hypervolume, foo=self.good_ps_2d)  # bad kwarg
self.assertRaises(
TypeError, hypervolume, self.good_ps_2d, foo="bar")  # extra kwarg
*/
}


BOOST_AUTO_TEST_CASE(hypervolume_test)
{
	//auto inf = std::numeric_limits<double>::infinity();
	//auto big = std::numeric_limits<double>::max();

	// by vector
	std::vector<vector_double> x1{ { 1,2 },{ 3,4 } };
	hypervolume hv1 = hypervolume(x1, true);
	BOOST_CHECK(hv1.get_points() == x1);
	BOOST_CHECK((hv1.get_refpoint(1.0) == vector_double{ 4.0, 5.0 }));

	// initilization list constructor
	hypervolume hv2{ { 6,4 },{ 3,5 } };
	std::vector<vector_double> x2{ { 6,4 },{ 3,5 } };
	BOOST_CHECK((hv2.get_points() == x2));
	BOOST_CHECK((hv2.get_refpoint(0.0) == vector_double{ 6.0, 5.0 }));

	// by population
	population pop1{ problem{ zdt{ 1,5 } }, 2 };
	hypervolume hv3 = hypervolume(pop1, true);

	// errors
	population pop2{ problem{ rosenbrock(10) }, 2 };
	hypervolume hv4;
	BOOST_CHECK_THROW(hypervolume(pop2, true), std::invalid_argument);

	// computation of hypervolume indicator
	hypervolume hv5 = hypervolume{ {1, 2},{2, 1} };
	BOOST_CHECK((hv5.compute({ 3,3 }) == 3));

	hypervolume hv6 = { {1, 1, 1},{2, 2, 2,} };
	BOOST_CHECK((hv6.compute({ 3, 3, 3 }) == 8));


	/*# simple 2D test

		self.assertEqual(hv.compute(r = [3, 3]), 3)

		# point on the border of refpoint(2D)
		hv = hypervolume([[1, 2], [2, 1]])
		self.assertEqual(hv.compute([2, 2]), 0)*/


}
