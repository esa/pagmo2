#define BOOST_TEST_MODULE hypervolume_utilities_test
#include <boost/test/unit_test.hpp>
#include <exception>
#include <tuple>

#include "../include/detail/hypervolume_all.hpp"
#include "../include/population.hpp"
#include "../include/problem.hpp"
#include "../include/types.hpp"
#include "../include/io.hpp"
#include "../include/problems/zdt.hpp"
#include "../include/problems/rosenbrock.hpp"

using namespace pagmo;
/**
 * Assertion method that tests correct computation of contributions for the whole contribution method
 * and the single exclusive method.
 */
/*void assertContribs(const std::vector<vector_double> &points, std::vector<double> &ref, std::vector<double> &answers) {
	hypervolume hv = hypervolume(points, true);
	BOOST_CHECK((hv.contributions(1.0) == answers));

	self.assertEqual(hv.contributions(R), ans)
			self.assertEqual(tuple(hv.exclusive(i, R) for i in range(len(S))), ans)
}*/


BOOST_AUTO_TEST_CASE(hypervolume_test)
{
	//auto inf = std::numeric_limits<double>::infinity();
	//auto big = std::numeric_limits<double>::max();
	hypervolume hv;

	// error threshold
	const double eps = 10e-8;

	// by vector
	std::vector<vector_double> x1{ { 1,2 },{ 3,4 } };
	hv = hypervolume(x1, true);
	BOOST_CHECK(hv.get_points() == x1);
	BOOST_CHECK((hv.get_refpoint(1.0) == vector_double{ 4.0, 5.0 }));

	// by list constructor
	hv = hypervolume{ { 6,4 },{ 3,5 } };
	std::vector<vector_double> x2{ { 6,4 },{ 3,5 } };
	BOOST_CHECK((hv.get_points() == x2));
	BOOST_CHECK((hv.get_refpoint(0.0) == vector_double{ 6.0, 5.0 }));

	// by population
	population pop1{ problem{ zdt{ 1,5 } }, 2 };
	hv = hypervolume(pop1, true);

	// errors
	population pop2{ problem{ rosenbrock(10) }, 2 };
	BOOST_CHECK_THROW(hypervolume(pop2, true), std::invalid_argument);

	// 2d computation of hypervolume indicator
	hv = hypervolume{ {1, 2},{2, 1} };
	BOOST_CHECK((hv.compute({ 3,3 }) == 3));
	
	// point on the border of refpoint(2D)
	BOOST_CHECK((hv.compute({ 2,2 }) == 0));

	// 3d computation of hypervolume indicator
	hv = hypervolume{ {1, 1, 1},{2, 2, 2,} };
	BOOST_CHECK((hv.compute({ 3, 3, 3 }) == 8));

	// points on the border of refpoint(3D)
	hv = hypervolume{ {1, 2, 1},{2, 1, 1} };
	BOOST_CHECK((hv.compute({ 2, 2, 2 }) == 0));

	// 4d computation of hypervolume indicator
	hv = hypervolume{ {1, 1, 1, 1},{2, 2, 2, 2} };
	BOOST_CHECK((hv.compute({ 3, 3, 3, 3 }) == 16));
	BOOST_CHECK(fabs(hv.compute({ 3, 3, 3, 3 }) - 16) < eps);

	// points on the border of refpoint(4D)
	hv = hypervolume{ {1, 1, 1, 3},{2, 2, 2, 3} };
	BOOST_CHECK((hv.compute({ 3, 3, 3, 3 }) == 0));

	// 4d duplicate point
	hv = hypervolume{ { 1, 1, 1, 1 },{ 1, 1, 1, 1 } };
	BOOST_CHECK((hv.compute({ 2, 2, 2, 2 }) == 1));

	// 4d duplicate and dominated
	hv = hypervolume{ { 1.0, 1.0, 1.0, 1.0 },{ 1.0, 1.0, 1.0, 1.0 },{0.0,0.0,0.0,0.0} };
	for (int i = 0; i < 10; ++i) {
		std::cout << hv.compute({ 2.0, 2.0, 2.0, 2.0 }) << std::endl;
	}
	BOOST_CHECK((hv.compute({ 2.0, 2.0, 2.0, 2.0 }) == 16.0));

	// tests for invalid reference points
	hv = hypervolume{ {1, 3},{2, 2},{3, 1} };
	// equal to some other point
	BOOST_CHECK_THROW(hv.compute({ 3,1 }), std::invalid_argument);
	// refpoint dominating some points
	BOOST_CHECK_THROW(hv.compute({ 1.5,1.5 }), std::invalid_argument);
	// refpoint dominating all points
	BOOST_CHECK_THROW(hv.compute({ 0, 0 }), std::invalid_argument);

	// Calling specific algorithms
	std::shared_ptr<hv_algorithm> hv_algo_2d = hv2d().clone();
	std::shared_ptr<hv_algorithm> hv_algo_3d = hv3d().clone();
	std::shared_ptr<hv_algorithm> hv_algo_nd = hvwfg().clone();

	hv = hypervolume{ {2.3, 4.5},{3.4, 3.4},{6.0, 1.2} };
	BOOST_CHECK((hv.compute({ 7.0, 7.0 }) == 17.91));
	BOOST_CHECK((hv.compute({ 7.0, 7.0 }, hv_algo_2d) == 17.91));
	BOOST_CHECK_THROW(hv.compute({ 7.0, 7.0 }, hv_algo_3d), std::invalid_argument);
	BOOST_CHECK((hv.compute({ 7.0, 7.0 }, hv_algo_nd) == 17.91));

	hv = hypervolume{ { 2.3, 4.5, 3.2 },{ 3.4, 3.4, 3.4 },{ 6.0, 1.2, 3.6 } };
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0 }) == 66.386));
	BOOST_CHECK_THROW(hv.compute({ 7.0, 7.0, 7.0 }, hv_algo_2d), std::invalid_argument);
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0 }, hv_algo_3d) == 66.386));
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0 }, hv_algo_nd) == 66.386));

	hv = hypervolume{ { 2.3, 4.5, 3.2 },{ 3.4, 3.4, 3.4 },{ 6.0, 1.2, 3.6 } };
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0 }) == 66.386));
	BOOST_CHECK_THROW(hv.compute({ 7.0, 7.0, 7.0 }, hv_algo_2d), std::invalid_argument);
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0 }, hv_algo_3d) == 66.386));
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0 }, hv_algo_nd) == 66.386));

	hv = hypervolume{ {2.3, 4.5, 3.2, 1.9, 6.0},{3.4, 3.4, 3.4, 2.1, 5.8},{6.0, 1.2, 3.6, 3.0, 6.0} };
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0, 7.0, 7.0 }) == 373.21228));
	BOOST_CHECK_THROW(hv.compute({ 7.0, 7.0, 7.0, 7.0, 7.0 }, hv_algo_2d), std::invalid_argument);
	BOOST_CHECK_THROW(hv.compute({ 7.0, 7.0, 7.0, 7.0, 7.0 }, hv_algo_3d), std::invalid_argument);
	BOOST_CHECK((hv.compute({ 7.0, 7.0, 7.0, 7.0, 7.0 }, hv_algo_nd) == 373.21228));
	std::cout << hv.compute({ 7.0, 7.0, 7.0, 7.0, 7.0 }) << std::endl;




}
