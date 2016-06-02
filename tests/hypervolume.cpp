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
void assertContribs(const std::vector<vector_double> &points, std::vector<double> &ref, std::vector<double> &answers) {
	hypervolume hv = hypervolume(points, true);
	BOOST_CHECK((hv.contributions(ref) == answers));
	for (unsigned int i = 0; i < answers.size(); i++) {
		BOOST_CHECK((hv.exclusive(i, ref) == answers[i]));
	}
}


BOOST_AUTO_TEST_CASE(hypervolume_compute_test)
{
	hypervolume hv;

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

	// points on the border of refpoint(4D)
	hv = hypervolume{ {1, 1, 1, 3},{2, 2, 2, 3} };
	BOOST_CHECK((hv.compute({ 3, 3, 3, 3 }) == 0));

	// 4d duplicate point
	hv = hypervolume{ { 1, 1, 1, 1 },{ 1, 1, 1, 1 } };
	BOOST_CHECK((hv.compute({ 2, 2, 2, 2 }) == 1));

	// 4d duplicate and dominated
	hv = hypervolume{ { 1.0, 1.0, 1.0, 1.0 },{ 1.0, 1.0, 1.0, 1.0 },{0.0,0.0,0.0,0.0} };
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

}

BOOST_AUTO_TEST_CASE(hypervolume_contributions_test) {
	// Tests for contributions and exclusive hypervolumes
	std::vector<vector_double> points;
	std::vector<double> ref;
	std::vector<double> answers;

	/*  This test contains a front with 3 non dominated points,
		and many dominated points. Most of the dominated points
		lie on edges of the front, which makes their exclusive contribution
		equal to 0.*/
	points = { { 1, 6.5 },{ 1, 6 },{ 1, 5 },{ 2, 5 },{ 3, 5 },{ 3, 3 },{ 4, 6.5 },
	{ 4.5, 4 },{ 5, 3 },{ 5, 1.5 },{ 7, 1.5 },{ 7, 3.5 }, };
	ref = { 7.0, 6.5, };
	answers = { 0.0, 0.0, 1.0, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, };
	assertContribs(points, ref, answers);

	// same test with duplicates and points on the edge of the ref-point
	points = { { 1, 6.5 },{ 1, 6 },{ 1, 5 },{ 2, 5 },{ 3, 5 },{ 3, 3 },{ 4, 6.5 },
	{ 4.5, 4 },{ 5, 3 },{ 5, 1.5 },{ 7, 1.5 },{ 7, 3.5 },{ 7, 0.5 },{ 7, 1.0 },{ 7, 4.5 },
	{ 0.0, 6.5 },{ 5.5, 6.5 },{ 7, 0.5 },{ 5.5, 6.5 },{ 5, 5 },{ 5, 5 },{ 5, 5 } };
	ref = { 7.0, 6.5, };
	answers = { 0.0, 0.0, 1.0, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	assertContribs(points, ref, answers);

	// Gradually adding duplicate points to the set, making sure the contribution change accordingly.
	points = { {1, 1} };
	ref = { 2, 2 };
	answers = { 1.0 };
	assertContribs(points, ref, answers);

	points.push_back({ 1, 1 });
	answers = { 0.0, 0.0 };
	assertContribs(points, ref, answers);

	points.push_back({ 1, 1 });
	answers = { 0.0, 0.0, 0.0 };
	assertContribs(points, ref, answers);

	points.push_back({ 0.5, 0.5 });
	answers = { 0.0, 0.0, 0.0, 1.25 };
	assertContribs(points, ref, answers);

	points.push_back({ 0.5, 0.5 });
	answers = { 0.0, 0.0, 0.0, 0.0, 0.0 };
	assertContribs(points, ref, answers);

	// Next test contains a tricky front in 3D with some weakly dominated points on the "edges" of the bounding box.
	// Non - tricky base problem
	points = { { -6, -1, -6 },{ -1, -3, -5 },{ -3, -4, -4 },
	{ -4, -2, -3 },{ -5, -5, -2 },{ -2, -6, -1 } };
	ref = { 0, 0, 0 };
	answers = { 18, 2, 12, 1, 18, 2 };
	assertContribs(points, ref, answers);

	// Add some points that contribute nothing and do not alter other
	points = { { -6, -1, -6 },{ -1, -3, -5 },{ -3, -4, -4 },
	{ -4, -2, -3 },{ -5, -5, -2 },{ -2, -6, -1 }, {-3, -1, -3},{ -1, -1, -5 },{ -1, -2, -4 },
	{ -1, -3, -4 },{ -7, -7, 0 },{ 0, -5, -5 },{ -7, 0, -7 } };
	answers = { 18, 2, 12, 1, 18, 2, 0, 0, 0, 0, 0, 0, 0 };
	assertContribs(points, ref, answers);


	//	Gradually adding points, some of which are dominated or duplicates.
	//	Tests whether contributions and repeated exclusive method produce the same results.
	points = { {3, 3, 3} };
	ref = { 5, 5, 5 };
	answers = { 8.0 };
	assertContribs(points, ref, answers);

	// Decrease the contribution of first point.Second point is dominated.
	points.push_back({ 4, 4, 4 });
	answers = { 7, 0, };
	assertContribs(points, ref, answers);

	// Add duplicate point
	points.push_back({ 3, 3, 3 });
	answers = { 0, 0, 0 };
	assertContribs(points, ref, answers);

	points.push_back({ 3, 3, 2 });
	answers = { 0, 0, 0, 4 };
	assertContribs(points, ref, answers);

	points.push_back({ 3,3,1 });
	answers = { 0, 0, 0, 0, 4 };
	assertContribs(points, ref, answers);

	//	Combine extreme points together. Mixing small and large contributions in a single front
	points = { {-1, -1, -1}, { -1, -1, -1 }, { -1, -1, -1 } };
	ref = { 0, 0, 0 };
	answers = { 0, 0, 0 };
	assertContribs(points, ref, answers);
	

	// Adding a point far away
	points.push_back({-1000,-1000,-1000});
	answers = { 0, 0, 0, 999999999 };
	assertContribs(points, ref, answers);

	// Adding an even further point
	points.push_back({ -10000,-10000,-10000 });
	answers = { 0, 0, 0, 0, 999000000000 };
	assertContribs(points, ref, answers);

	//	Gradually adding points in 4d.	Tests whether contributions and repeated exclusive methods produce the same results.
	points = { {1, 1, 1, 1} };
	ref = { 5, 5, 5, 5 };
	answers = { 256 };
	assertContribs(points, ref, answers);

	points.push_back({ 4,4,4,4 });
	answers = { 255, 0 };
	assertContribs(points, ref, answers);

	points.push_back({ 3,3,3,3 });
	answers = { 240, 0, 0};
	assertContribs(points, ref, answers);

	points.push_back({ 1,1,1,1 });
	answers = { 0, 0, 0, 0 };
	assertContribs(points, ref, answers);

	//	Gradually adding points in 5d.	Tests whether contributions and repeated exclusive methods produce the same results.
	points = { { 1, 1, 1, 1, 1 } };
	ref = { 5, 5, 5, 5, 5 };
	answers = { 1024 };
	assertContribs(points, ref, answers);

	points.push_back({ 4,4,4,4,4});
	answers = { 1023, 0 };
	assertContribs(points, ref, answers);

	points.push_back({ 3,3,3,3,3 });
	answers = { 992, 0, 0 };
	assertContribs(points, ref, answers);

	points.push_back({ 1,1,1,1,1 });
	answers = { 0, 0, 0, 0 };
	assertContribs(points, ref, answers);

}
