#include "../include/rng.hpp"

#define BOOST_TEST_MODULE pagmo_rng_test
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(set_seed_and_next)
{
	// We check that the first N pseudo random numbers are identical if generated
	// right after the seed is set

	// We choose two seeds 
	details::random_engine_type::result_type seed{0u};
	details::random_engine_type::result_type seed2{1u};
	// Length of the pseudo-random sequence tested 
	unsigned int N = 10000u;

	random_device::set_seed(seed);
	std::vector<details::random_engine_type::result_type> prs1;
	std::generate_n(std::back_inserter(prs1),N,random_device::next);	

	random_device::set_seed(seed);
	std::vector<details::random_engine_type::result_type> prs2;
	std::generate_n(std::back_inserter(prs2),N,random_device::next);

	random_device::set_seed(seed2);
	std::vector<details::random_engine_type::result_type> prs3;
	std::generate_n(std::back_inserter(prs3),N,random_device::next);

	// We check that prs1 and prs2 are equal, since the seed was the same
	BOOST_CHECK(std::equal(prs1.begin(),prs1.end(),prs2.begin()));
	// We check that prs1 are prs3 are different since the seed was different
	BOOST_CHECK(!std::equal(prs1.begin(),prs1.end(),prs3.begin()));
}

