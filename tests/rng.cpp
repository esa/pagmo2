#include "../include/rng.hpp"

#define BOOST_TEST_MODULE pagmo_rng_test
#include <boost/test/unit_test.hpp>

#include <vector>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(set_seed_and_next)
{
	// We check that the first N pseudo random numbers are identical if generated
	// right after the seed is set

	// We choose a seed
	details::random_engine_type::result_type seed{0};
	// Number of trials
	unsigned int N = 10000u;
	// We define two std::vectors that will contain two pseudorandom sequences
	std::vector<details::random_engine_type::result_type> prs1{N}, prs2{N};

	// We fill the first one
	random_device::set_seed(seed);
	for (auto i=0u; i < N; ++i) {
		prs1[i] = random_device::next();
	}
	// And the second one
	random_device::set_seed(seed);
	for (auto i=0u; i < N; ++i) {
		prs2[i]= random_device::next();
	}
	// We check that they are equal, since the seed was the same
	for (auto i=0u; i < N; ++i) {
		BOOST_CHECK_EQUAL(prs1[i], prs2[i]);  
	}
}

