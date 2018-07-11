/* Copyright 2017-2018 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#define BOOST_TEST_MODULE rng_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <thread>
#include <vector>

#include <pagmo/rng.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(set_seed_and_next)
{
    // We check that the first N pseudo random numbers are identical if generated
    // right after the same seed is set and different otherwise.

    // We choose two seeds
    unsigned seed{0u}, seed2{1u};

    // Length of the pseudo-random sequence tested
    unsigned int N = 10000u;

    // We generate three pseudo random sequences, two with the same seed
    random_device::set_seed(seed);
    std::vector<detail::random_engine_type::result_type> prs1;
    std::generate_n(std::back_inserter(prs1), N, random_device::next);

    random_device::set_seed(seed);
    std::vector<detail::random_engine_type::result_type> prs2;
    std::generate_n(std::back_inserter(prs2), N, random_device::next);

    random_device::set_seed(seed2);
    std::vector<detail::random_engine_type::result_type> prs3;
    std::generate_n(std::back_inserter(prs3), N, random_device::next);

    // We check that prs1 and prs2 are equal, since the seed was the same
    BOOST_CHECK(std::equal(prs1.begin(), prs1.end(), prs2.begin()));
    // We check that prs1 are prs3 are different since the seed was different
    BOOST_CHECK(!std::equal(prs1.begin(), prs1.end(), prs3.begin()));
}

// This test just runs calls to random_device::next() in two separate threads. If this executable
// is compiled with -fsanitize=thread in clang/gcc, it should check that the locking logic
// in random_device is correct.
BOOST_AUTO_TEST_CASE(data_races_test)
{
    unsigned int N = 10000u;
    std::vector<detail::random_engine_type::result_type> prs4, prs5;
    std::thread t1([&]() { std::generate_n(std::back_inserter(prs4), N, random_device::next); });
    std::thread t2([&]() { std::generate_n(std::back_inserter(prs5), N, random_device::next); });
    t1.join();
    t2.join();
}
