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

#define BOOST_TEST_MODULE cec2013_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2013.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(cec2013_test)
{
    std::mt19937 r_engine(32u);
    // We check that all problems can be constructed at all dimensions and that the name returned makes sense
    // (only for dim =2 for speed). We also perform a fitness test (we only check no throws, not correctness)
    std::vector<unsigned int> allowed_dims = {2u, 5u, 10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u, 90u, 100u};
    for (unsigned int i = 1u; i <= 28u; ++i) {
        for (auto dim : allowed_dims) {
            cec2013 udp{i, dim};
            auto x = random_decision_vector({vector_double(dim, -100.), vector_double(dim, 100.)},
                                            r_engine); // a random vector
            BOOST_CHECK_NO_THROW(udp.fitness(x));
        }
        BOOST_CHECK((cec2013{i, 2u}.get_name().find("CEC2013 - f")) != std::string::npos);
    }
    // We check that wrong problem ids and dimensions cannot be constructed
    BOOST_CHECK_THROW((cec2013{0u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2013{29u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2013{10u, 3u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(cec2013_serialization_test)
{
    problem p{cec2013{1u, 2u}};
    // Call objfun to increase the internal counters.
    p.fitness(vector_double(2u, 0.));
    // Store the string representation of p.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(p);
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(p);
    }
    // Change the content of p before deserializing.
    p = problem{null_problem{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(p);
    }
    auto after = boost::lexical_cast<std::string>(p);
    BOOST_CHECK_EQUAL(before, after);
}
