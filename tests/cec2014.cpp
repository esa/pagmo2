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

#define BOOST_TEST_MODULE cec2014_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2014.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

#include <pagmo/io.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(cec2014_test)
{
    std::mt19937 r_engine(32u);
    // We check that all problems can be constructed at all dimensions and that the name returned makes sense
    // (only for dim =2 for speed). We also perform a fitness test (we only check no throws, not correctness)
    std::vector<unsigned int> allowed_dims = {2u, 10u, 20u, 30u, 50u, 100u};
    for (unsigned int i = 1u; i <= 30u; ++i) {
        for (auto dim : allowed_dims) {
            if (dim == 2
                && ((i >= 17u && i <= 22u) || (i >= 29u && i <= 30u))) { // Not all functions are defined for dim = 2
                continue;
            }
            cec2014 udp{i, dim};
            auto x = random_decision_vector({vector_double(dim, -100.), vector_double(dim, 100.)},
                                            r_engine); // a random vector
            BOOST_CHECK_NO_THROW(udp.fitness(x));
        }
        BOOST_CHECK((cec2014{i, 10u}.get_name().find("CEC2014 - f")) != std::string::npos);
    }
    // We check that wrong problem ids and dimensions cannot be constructed
    BOOST_CHECK_THROW((cec2014{0u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2014{29u, 2u}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2014{10u, 3u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(cec2014_correctness_test)
{
    std::unordered_map<unsigned, std::pair<unsigned, double>> results{
        {1, {100, 4604017218.1559124}},
        {2, {200, 16424929791.945568}},
        {3, {300, 8798332.5245634764}},
        {4, {400, 12017.897331937622}},
        {5, {500, 521.92704321874453}},
        {6, {600, 615.13507216412961}},
        {7, {700, 1119.3723738034998}},
        {8, {800, 984.24557115189464}},
        {9, {900, 1021.6476551540424}},
        {10, {1000, 3369.983857702578}},
        {11, {1100, 4016.4772158320311}},
        {12, {1200, 1211.0162141335773}},
        {13, {1300, 1308.0721648633023}},
        {14, {1400, 1466.1139987414285}},
        {15, {1500, 113563.20584342665}},
        {16, {1600, 1604.7838413642057}},
        {17, {1700, 33584263.0596224}},
        {18, {1800, 199405813.78039557}},
        {19, {1900, 3039.1757814055372}},
        {20, {2000, 824178075.74895775}},
        {21, {2100, 2675464151.9326577}},
        {22, {2200, 11523.440402324031}},
        {23, {2300, 2500}},
        {24, {2400, 2600}},
        {25, {2500, 2700}},
        {26, {2600, 2800}},
        {27, {2700, 2900}},
        {28, {2800, 3000}},
        {29, {2900, 3100}},
        {30, {3000, 3200}},
    };

    for (auto i = 1u; i <= 30u; ++i) {

        pagmo::cec2014 prob = pagmo::cec2014(i, 10u);
        auto x_min = prob.get_origin_shift();
        x_min.resize(10u); // uses only the first _dimensions_ elements since it will be longer for func_num > 23

        auto f_origin = prob.fitness(vector_double(10u, 0.))[0];
        auto f_min = prob.fitness(x_min)[0];

        BOOST_CHECK_EQUAL(f_min, results[i].first);
        BOOST_CHECK_CLOSE(f_origin, results[i].second, 1e-12); // Tolearnce EPS added to be safe
    }
}

BOOST_AUTO_TEST_CASE(cec2014_serialization_test)
{
    problem p{cec2014{1u, 10u}};
    // Call objfun to increase the internal counters.
    p.fitness(vector_double(10u, 0.));
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
