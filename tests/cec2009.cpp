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

#define BOOST_TEST_MODULE cec2009_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2009.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;
using statics = detail::cec2009_statics<>;

BOOST_AUTO_TEST_CASE(cec2009_construction_test)
{
    // We check that all problems can be constructed
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, false};
    }
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, true};
    }
    // We check that wrong problem ids and dimensions cannot be constructed
    BOOST_CHECK_THROW((cec2009{0u}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2009{11u}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2009{1u, false, 0u}), std::invalid_argument);
    BOOST_CHECK_THROW((cec2009{8u, true, 0u}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(cec2009_fitness_test)
{
    std::mt19937 r_engine(32u);

    // We check that all problems return a fitness of the correct dims
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, false};
        auto x = random_decision_vector(udp.get_bounds(), r_engine); // a random vector
        auto f = udp.fitness(x);
        BOOST_CHECK_EQUAL(f.size(), udp.get_nobj());
        BOOST_CHECK((udp.get_name().find("CEC2009 - UF")) != std::string::npos);
    }
    // We check that all problems return a fitness of the correct dims
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, true};
        auto x = random_decision_vector(udp.get_bounds(), r_engine); // a random vector
        auto f = udp.fitness(x);
        BOOST_CHECK_EQUAL(f.size(), statics::m_nic[i - 1u] + udp.get_nobj());
        BOOST_CHECK((udp.get_name().find("CEC2009 - CF")) != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(cec2009_getters_test)
{
    // We check that all problems return a fitness
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, false};
        BOOST_CHECK((udp.get_name().find("CEC2009 - UF")) != std::string::npos);
        BOOST_CHECK((udp.get_nic() == 0u));
        BOOST_CHECK((problem(udp).get_nec() == 0u));
        BOOST_CHECK((udp.get_nobj() == statics::m_nobj[i - 1u]));
    }
    for (unsigned i = 1u; i <= 10u; ++i) {
        cec2009 udp{i, true};
        BOOST_CHECK((udp.get_name().find("CEC2009 - CF")) != std::string::npos);
        BOOST_CHECK((udp.get_nic() == statics::m_nic[i - 1u]));
        BOOST_CHECK((problem(udp).get_nec() == 0u));
        BOOST_CHECK((udp.get_nobj() == statics::m_nobj[i - 1u]));
    }
}

BOOST_AUTO_TEST_CASE(cec2009_serialization_test)
{
    problem p{cec2009{1u, false, 30u}};
    // Call objfun to increase the internal counters.
    p.fitness(vector_double(30u, 0.5));
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
