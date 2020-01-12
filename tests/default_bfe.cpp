/* Copyright 2017-2020 PaGMO development team

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

#define BOOST_TEST_MODULE default_bfe_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// UDP which implements batch_fitness.
struct bf0 {
    vector_double fitness(const vector_double &) const
    {
        return {0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    vector_double batch_fitness(const vector_double &dvs) const
    {
        ++s_counter;
        return vector_double(dvs.size(), 1.);
    }
    static unsigned s_counter;
};

unsigned bf0::s_counter = 0;

// UDP without batch_fitness, but supporting thread_bfe.
struct bf1 {
    vector_double fitness(const vector_double &) const
    {
        return {0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
};

// UDP without batch_fitness, and not thread-safe.
struct bf2 {
    vector_double fitness(const vector_double &) const
    {
        return {0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
    std::string get_name() const
    {
        return "baffo";
    }
};

BOOST_AUTO_TEST_CASE(basic_tests)
{
    BOOST_CHECK(is_udbfe<default_bfe>::value);

    bfe bfe0{};
    BOOST_CHECK(bfe0.is<default_bfe>());
    BOOST_CHECK(bfe0.get_name() == "Default batch fitness evaluator");
    BOOST_CHECK(bfe0.get_extra_info().empty());
    BOOST_CHECK_EQUAL(bfe0.get_thread_safety(), thread_safety::basic);

    // Use UDP member function.
    problem p{bf0{}};
    BOOST_CHECK(p.get_fevals() == 0u);
    BOOST_CHECK(bfe0(p, {1., 2., 3.}) == vector_double(3, 1.));
    // Check fevals counters.
    BOOST_CHECK(p.get_fevals() == 3u);
    BOOST_CHECK(bf0::s_counter == 1u);

    // UDP without batch_fitness() member function, but supporting thread_bfe.
    p = problem{bf1{}};
    BOOST_CHECK(p.get_fevals() == 0u);
    BOOST_CHECK(bfe0(p, {1., 2., 3.}) == vector_double(3, 0.));
    // Check fevals counters.
    BOOST_CHECK(p.get_fevals() == 3u);

    // UDP without batch_fitness() member function, which is not thread-safe enough.
    p = problem{bf2{}};
    BOOST_CHECK(p.get_thread_safety() == thread_safety::none);
    BOOST_CHECK_EXCEPTION(bfe0(p, {1.}), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(),
                               "Cannot execute fitness evaluations in batch mode for a problem of type 'baffo': the "
                               "problem does not implement the batch_fitness() member function, and its thread safety "
                               "level is not sufficient to run a thread-based batch fitness evaluation implementation");
    });
}

BOOST_AUTO_TEST_CASE(s11n)
{
    bfe bfe0{default_bfe{}};
    BOOST_CHECK(bfe0.is<default_bfe>());
    // Store the string representation.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(bfe0);
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << bfe0;
    }
    // Change the content of p before deserializing.
    bfe0 = bfe{member_bfe{}};
    BOOST_CHECK(!bfe0.is<default_bfe>());
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> bfe0;
    }
    auto after = boost::lexical_cast<std::string>(bfe0);
    BOOST_CHECK_EQUAL(before, after);
    BOOST_CHECK(bfe0.is<default_bfe>());
}
