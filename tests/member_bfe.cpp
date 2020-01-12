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

#define BOOST_TEST_MODULE member_bfe_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(basic_tests)
{
    BOOST_CHECK(is_udbfe<member_bfe>::value);

    bfe bfe0{member_bfe{}};
    BOOST_CHECK(bfe0.get_name() == "Member function batch fitness evaluator");
    BOOST_CHECK(bfe0.get_extra_info().empty());
    BOOST_CHECK_EQUAL(bfe0.get_thread_safety(), thread_safety::basic);

    // Verify a problem which does not have batch_fitness.
    problem p;
    BOOST_CHECK_EXCEPTION(bfe0(p, {1.}), not_implemented_error, [](const not_implemented_error &nie) {
        return boost::contains(nie.what(), "The batch_fitness() method has been invoked, but it "
                                           "is not implemented in a UDP of type 'Null problem'");
    });

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
            return vector_double(dvs.size(), 1.);
        }
    };
    p = problem{bf0{}};
    BOOST_CHECK(p.get_fevals() == 0u);
    BOOST_CHECK(bfe0(p, {1., 2., 3.}) == vector_double(3, 1.));
    // Check fevals counters.
    BOOST_CHECK(p.get_fevals() == 3u);

    // Double-check error checking in the bfe.
    // A UDP which provides batch_fitness(), but with wrong retval.
    struct bf2 {
        vector_double fitness(const vector_double &) const
        {
            return {0, 0};
        }
        std::pair<vector_double, vector_double> get_bounds() const
        {
            return {{0}, {1}};
        }
        vector_double batch_fitness(const vector_double &dvs) const
        {
            return vector_double(dvs.size(), 1.);
        }
        vector_double::size_type get_nobj() const
        {
            return 2;
        }
    };
    p = problem{bf2{}};
    BOOST_CHECK_EXCEPTION(bfe0(p, vector_double{1.}), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(),
                               "An invalid result was produced by a batch fitness evaluation: the length of "
                               "the vector representing the fitness vectors, 1, is not an exact multiple of the "
                               "fitness dimension of the problem, 2");
    });
}

BOOST_AUTO_TEST_CASE(s11n)
{
    bfe bfe0{member_bfe{}};
    BOOST_CHECK(bfe0.is<member_bfe>());
    // Store the string representation.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(bfe0);
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << bfe0;
    }
    // Change the content of p before deserializing.
    bfe0 = bfe{};
    BOOST_CHECK(!bfe0.is<member_bfe>());
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> bfe0;
    }
    auto after = boost::lexical_cast<std::string>(bfe0);
    BOOST_CHECK_EQUAL(before, after);
    BOOST_CHECK(bfe0.is<member_bfe>());
}
