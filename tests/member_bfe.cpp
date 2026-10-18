/* Copyright 2017-2026 PaGMO development team

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

#include <cmath>
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

// A UDP which rewrites the decision vectors by snapping them to
// a grid with step 1/8, and evaluates the snapped vectors.
struct snap_udp {
    static vector_double snap(vector_double x)
    {
        for (auto &xi : x) {
            xi = std::round(xi * 8.) / 8.;
        }
        return x;
    }
    static vector_double eval(const vector_double &x)
    {
        return {x[0] + 2. * x[1], x[0] - x[1]};
    }
    vector_double fitness(const vector_double &x) const
    {
        return eval(snap(x));
    }
    vector_double fitness_rewrite(vector_double &x) const
    {
        x = snap(x);
        return eval(x);
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-1., -1.}, {1., 1.}};
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    thread_safety get_thread_safety() const
    {
        return m_ts;
    }
    thread_safety m_ts = thread_safety::basic;
};

// Same as above, with a batch_fitness() member function (which cannot rewrite).
struct snap_udp_batch : snap_udp {
    vector_double batch_fitness(const vector_double &dvs) const
    {
        vector_double retval;
        for (decltype(dvs.size()) i = 0; i < dvs.size(); i += 2u) {
            const auto fv = fitness(vector_double(dvs.data() + i, dvs.data() + i + 2u));
            retval.insert(retval.end(), fv.begin(), fv.end());
        }
        return retval;
    }
};

BOOST_AUTO_TEST_CASE(fitness_rewrite)
{
    BOOST_CHECK(!has_bfe_fitness_rewrite<member_bfe>::value);
    BOOST_CHECK(!bfe{member_bfe{}}.has_fitness_rewrite());
    problem p{snap_udp_batch{}};
    const vector_double dvs0{.1, .2, .3, .4};
    auto dvs = dvs0;
    BOOST_CHECK_EXCEPTION(bfe{member_bfe{}}.fitness_rewrite(p, dvs), not_implemented_error,
                          [](const not_implemented_error &nie) {
                              return boost::contains(nie.what(), "does not support in-place rewriting");
                          });
    BOOST_CHECK(dvs == dvs0);
    BOOST_CHECK_EQUAL(p.get_fevals(), 0u);
    // The call operator works as usual.
    BOOST_CHECK(bfe{member_bfe{}}(p, dvs0) == bfe{}(p, dvs0));
}
