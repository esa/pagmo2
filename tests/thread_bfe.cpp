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

#define BOOST_TEST_MODULE thread_bfe_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <initializer_list>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;

static std::mt19937 rng;

BOOST_AUTO_TEST_CASE(basic_tests)
{
    BOOST_CHECK(is_udbfe<thread_bfe>::value);

    bfe bfe0{thread_bfe{}};
    BOOST_CHECK(bfe0.get_name() == "Multi-threaded batch fitness evaluator");
    BOOST_CHECK(bfe0.get_extra_info().empty());
    BOOST_CHECK_EQUAL(bfe0.get_thread_safety(), thread_safety::basic);

    // Try with a problem providing the constant thread safety level.
    problem p0{rosenbrock{2}};
    // Rosenbrock has dimension 2, thus these are 5000 dvs.
    vector_double dvs(10000u);
    for (auto &x : dvs) {
        x = uniform_real_from_range(-1., 1., rng);
    }
    auto fvs = bfe0(p0, dvs);
    BOOST_CHECK_EQUAL(p0.get_fevals(), 5000u);
    vector_double tmp_dv(2u);
    for (decltype(dvs.size()) i = 0; i < dvs.size(); i += 2u) {
        tmp_dv[0] = dvs[i];
        tmp_dv[1] = dvs[i + 1u];
        BOOST_CHECK(fvs[i / 2u] == p0.fitness(tmp_dv)[0]);
    }

    // Try with a problem providing the basic thread safety level.
    p0 = problem{inventory{4}};
    for (auto &x : dvs) {
        x = uniform_real_from_range(0., 1., rng);
    }
    fvs = bfe0(p0, dvs);
    BOOST_CHECK_EQUAL(p0.get_fevals(), 2500u);
    tmp_dv.resize(4u);
    for (decltype(dvs.size()) i = 0; i < dvs.size(); i += 4u) {
        tmp_dv[0] = dvs[i];
        tmp_dv[1] = dvs[i + 1u];
        tmp_dv[2] = dvs[i + 2u];
        tmp_dv[3] = dvs[i + 3u];
        BOOST_CHECK(fvs[i / 4u] == p0.fitness(tmp_dv)[0]);
    }

    // A problem not providing any thread safety.
    struct unsafe_prob {
        vector_double fitness(const vector_double &) const
        {
            return vector_double{1.};
        }
        std::pair<vector_double, vector_double> get_bounds() const
        {
            return {{0.}, {1.}};
        }
        thread_safety get_thread_safety() const
        {
            return thread_safety::none;
        }
        std::string get_name() const
        {
            return "unsafe_prob";
        }
    };
    p0 = problem{unsafe_prob{}};
    BOOST_CHECK_EQUAL(p0.get_thread_safety(), thread_safety::none);
    BOOST_CHECK_EXCEPTION(bfe0(p0, dvs), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), "Cannot use a thread_bfe on the problem 'unsafe_prob', which does not "
                                          "provide the required level of thread safety");
    });
}

BOOST_AUTO_TEST_CASE(s11n)
{
    bfe bfe0{thread_bfe{}};
    BOOST_CHECK(bfe0.is<thread_bfe>());
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
    BOOST_CHECK(!bfe0.is<thread_bfe>());
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> bfe0;
    }
    auto after = boost::lexical_cast<std::string>(bfe0);
    BOOST_CHECK_EQUAL(before, after);
    BOOST_CHECK(bfe0.is<thread_bfe>());
}