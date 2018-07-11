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

#define BOOST_TEST_MODULE luksan_vlcek1_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/luksan_vlcek1.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/gradients_and_hessians.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(luksan_vlcek1_test)
{
    // 1 - Construction
    BOOST_CHECK_NO_THROW(luksan_vlcek1{3});
    BOOST_CHECK_NO_THROW(problem{luksan_vlcek1{3}});
    BOOST_CHECK_THROW(luksan_vlcek1{2}, std::invalid_argument);
    BOOST_CHECK_THROW(problem{luksan_vlcek1{1}}, std::invalid_argument);
    problem prob{luksan_vlcek1{3}};
    BOOST_CHECK_EQUAL(prob.get_nic(), 0u);
    BOOST_CHECK_EQUAL(prob.get_nec(), 1u);
    BOOST_CHECK_EQUAL(prob.get_nobj(), 1u);
    BOOST_CHECK_EQUAL(prob.gradient_sparsity().size(), 6u);
    problem prob2{luksan_vlcek1{100}};
    BOOST_CHECK_EQUAL(prob2.get_nic(), 0u);
    BOOST_CHECK_EQUAL(prob2.get_nec(), 98u);
    BOOST_CHECK_EQUAL(prob2.get_nobj(), 1u);
    BOOST_CHECK_EQUAL(prob2.gradient_sparsity().size(), 394u);
    {
        auto sp = estimate_sparsity([prob](const vector_double &x) { return prob.fitness(x); }, {0.1, 0.2, 0.3}, 1e-8);
        BOOST_CHECK(prob.gradient_sparsity() == sp);
    }
    {
        auto sp = estimate_sparsity([prob2](const vector_double &x) { return prob2.fitness(x); },
                                    vector_double(100, 0.1), 1e-8);
        BOOST_CHECK(prob2.gradient_sparsity() == sp);
    }
    auto res = prob.gradient({1., 2., 3.});
    auto gh = estimate_gradient_h([prob](const vector_double &x) { return prob.fitness(x); }, {1., 2., 3.}, 1e-2);
    for (unsigned i = 0; i < res.size(); ++i) {
        BOOST_CHECK_CLOSE(gh[i], res[i], 1e-8);
    }
    res = prob2.gradient(vector_double(100, 0.1));
    gh = estimate_gradient_h([prob2](const vector_double &x) { return prob2.fitness(x); }, vector_double(100, 0.1),
                             1e-2);
    auto ghs = estimate_sparsity([prob2](const vector_double &x) { return prob2.fitness(x); }, vector_double(100, 0.1),
                                 1e-8);
    unsigned counter = 0u;
    for (auto &pair : ghs) {
        auto i = pair.first;
        auto j = pair.second;
        BOOST_CHECK_CLOSE(gh[i * 100 + j], res[counter], 1e-8);
        ++counter;
    }
}

BOOST_AUTO_TEST_CASE(luksan_vlcek1_serialization_test)
{
    problem p{luksan_vlcek1{3}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1., 1.});
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
