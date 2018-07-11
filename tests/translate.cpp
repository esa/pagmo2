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

#define BOOST_TEST_MODULE translate_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <stdexcept>
#include <string>

#include <pagmo/io.hpp>
#include <pagmo/problems/cec2006.hpp>
#include <pagmo/problems/cec2009.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(translate_construction_test)
{
    // First we check directly the two constructors
    problem p0{translate{}};
    problem p1{translate{null_problem{}, {0.}}};

    auto p0_string = boost::lexical_cast<std::string>(p0);
    auto p1_string = boost::lexical_cast<std::string>(p1);

    // We check that the default constructor constructs a problem
    // which has an identical representation to the problem
    // built by the explicit constructor.
    BOOST_CHECK(p0_string == p1_string);

    // We check that wrong size for translation results in an invalid_argument
    // exception
    BOOST_CHECK_THROW((translate{null_problem{}, {1, 2}}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(translate_functional_test)
{
    // Then we check that the hock_schittkowsky_71 problem is actually translated
    {
        hock_schittkowsky_71 hs;
        problem p0{hs};
        translate t1{hs, {0.1, -0.2, 0.3, 0.4}};
        problem p1{t1};
        problem p2{translate{t1, {-0.1, 0.2, -0.3, -0.4}}};
        vector_double x{3., 3., 3., 3.};
        // Fitness gradients and hessians are the same if the translation  is zero
        BOOST_CHECK(p0.fitness(x) == p2.fitness(x));
        BOOST_CHECK(p0.gradient(x) == p2.gradient(x));
        BOOST_CHECK(p0.hessians(x) == p2.hessians(x));
        // Bounds are unchanged if the translation is zero
        BOOST_CHECK(p0.get_bounds().first != p1.get_bounds().first);
        BOOST_CHECK(p0.get_bounds().first != p1.get_bounds().second);
        auto bounds0 = p0.get_bounds();
        auto bounds2 = p2.get_bounds();
        for (auto i = 0u; i < 4u; ++i) {
            BOOST_CHECK_CLOSE(bounds0.first[i], bounds2.first[i], 1e-13);
            BOOST_CHECK_CLOSE(bounds0.second[i], bounds2.second[i], 1e-13);
        }
        // We check that the problem's name has [translated] appended
        BOOST_CHECK(p1.get_name().find("[translated]") != std::string::npos);
        // We check that extra info has "Translation Vector:" somewhere"
        BOOST_CHECK(p1.get_extra_info().find("Translation Vector:") != std::string::npos);
        // We check we recover the translation vector
        auto translationvector = p1.extract<translate>()->get_translation();
        BOOST_CHECK((translationvector == vector_double{0.1, -0.2, 0.3, 0.4}));
    }
}

BOOST_AUTO_TEST_CASE(translate_serialization_test)
{
    // Do the checking with the full problem.
    hock_schittkowsky_71 p0{};
    problem p{translate{p0, {0.1, -0.2, 0.3, 0.4}}};
    // Call objfun, grad and hess to increase
    // the internal counters.
    p.fitness({1., 1., 1., 1.});
    p.gradient({1., 1., 1., 1.});
    p.hessians({1., 1., 1., 1.});
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

BOOST_AUTO_TEST_CASE(translate_stochastic_test)
{
    hock_schittkowsky_71 p0{};
    problem p{translate{p0, {0.1, -0.2, 0.3, 0.4}}};
    BOOST_CHECK(!p.is_stochastic());
}

struct ts2 {
    vector_double fitness(const vector_double &) const
    {
        return {2};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0}, {1}};
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

BOOST_AUTO_TEST_CASE(translate_thread_safety_test)
{
    hock_schittkowsky_71 p0{};
    translate t{p0, {0.1, -0.2, 0.3, 0.4}};
    BOOST_CHECK(t.get_thread_safety() == thread_safety::basic);
    BOOST_CHECK((translate{ts2{}, {1}}.get_thread_safety() == thread_safety::none));
}

template <typename T>
void check_inheritance(T udp, const vector_double &t)
{
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).get_nobj(), problem(udp).get_nobj());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).get_nec(), problem(udp).get_nec());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).get_nic(), problem(udp).get_nic());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).get_nix(), problem(udp).get_nix());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).has_gradient(), problem(udp).has_gradient());
    BOOST_CHECK(translate(udp, t).gradient_sparsity() == problem(udp).gradient_sparsity());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).has_gradient_sparsity(), problem(udp).has_gradient_sparsity());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).has_hessians(), problem(udp).has_hessians());
    BOOST_CHECK(problem(translate(udp, t)).hessians_sparsity() == problem(udp).hessians_sparsity());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).has_hessians_sparsity(), problem(udp).has_hessians_sparsity());
    BOOST_CHECK_EQUAL(problem(translate(udp, t)).has_set_seed(), problem(udp).has_set_seed());
}

BOOST_AUTO_TEST_CASE(translate_inheritance_test)
{
    check_inheritance(hock_schittkowsky_71{}, vector_double(4, 0.5));
    check_inheritance(cec2006{1}, vector_double(13, 0.5));
    check_inheritance(cec2009{1}, vector_double(30, 0.5));
    // We check the forwarding of the integer dimension. The translation needs to be integer too as to
    // not create a non integer bound.
    check_inheritance(null_problem{2, 2, 3, 1}, vector_double(1, 1));
    check_inheritance(null_problem{2, 2, 3, 0}, vector_double(1, 1));

    // We check if set_seed is working
    problem p{translate{inventory{10u, 10u, 1234567u}, vector_double(10, 1.)}};
    std::ostringstream ss1, ss2;
    ss1 << p;
    BOOST_CHECK(ss1.str().find(std::to_string(1234567u)) != std::string::npos);
    p.set_seed(5672543u);
    ss2 << p;
    BOOST_CHECK(ss2.str().find(std::to_string(5672543u)) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(translate_inner_algo_get_test)
{
    // We check that the correct overload is called according to (*this) being const or not
    {
        const translate udp(hock_schittkowsky_71{}, vector_double(4, 0.5));
        BOOST_CHECK(std::is_const<decltype(udp)>::value);
        BOOST_CHECK(std::is_const<std::remove_reference<decltype(udp.get_inner_problem())>::type>::value);
    }
    {
        translate udp(hock_schittkowsky_71{}, vector_double(4, 0.5));
        BOOST_CHECK(!std::is_const<decltype(udp)>::value);
        BOOST_CHECK(!std::is_const<std::remove_reference<decltype(udp.get_inner_problem())>::type>::value);
    }
}