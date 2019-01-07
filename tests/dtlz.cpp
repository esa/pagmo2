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

#define BOOST_TEST_MODULE dtlz_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pagmo/problem.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(zdt_construction_test)
{
    dtlz dtlz_default{};
    dtlz dtlz5{5u, 7u, 3u, 100u};

    BOOST_CHECK_THROW((dtlz{0u, 7u, 3u, 100u}), std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{9u, 7u, 3u, 100u}), std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, 7u, 1u, 100u}), std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, 7u, std::numeric_limits<vector_double::size_type>::max() - 1u, 100u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, std::numeric_limits<vector_double::size_type>::max() - 1u, 3u, 100u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((dtlz{1u, 3u, 3u, 100u}), std::invalid_argument);

    BOOST_CHECK_NO_THROW(problem{dtlz_default});
    BOOST_CHECK_NO_THROW(problem{dtlz5});
    // We also test get_nobj() here as not to add one more small test
    BOOST_CHECK(dtlz_default.get_nobj() == 3u);
    // We also test get_name()
    BOOST_CHECK(dtlz5.get_name().find("DTLZ5") != std::string::npos);

    BOOST_CHECK(problem{dtlz5}.get_nx() == 7u);
    BOOST_CHECK(dtlz5.get_bounds().first.size() == 7u);
}

BOOST_AUTO_TEST_CASE(dtlz1_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{1u, 5u, 3u};
    f1 = {0.125, 0.125, 0.25};
    f2 = {0.059999999999999824, 0.2399999999999993, 2.699999999999992};
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz2_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{2u, 5u, 3u};
    f1 = {0.5000000000000001, 0.5, 0.7071067811865475};
    f2 = {0.9863148040113404, 0.3204731065093832, 0.16425618829224242};
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz3_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{3u, 5u, 3u};
    f1 = {0.5000000000000001, 0.5, 0.7071067811865475};
    f2 = {5.6360845943505, 1.8312748943393273, 0.9386067902413824};
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz4_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{4u, 5u, 3u};
    f1 = {1.0, 1.2391398122732624e-30, 1.2391398122732624e-30};
    f2 = {1.05, 2.090781951822753e-70, 1.6493361431346507e-100};
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz5_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{5u, 5u, 3u};
    f1 = {0.5000000000000001, 0.5, 0.7071067811865475};
    f2 = {0.7495908626265831, 0.7166822470763723, 0.16425618829224242};
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz6_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{6u, 5u, 3u};
    f1 = {1.8995494873052114, 1.8995494873052112, 2.6863686473458888};
    f2 = {3.3343308165801333, 1.5714799440921394, 0.5838204128120267};
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz7_fitness_test)
{
    vector_double dv1{0.5, 0.5, 0.5, 0.5, 0.5};
    vector_double dv2{0.1, 0.2, 0.3, 0.4, 0.5};
    vector_double f1, f2;
    // dtlz1
    dtlz udp{7u, 5u, 3u};
    f1 = {0.5, 0.5, 19.5};
    f2 = {0.1, 0.2, 16.228886997303473};
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv1)[i], f1[i], 1e-12);
    }
    for (unsigned int i = 0u; i < 3u; ++i) {
        BOOST_CHECK_CLOSE(udp.fitness(dv2)[i], f2[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(dtlz_get_bounds_test)
{
    std::pair<vector_double, vector_double> bounds({vector_double(4, 0.), vector_double(4, 1.)});
    for (unsigned int i = 1u; i <= 7u; ++i) {
        dtlz udp{i, 4u};
        BOOST_CHECK(udp.get_bounds() == bounds);
    }
}

BOOST_AUTO_TEST_CASE(dtlz_p_distance_test)
{
    vector_double x(4u, 0.231);
    vector_double x_wrong(3u, 0.231);
    // The following numbers were computed in PyGMO legacy
    vector_double res = {288.09711053693565,  0.14472200000000002, 288.09711053693565, 0.14472200000000002,
                         0.14472200000000002, 1.7273931523406256,  2.0790000000000002};
    for (unsigned int i = 1u; i <= 7u; ++i) {
        dtlz udp{i, 4u};
        BOOST_CHECK_CLOSE(udp.p_distance(x), res[i - 1u], 1e-12);
    }
    dtlz udp{3u, 4u};
    BOOST_CHECK_THROW(udp.p_distance(x_wrong), std::invalid_argument);
    BOOST_CHECK_NO_THROW(udp.p_distance(population{udp, 20u, 32u}));
}

BOOST_AUTO_TEST_CASE(dtlz_serialization_test)
{
    problem p{dtlz{4u, 4u}};
    // Call objfun to increase the internal counters.
    p.fitness({1., 1., 1., 1.});
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
