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

#define BOOST_TEST_MODULE algorithm_type_traits_test
#include <boost/test/included/unit_test.hpp>

#include <utility>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>

using namespace pagmo;

struct hsv_00 {
};

// The good one
struct hsv_01 {
    void set_verbosity(unsigned int);
};

// also good
struct hsv_02 {
    void set_verbosity(unsigned int) const;
};

// also good
struct hsv_03 {
    void set_verbosity(int);
};

struct hsv_04 {
    double set_verbosity(unsigned int);
};

BOOST_AUTO_TEST_CASE(has_set_verbose_test)
{
    BOOST_CHECK((!has_set_verbosity<hsv_00>::value));
    BOOST_CHECK((has_set_verbosity<hsv_01>::value));
    BOOST_CHECK((has_set_verbosity<hsv_02>::value));
    BOOST_CHECK((has_set_verbosity<hsv_03>::value));
    BOOST_CHECK((!has_set_verbosity<hsv_04>::value));
}

struct hev_00 {
};

// The good one
struct hev_01 {
    population evolve(population) const;
};

struct hev_02 {
    population evolve(const population &);
};

struct hev_03 {
    population evolve(population &) const;
};

struct hev_04 {
    double evolve(const population &) const;
};

struct hev_05 {
    population evolve(const double &) const;
};

BOOST_AUTO_TEST_CASE(has_evolve_test)
{
    BOOST_CHECK((!has_evolve<hev_00>::value));
    BOOST_CHECK((has_evolve<hev_01>::value));
    BOOST_CHECK((!has_evolve<hev_02>::value));
    BOOST_CHECK((!has_evolve<hev_03>::value));
    BOOST_CHECK((!has_evolve<hev_04>::value));
    BOOST_CHECK((!has_evolve<hev_05>::value));
}
