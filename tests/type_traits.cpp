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

#define BOOST_TEST_MODULE type_traits_test
#include <boost/test/included/unit_test.hpp>

#include <type_traits>
#include <utility>

#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>

using namespace pagmo;

struct s1 {
    void foo();
};

struct s2 {
    void foo() const;
};

struct s3 {
};

struct s4 {
    void foo(int) const;
};

template <typename T>
using foo_t = decltype(std::declval<const T &>().foo());

BOOST_AUTO_TEST_CASE(type_traits_test_00)
{
    BOOST_CHECK((!is_detected<foo_t, s1>::value));
    BOOST_CHECK((is_detected<foo_t, s2>::value));
    BOOST_CHECK((!is_detected<foo_t, s3>::value));
    BOOST_CHECK((!is_detected<foo_t, s4>::value));
    BOOST_CHECK((std::is_same<detected_t<foo_t, s4>, detail::nonesuch>::value));
}

struct ngts1 {
    int get_thread_safety() const;
};

struct ngts2 {
    thread_safety get_thread_safety();
};

struct ygts1 {
    thread_safety get_thread_safety() const;
};

BOOST_AUTO_TEST_CASE(type_traits_has_get_thread_safety_test)
{
    BOOST_CHECK((!has_get_thread_safety<s1>::value));
    BOOST_CHECK((!has_get_thread_safety<s2>::value));
    BOOST_CHECK((!has_get_thread_safety<s3>::value));
    BOOST_CHECK((!has_get_thread_safety<ngts1>::value));
    BOOST_CHECK((!has_get_thread_safety<ngts2>::value));
    BOOST_CHECK((has_get_thread_safety<ygts1>::value));
}
