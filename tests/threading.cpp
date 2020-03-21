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

#define BOOST_TEST_MODULE threading_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>

#include <pagmo/threading.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(threading_test)
{
    // Check the ordering of the safety levels.
    BOOST_CHECK(thread_safety::none < thread_safety::basic);
    BOOST_CHECK(thread_safety::basic < thread_safety::constant);
    BOOST_CHECK(thread_safety::none <= thread_safety::basic);
    BOOST_CHECK(thread_safety::basic <= thread_safety::constant);
    BOOST_CHECK(thread_safety::none <= thread_safety::none);
    BOOST_CHECK(thread_safety::basic <= thread_safety::basic);
    BOOST_CHECK(thread_safety::constant <= thread_safety::constant);
    BOOST_CHECK(thread_safety::basic > thread_safety::none);
    BOOST_CHECK(thread_safety::constant > thread_safety::basic);
    BOOST_CHECK(thread_safety::basic >= thread_safety::none);
    BOOST_CHECK(thread_safety::constant >= thread_safety::basic);
    BOOST_CHECK(thread_safety::none >= thread_safety::none);
    BOOST_CHECK(thread_safety::basic >= thread_safety::basic);
    BOOST_CHECK(thread_safety::constant >= thread_safety::constant);

    // Test the streaming operator.
    std::ostringstream oss;
    oss << thread_safety::none;
    BOOST_CHECK_EQUAL("none", oss.str());
    oss.str("");
    oss << thread_safety::basic;
    BOOST_CHECK_EQUAL("basic", oss.str());
    oss.str("");
    oss << thread_safety::constant;
    BOOST_CHECK_EQUAL("constant", oss.str());
    oss.str("");
    oss << static_cast<thread_safety>(100);
    BOOST_CHECK_EQUAL("unknown value", oss.str());
}
