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

#include <pagmo/io.hpp>

#define BOOST_TEST_MODULE io_test
#include <boost/test/included/unit_test.hpp>

#include <initializer_list>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/threading.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(stream_print_test_00)
{
    // A few simple tests.
    std::ostringstream ss1, ss2;
    stream(ss1, 1, 2, 3);
    ss2 << 1 << 2 << 3;
    BOOST_CHECK_EQUAL(ss1.str(), ss2.str());
    ss1.str("");
    ss2.str("");
    stream(ss1, "Hello ", std::string(" world"));
    ss2 << "Hello " << std::string(" world");
    BOOST_CHECK_EQUAL(ss1.str(), ss2.str());
    ss1.str("");
    ss2.str("");
    // Try with floating-point too.
    stream(ss1, 1.234);
    ss2 << 1.234;
    BOOST_CHECK_EQUAL(ss1.str(), ss2.str());
    ss1.str("");
    ss2.str("");
    // Custom precision.
    ss1 << std::setprecision(10);
    ss2 << std::setprecision(10);
    stream(ss1, 1.234);
    ss2 << 1.234;
    BOOST_CHECK_EQUAL(ss1.str(), ss2.str());
    ss1.str("");
    ss2.str("");
    // Special handling of bool.
    stream(ss1, true, ' ', false);
    BOOST_CHECK_EQUAL(ss1.str(), "true false");
    ss1.str("");
    // Vectors.
    stream(ss1, std::vector<int>{});
    BOOST_CHECK_EQUAL(ss1.str(), "[]");
    ss1.str("");
    stream(ss1, std::vector<int>{1, 2, 3});
    ss2 << "[" << 1 << ", " << 2 << ", " << 3 << "]";
    BOOST_CHECK_EQUAL(ss1.str(), ss2.str());
    ss1.str("");
    ss2.str("");
    // Vector larger than the print limit.
    stream(ss1, std::vector<int>{1, 2, 3, 4, 5, 6});
    ss2 << "[" << 1 << ", " << 2 << ", " << 3 << ", " << 4 << ", " << 5 << ", ... ]";
    BOOST_CHECK_EQUAL(ss1.str(), ss2.str());
    // Go for the print as well, yay.
    print(std::vector<int>{1, 2, 3, 4, 5, 6});
    // Thread safety levels.
    ss1.str("");
    stream(ss1, thread_safety::none);
    BOOST_CHECK_EQUAL(ss1.str(), "none");
    ss1.str("");
    stream(ss1, thread_safety::basic);
    BOOST_CHECK_EQUAL(ss1.str(), "basic");
}

BOOST_AUTO_TEST_CASE(stream_print_test_01)
{
    // Map.
    using map_t = std::map<int, int>;
    std::stringstream ss;
    stream(ss, map_t{{0, 0}, {1, 1}, {2, 2}});
    print(map_t{{0, 0}, {1, 1}, {2, 2}});
    ss.str("");
    stream(ss, map_t{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});
    print(map_t{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});
    // Pair.
    using pair_t = std::pair<int, int>;
    ss.str("");
    stream(ss, pair_t{1, 2});
    print(pair_t{1, 2});
}

BOOST_AUTO_TEST_CASE(stream_table_test)
{
    detail::table t({"a", "b", "c"});
    BOOST_CHECK_THROW(t.add_row(), std::invalid_argument);
}