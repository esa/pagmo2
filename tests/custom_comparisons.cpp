/* Copyright 2017 PaGMO development team

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

#define BOOST_TEST_MODULE custom_comparisons_test
#include <boost/test/included/unit_test.hpp>

#include <exception>
#include <limits>
#include <tuple>

#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/io.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(less_than_f_test)
{
    auto a_nan = std::nan("");
    auto a_big_double = 1e4;
    auto a_small_double = -1e4;

    // Test all branches on T=double
    BOOST_CHECK((detail::less_than_f(a_nan, a_big_double) == false));
    BOOST_CHECK((detail::less_than_f<double, true>(a_nan, a_big_double) == false));
    BOOST_CHECK((detail::less_than_f<double, false>(a_nan, a_big_double) == true));

    BOOST_CHECK((detail::less_than_f(a_nan, a_nan) == false));
    BOOST_CHECK((detail::less_than_f<double, true>(a_nan, a_nan) == false));
    BOOST_CHECK((detail::less_than_f<double, false>(a_nan, a_nan) == false));

    BOOST_CHECK((detail::less_than_f(a_big_double, a_nan) == true));
    BOOST_CHECK((detail::less_than_f<double, true>(a_big_double, a_nan) == true));
    BOOST_CHECK((detail::less_than_f<double, false>(a_big_double, a_nan) == false));

    BOOST_CHECK((detail::less_than_f(a_small_double, a_big_double) == true));
    BOOST_CHECK((detail::less_than_f<double, true>(a_small_double, a_big_double) == true));
    BOOST_CHECK((detail::less_than_f<double, false>(a_small_double, a_big_double) == true));
}

BOOST_AUTO_TEST_CASE(greater_than_f_test)
{
    auto a_nan = std::nan("");
    auto a_big_double = 1e4;
    auto a_small_double = -1e4;

    // Test all branches on T=double
    BOOST_CHECK((detail::greater_than_f(a_nan, a_big_double) == true));
    BOOST_CHECK((detail::greater_than_f<double, true>(a_nan, a_big_double) == true));
    BOOST_CHECK((detail::greater_than_f<double, false>(a_nan, a_big_double) == false));

    BOOST_CHECK((detail::greater_than_f(a_nan, a_nan) == false));
    BOOST_CHECK((detail::greater_than_f<double, true>(a_nan, a_nan) == false));
    BOOST_CHECK((detail::greater_than_f<double, false>(a_nan, a_nan) == false));

    BOOST_CHECK((detail::greater_than_f(a_big_double, a_nan) == false));
    BOOST_CHECK((detail::greater_than_f<double, true>(a_big_double, a_nan) == false));
    BOOST_CHECK((detail::greater_than_f<double, false>(a_big_double, a_nan) == true));

    BOOST_CHECK((detail::greater_than_f(a_small_double, a_big_double) == false));
    BOOST_CHECK((detail::greater_than_f<double, true>(a_small_double, a_big_double) == false));
    BOOST_CHECK((detail::greater_than_f<double, false>(a_small_double, a_big_double) == false));
}
