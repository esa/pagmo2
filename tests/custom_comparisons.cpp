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

#define BOOST_TEST_MODULE custom_comparisons_test
#include <boost/test/included/unit_test.hpp>

#include <limits>
#include <stdexcept>
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

    BOOST_CHECK((detail::less_than_f(a_big_double, a_small_double) == false));
    BOOST_CHECK((detail::less_than_f<double, true>(a_big_double, a_small_double) == false));
    BOOST_CHECK((detail::less_than_f<double, false>(a_big_double, a_small_double) == false));
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

    BOOST_CHECK((detail::greater_than_f(a_big_double, a_small_double) == true));
    BOOST_CHECK((detail::greater_than_f<double, true>(a_big_double, a_small_double) == true));
    BOOST_CHECK((detail::greater_than_f<double, false>(a_big_double, a_small_double) == true));
}

BOOST_AUTO_TEST_CASE(equal_to_f_test)
{
    auto a_nan = std::nan("");
    auto a_double = 123.456;

    // Test all branches on T=double
    BOOST_CHECK((detail::equal_to_f(a_nan, a_double) == false));
    BOOST_CHECK((detail::equal_to_f<double>(a_nan, a_double) == false));

    BOOST_CHECK((detail::equal_to_f(a_nan, a_nan) == true));
    BOOST_CHECK((detail::equal_to_f<double>(a_nan, a_nan) == true));

    BOOST_CHECK((detail::equal_to_f(a_double, a_double) == true));
    BOOST_CHECK((detail::equal_to_f<double>(a_double, a_double) == true));

    BOOST_CHECK((detail::equal_to_f(a_double, a_nan) == false));
    BOOST_CHECK((detail::equal_to_f<double>(a_double, a_nan) == false));
}

BOOST_AUTO_TEST_CASE(equal_to_vf_test)
{
    auto a_nan = std::nan("");
    vector_double v1 = {1., 2., 3., 4.};
    vector_double v2 = {1., 2., 3., 4., 5.};
    vector_double v3 = {1., 2., 3.1, 4.};
    vector_double v4 = {1., a_nan, 3.1, 4.};
    vector_double v5 = {1., a_nan, 3.1, 4.};
    vector_double v6 = {1., a_nan, a_nan, 4.};
    vector_double v7 = {1., 2.1, 3.1, 4.};

    // Test all branches on T=double
    BOOST_CHECK((detail::equal_to_vf<double>()(v1, v2) == false));
    BOOST_CHECK((detail::equal_to_vf<double>()(v1, v3) == false));
    BOOST_CHECK((detail::equal_to_vf<double>()(v3, v4) == false));
    BOOST_CHECK((detail::equal_to_vf<double>()(v4, v5) == true));
    BOOST_CHECK((detail::equal_to_vf<double>()(v4, v6) == false));
    BOOST_CHECK((detail::equal_to_vf<double>()(v3, v6) == false));
    BOOST_CHECK((detail::equal_to_vf<double>()(v3, v6) == false));
    BOOST_CHECK((detail::equal_to_vf<double>()(v3, v6) == false));
}

BOOST_AUTO_TEST_CASE(hash_vf_test)
{
    auto a_nan = std::nan("");
    vector_double v1 = {1., a_nan, 3., 4.};
    vector_double v2 = {1., 2., 3., 4.};
    vector_double v3 = {1., a_nan, 3., 4.};
    BOOST_CHECK((detail::hash_vf<double>()(v1) != detail::hash_vf<double>()(v2)));
    BOOST_CHECK((detail::hash_vf<double>()(v3) == detail::hash_vf<double>()(v1)));
}