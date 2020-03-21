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

#define BOOST_TEST_MODULE base_sr_policy
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/numeric/conversion/converter_policies.hpp>
#include <boost/variant/get.hpp>

#include <pagmo/detail/base_sr_policy.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

using bsrp = detail::base_sr_policy;

BOOST_AUTO_TEST_CASE(basic_test)
{
    bsrp b0(0);
    BOOST_CHECK(b0.get_migr_rate().which() == 0);
    BOOST_CHECK(boost::get<pop_size_t>(b0.get_migr_rate()) == 0);

    b0 = bsrp(4u);
    BOOST_CHECK(b0.get_migr_rate().which() == 0);
    BOOST_CHECK(boost::get<pop_size_t>(b0.get_migr_rate()) == 4);

    b0 = bsrp(.1);
    BOOST_CHECK(b0.get_migr_rate().which() == 1);
    BOOST_CHECK(boost::get<double>(b0.get_migr_rate()) == .1);

    b0 = bsrp(0.l);
    BOOST_CHECK(b0.get_migr_rate().which() == 1);
    BOOST_CHECK(boost::get<double>(b0.get_migr_rate()) == 0.);

    b0 = bsrp(1.f);
    BOOST_CHECK(b0.get_migr_rate().which() == 1);
    BOOST_CHECK(boost::get<double>(b0.get_migr_rate()) == 1.);

    BOOST_CHECK((!std::is_constructible<bsrp, const std::string &>::value));

    // Minimal serialization test.
    {
        std::stringstream ss;
        {
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << b0;
        }
        bsrp b1(0);
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> b1;
        }
        BOOST_CHECK(b1.get_migr_rate().which() == 1);
        BOOST_CHECK(boost::get<double>(b1.get_migr_rate()) == 1.);
    }

    // Error handling.
    BOOST_CHECK_EXCEPTION(b0 = bsrp(-1.), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "Invalid fractional migration rate specified in the constructor of a replacement/selection "
                       "policy: the rate must be in the [0., 1.] range, but it is ");
    });
    BOOST_CHECK_EXCEPTION(b0 = bsrp(2.), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "Invalid fractional migration rate specified in the constructor of a replacement/selection "
                       "policy: the rate must be in the [0., 1.] range, but it is ");
    });
    BOOST_CHECK_EXCEPTION(
        b0 = bsrp(std::numeric_limits<double>::infinity()), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "Invalid fractional migration rate specified in the constructor of a replacement/selection "
                           "policy: the rate must be in the [0., 1.] range, but it is ");
        });
    BOOST_CHECK_THROW(b0 = bsrp(-1), boost::numeric::negative_overflow);
}
