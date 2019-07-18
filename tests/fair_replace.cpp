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

#define BOOST_TEST_MODULE fair_replace_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/numeric/conversion/converter_policies.hpp>
#include <boost/variant/get.hpp>

#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(fair_replace_basic)
{
    fair_replace f00;
    BOOST_REQUIRE(f00.get_migr_rate().which() == 1);
    BOOST_REQUIRE(boost::get<double>(f00.get_migr_rate()) == .1);

    fair_replace f01(.2);
    BOOST_REQUIRE(f01.get_migr_rate().which() == 1);
    BOOST_REQUIRE(boost::get<double>(f01.get_migr_rate()) == .2);

    fair_replace f02(2);
    BOOST_REQUIRE(f02.get_migr_rate().which() == 0);
    BOOST_REQUIRE(boost::get<pop_size_t>(f02.get_migr_rate()) == 2u);

    BOOST_CHECK_EXCEPTION(f02 = fair_replace(-1.), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "Invalid fractional migration rate specified in the constructor of a replacement/selection "
                       "policy: the rate must be in the [0., 1.] range, but it is ");
    });
    BOOST_CHECK_EXCEPTION(f02 = fair_replace(2.), std::invalid_argument, [](const std::invalid_argument &ia) {
        return boost::contains(
            ia.what(), "Invalid fractional migration rate specified in the constructor of a replacement/selection "
                       "policy: the rate must be in the [0., 1.] range, but it is ");
    });
    BOOST_CHECK_EXCEPTION(
        f02 = fair_replace(std::numeric_limits<double>::infinity()), std::invalid_argument,
        [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "Invalid fractional migration rate specified in the constructor of a replacement/selection "
                           "policy: the rate must be in the [0., 1.] range, but it is ");
        });
    BOOST_CHECK_THROW(f02 = fair_replace(-1), boost::numeric::negative_overflow);

    auto f03(f02);
    BOOST_REQUIRE(f03.get_migr_rate().which() == 0);
    BOOST_REQUIRE(boost::get<pop_size_t>(f03.get_migr_rate()) == 2u);

    auto f04(std::move(f01));
    BOOST_REQUIRE(f04.get_migr_rate().which() == 1);
    BOOST_REQUIRE(boost::get<double>(f04.get_migr_rate()) == .2);

    f03 = f04;
    BOOST_REQUIRE(f03.get_migr_rate().which() == 1);
    BOOST_REQUIRE(boost::get<double>(f03.get_migr_rate()) == .2);

    f04 = std::move(f02);
    BOOST_REQUIRE(f04.get_migr_rate().which() == 0);
    BOOST_REQUIRE(boost::get<pop_size_t>(f04.get_migr_rate()) == 2u);

    BOOST_REQUIRE(f04.get_name() == "Fair replace");
    BOOST_REQUIRE(boost::contains(f04.get_extra_info(), "Absolute migration rate:"));
    BOOST_REQUIRE(boost::contains(f03.get_extra_info(), "Fractional migration rate:"));

    // Minimal serialization test.
    {
        r_policy r0(f04);

        std::stringstream ss;
        {
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << r0;
        }
        r_policy r1;
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> r1;
        }
        BOOST_REQUIRE(r1.is<fair_replace>());
        BOOST_REQUIRE(r1.extract<fair_replace>()->get_migr_rate().which() == 0);
        BOOST_REQUIRE(boost::get<pop_size_t>(r1.extract<fair_replace>()->get_migr_rate()) == 2u);
    }
}

BOOST_AUTO_TEST_CASE(fair_replace_replace)
{
    fair_replace f00;

    BOOST_CHECK_EXCEPTION(f00.replace(individuals_group_t{}, 0, 0, 2, 1, 0, vector_double{}, individuals_group_t{}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "The 'fair_replace' replacement policy is unable to deal with "
                                                     "multiobjective constrained optimisation problems");
                          });

    f00 = fair_replace(100);

    BOOST_CHECK_EXCEPTION(f00.replace(individuals_group_t{}, 0, 0, 1, 0, 0, vector_double{}, individuals_group_t{}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(), "The absolute migration rate (100) in a 'fair_replace' replacement policy "
                                             "is larger than the number of input individuals (0)");
                          });
}
