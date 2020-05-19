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

#define BOOST_TEST_MODULE generic_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/algorithm/string/predicate.hpp>
#include <limits>

#include <pagmo/utils/genetic_operators.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(sbx_crossover_test)
{
    detail::random_engine_type random_engine(32u);
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto inf = std::numeric_limits<double>::infinity();
    BOOST_CHECK_NO_THROW(
        sbx_crossover({0.1, 0.2, 3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine));
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(),
                                   "The length of the chromosomes of the parents should be equal: parent1 length is");
        });
    BOOST_CHECK_EXCEPTION(sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{}, {}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(), "The bounds dimension cannot be zero");
                          });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "The length of the lower bounds vector is");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2}, {3, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "The length of the chromosomes of the parents should be the same as that "
                                              "of the bounds: parent1 length is");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{nan, -2, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "A NaN value was encountered in the problem bounds, index");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, inf, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Infinite value detected in the bounds at position");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, nan, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "A NaN value was encountered in the problem bounds, index");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -inf}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Infinite value detected in the bounds at position");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, -5, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument,
        [](const std::invalid_argument &ia) { return boost::contains(ia.what(), "The lower bound at position"); });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, 8, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument,
        [](const std::invalid_argument &ia) { return boost::contains(ia.what(), "The lower bound at position"); });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, 3, 3}}, 32u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "The integer part cannot be larger than the bounds size");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2.6}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "A lower bound of the integer part of the decision vector is");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, 3, 3.43}}, 1u, 0.9, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "An upper bound of the integer part of the decision vector is");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, 3, 3}}, 1u, nan, 10, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Crossover probability is not finite, value is");
        });
    BOOST_CHECK_EXCEPTION(
        sbx_crossover({0.1, 0.2, 0.3}, {0.2, 2.2, -1}, {{-2, -2, -2}, {3, 3, 3}}, 1u, 0.4, nan, random_engine),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Crossover distribution index is not finite, value is");
        });
}

BOOST_AUTO_TEST_CASE(polynomial_mutation_test)
{
    detail::random_engine_type random_engine(32u);
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto inf = std::numeric_limits<double>::infinity();
    vector_double dv = {-0.3, 2.4, 5};
    BOOST_CHECK_NO_THROW(polynomial_mutation(dv, {{-2, -2, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine));
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{}, {}}, 1u, 0.9, 10, random_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(), "The bounds dimension cannot be zero");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(), "The length of the lower bounds vector is");
                          });
    BOOST_CHECK_EXCEPTION(
        polynomial_mutation(dv, {{-2, -2}, {3, 3}}, 1u, 0.9, 10, random_engine), std::invalid_argument,
        [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "The length of the chromosome should be the same as that of the bounds: detected length is");
        });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{nan, -2, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "A NaN value was encountered in the problem bounds, index");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -2}, {3, inf, 3}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(), "Infinite value detected in the bounds at position");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -2}, {3, nan, 3}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "A NaN value was encountered in the problem bounds, index");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -inf}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(), "Infinite value detected in the bounds at position");
                          });
    BOOST_CHECK_EXCEPTION(
        polynomial_mutation(dv, {{-2, -2, -2}, {3, -5, 3}}, 1u, 0.9, 10, random_engine), std::invalid_argument,
        [](const std::invalid_argument &ia) { return boost::contains(ia.what(), "The lower bound at position"); });
    BOOST_CHECK_EXCEPTION(
        polynomial_mutation(dv, {{-2, 8, -2}, {3, 3, 3}}, 1u, 0.9, 10, random_engine), std::invalid_argument,
        [](const std::invalid_argument &ia) { return boost::contains(ia.what(), "The lower bound at position"); });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -2}, {3, 3, 3}}, 32u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "The integer part cannot be larger than the bounds size");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -2.6}, {3, 3, 3}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "A lower bound of the integer part of the decision vector is");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -2}, {3, 3, 3.43}}, 1u, 0.9, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "An upper bound of the integer part of the decision vector is");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -2}, {3, 3, 3}}, 1u, nan, 10, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(), "Mutation probability is not finite, value is");
                          });
    BOOST_CHECK_EXCEPTION(polynomial_mutation(dv, {{-2, -2, -2}, {3, 3, 3}}, 1u, 0.4, nan, random_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(), "Mutation distribution index is not finite, value is");
                          });
}