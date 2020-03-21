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

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/generic.hpp>

using namespace pagmo;

// A UDP with user-defined bounds.
struct udp00 {
    udp00() = default;
    explicit udp00(vector_double lb, vector_double ub, vector_double::size_type nix = 0)
        : m_lb(lb), m_ub(ub), m_nix(nix)
    {
    }
    vector_double fitness(const vector_double &) const
    {
        return {0};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {m_lb, m_ub};
    }
    vector_double::size_type get_nix() const
    {
        return m_nix;
    }
    vector_double m_lb, m_ub;
    vector_double::size_type m_nix;
};

BOOST_AUTO_TEST_CASE(uniform_real_from_range_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    auto nan = std::numeric_limits<double>::quiet_NaN();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_THROW(uniform_real_from_range(1, 0, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-big, big, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-3, inf, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-nan, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(nan, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-nan, 3, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(-3, nan, r_engine), std::invalid_argument);
    BOOST_CHECK_THROW(uniform_real_from_range(inf, inf, r_engine), std::invalid_argument);

    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(1, 0, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "Cannot generate a random integer if the lower bound is larger than the upper bound");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(0, inf, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot generate a random integer if the bounds are not finite");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(-inf, 0, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot generate a random integer if the bounds are not finite");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(-inf, inf, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot generate a random integer if the bounds are not finite");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(0, nan, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot generate a random integer if the bounds are not finite");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(-nan, 0, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot generate a random integer if the bounds are not finite");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(-nan, nan, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(), "Cannot generate a random integer if the bounds are not finite");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(0, .1, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "Cannot generate a random integer if the lower/upper bounds are not integral values");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(0.1, 2, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "Cannot generate a random integer if the lower/upper bounds are not integral values");
        });
    BOOST_CHECK_EXCEPTION(
        uniform_integral_from_range(0.1, 0.2, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "Cannot generate a random integer if the lower/upper bounds are not integral values");
        });
    if (big > static_cast<double>(std::numeric_limits<long long>::max())
        && -big < static_cast<double>(std::numeric_limits<long long>::min())) {
        BOOST_CHECK_EXCEPTION(
            uniform_integral_from_range(0, big, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
                return boost::contains(ia.what(),
                                       "Cannot generate a random integer if the lower/upper bounds are not within "
                                       "the bounds of the long long type");
            });
        BOOST_CHECK_EXCEPTION(
            uniform_integral_from_range(-big, 0, r_engine), std::invalid_argument, [](const std::invalid_argument &ia) {
                return boost::contains(ia.what(),
                                       "Cannot generate a random integer if the lower/upper bounds are not within "
                                       "the bounds of the long long type");
            });
    }
}

BOOST_AUTO_TEST_CASE(random_decision_vector_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{0}, {inf}}}, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{-inf}, {0}}}, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{-inf}, {inf}}}, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{-big}, {big}}}, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real within bounds that are too large");
                          });
    BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{0, 0}, {1, inf}, 1}}, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random integer if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{0, -inf}, {1, 0}, 1}}, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random integer if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{0, -inf}, {1, inf}, 1}}, r_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random integer if the bounds are not finite");
                          });
    if (big > static_cast<double>(std::numeric_limits<long long>::max())
        && -big < static_cast<double>(std::numeric_limits<long long>::min())) {
        BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{0, 0}, {1, big}, 1}}, r_engine),
                              std::invalid_argument, [](const std::invalid_argument &ia) {
                                  return boost::contains(
                                      ia.what(),
                                      "Cannot generate a random integer if the lower/upper bounds are not within "
                                      "the bounds of the long long type");
                              });
        BOOST_CHECK_EXCEPTION(random_decision_vector(problem{udp00{{0, -big}, {1, 0}, 1}}, r_engine),
                              std::invalid_argument, [](const std::invalid_argument &ia) {
                                  return boost::contains(
                                      ia.what(),
                                      "Cannot generate a random integer if the lower/upper bounds are not within "
                                      "the bounds of the long long type");
                              });
    }

    // Test the results
    BOOST_CHECK((random_decision_vector(problem{udp00{{3, 4}, {3, 4}}}, r_engine) == vector_double{3, 4}));
    BOOST_CHECK((random_decision_vector(problem{udp00{{3, 4}, {3, 4}, 1}}, r_engine) == vector_double{3, 4}));
    BOOST_CHECK((random_decision_vector(problem{udp00{{0, 0}, {1, 1}}}, r_engine)[0] >= 0.));
    BOOST_CHECK((random_decision_vector(problem{udp00{{0, 0}, {1, 1}}}, r_engine)[1] < 1.));
    BOOST_CHECK((random_decision_vector(problem{udp00{{0, 0}, {1, 0}}}, r_engine)[1] == 0.));
    for (auto i = 0; i < 100; ++i) {
        const auto tmp = random_decision_vector(problem{udp00{{0}, {2}, 1}}, r_engine)[0];
        BOOST_CHECK(tmp == 0. || tmp == 1. || tmp == 2.);
    }
    for (auto i = 0; i < 100; ++i) {
        const auto res = random_decision_vector(problem{udp00{{0, -20}, {1, 20}, 1}}, r_engine);
        BOOST_CHECK(std::trunc(res[1]) == res[1]);
    }
}

BOOST_AUTO_TEST_CASE(batch_random_decision_vector_test)
{
    auto inf = std::numeric_limits<double>::infinity();
    auto big = std::numeric_limits<double>::max();
    detail::random_engine_type r_engine(pagmo::random_device::next());

    // Test the throws
    BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{0}, {inf}}}, 0, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{-inf}, {0}}}, 0, r_engine), std::invalid_argument,
                          [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{-inf}, {inf}}}, 0, r_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{-big}, {big}}}, 0, r_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random real within bounds that are too large");
                          });
    BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{0, 0}, {1, inf}, 1}}, 0, r_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random integer if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{0, -inf}, {1, 0}, 1}}, 0, r_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random integer if the bounds are not finite");
                          });
    BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{0, -inf}, {1, inf}, 1}}, 0, r_engine),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "Cannot generate a random integer if the bounds are not finite");
                          });
    if (big > static_cast<double>(std::numeric_limits<long long>::max())
        && -big < static_cast<double>(std::numeric_limits<long long>::min())) {
        BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{0, 0}, {1, big}, 1}}, 10, r_engine),
                              std::invalid_argument, [](const std::invalid_argument &ia) {
                                  return boost::contains(
                                      ia.what(),
                                      "Cannot generate a random integer if the lower/upper bounds are not within "
                                      "the bounds of the long long type");
                              });
        BOOST_CHECK_EXCEPTION(batch_random_decision_vector(problem{udp00{{0, -big}, {1, 0}, 1}}, 10, r_engine),
                              std::invalid_argument, [](const std::invalid_argument &ia) {
                                  return boost::contains(
                                      ia.what(),
                                      "Cannot generate a random integer if the lower/upper bounds are not within "
                                      "the bounds of the long long type");
                              });
    }

    // Test the results
    BOOST_CHECK(batch_random_decision_vector(problem{udp00{{3, 4}, {3, 4}}}, 0, r_engine).empty());
    BOOST_CHECK(
        (batch_random_decision_vector(problem{udp00{{3, 4}, {3, 4}}}, 3, r_engine) == vector_double{3, 4, 3, 4, 3, 4}));
    BOOST_CHECK((batch_random_decision_vector(problem{udp00{{3, 4}, {3, 4}, 1}}, 3, r_engine)
                 == vector_double{3, 4, 3, 4, 3, 4}));
    auto tmp = batch_random_decision_vector(problem{udp00{{0, 0}, {1, 1}}}, 3, r_engine);
    BOOST_CHECK(tmp.size() == 6u);
    BOOST_CHECK(tmp[0] >= 0.);
    BOOST_CHECK(tmp[2] >= 0.);
    BOOST_CHECK(tmp[4] >= 0.);
    BOOST_CHECK(tmp[1] < 1.);
    BOOST_CHECK(tmp[3] < 1.);
    BOOST_CHECK(tmp[5] < 1.);
    tmp = batch_random_decision_vector(problem{udp00{{0, 0}, {1, 0}}}, 3, r_engine);
    BOOST_CHECK(tmp.size() == 6u);
    BOOST_CHECK(tmp[1] == 0.);
    BOOST_CHECK(tmp[3] == 0.);
    BOOST_CHECK(tmp[5] == 0.);
    for (auto i = 0; i < 100; ++i) {
        tmp = batch_random_decision_vector(problem{udp00{{0}, {2}, 1}}, 3, r_engine);
        BOOST_CHECK(tmp.size() == 3u);
        BOOST_CHECK(tmp[0] == 0. || tmp[0] == 1. || tmp[0] == 2.);
        BOOST_CHECK(tmp[1] == 0. || tmp[1] == 1. || tmp[1] == 2.);
        BOOST_CHECK(tmp[2] == 0. || tmp[2] == 1. || tmp[2] == 2.);
    }
    for (auto i = 0; i < 100; ++i) {
        tmp = batch_random_decision_vector(problem{udp00{{0, -20}, {1, 20}, 1}}, 3, r_engine);
        BOOST_CHECK(tmp.size() == 6u);
        BOOST_CHECK(std::trunc(tmp[1]) == tmp[1]);
        BOOST_CHECK(std::trunc(tmp[3]) == tmp[3]);
        BOOST_CHECK(std::trunc(tmp[5]) == tmp[5]);
    }
}

BOOST_AUTO_TEST_CASE(force_bounds_test)
{
    detail::random_engine_type r_engine(32u);
    // force_bounds_random
    {
        vector_double x{1., 2., 3.};
        vector_double x_fix = x;
        detail::force_bounds_random(x_fix, {0., 0., 0.}, {3., 3., 3.}, r_engine);
        BOOST_CHECK(x == x_fix);
        detail::force_bounds_random(x_fix, {0., 0., 0.}, {1., 1., 1.}, r_engine);
        BOOST_CHECK(x != x_fix);
        BOOST_CHECK_EQUAL(x_fix[0], 1.);
        BOOST_CHECK(x_fix[1] <= 1. && x_fix[1] >= 0.);
        BOOST_CHECK(x_fix[2] <= 1. && x_fix[2] >= 0.);
    }
    // force_bounds_reflection
    {
        vector_double x{1., 2., 5.};
        vector_double x_fix = x;
        detail::force_bounds_reflection(x_fix, {0., 0., 0.}, {3., 3., 5.});
        BOOST_CHECK(x == x_fix);
        detail::force_bounds_reflection(x_fix, {0., 0., 0.}, {1., 1.9, 2.1});
        BOOST_CHECK(x != x_fix);
        BOOST_CHECK_EQUAL(x_fix[0], 1.);
        BOOST_CHECK_CLOSE(x_fix[1], 1.8, 1e-8);
        BOOST_CHECK_CLOSE(x_fix[2], 0.8, 1e-8);
    }
    // force_bounds_stick
    {
        vector_double x{1., 2., 5.};
        vector_double x_fix = x;
        detail::force_bounds_stick(x_fix, {0., 0., 0.}, {3., 3., 5.});
        BOOST_CHECK(x == x_fix);
        // ub
        detail::force_bounds_stick(x_fix, {0., 0., 0.}, {1., 1.9, 2.1});
        BOOST_CHECK(x != x_fix);
        BOOST_CHECK_EQUAL(x_fix[0], 1.);
        BOOST_CHECK_EQUAL(x_fix[1], 1.9);
        BOOST_CHECK_EQUAL(x_fix[2], 2.1);
        // lb
        detail::force_bounds_stick(x_fix, {2., 2., 2.}, {3., 3., 3.});
        BOOST_CHECK_EQUAL(x_fix[0], 2.);
        BOOST_CHECK_EQUAL(x_fix[1], 2.);
        BOOST_CHECK_EQUAL(x_fix[2], 2.1);
    }
}

BOOST_AUTO_TEST_CASE(binomial_coefficient_test)
{
    BOOST_CHECK_EQUAL(binomial_coefficient(0u, 0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(1u, 0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(1u, 1u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u, 0u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u, 1u), 2u);
    BOOST_CHECK_EQUAL(binomial_coefficient(2u, 2u), 1u);
    BOOST_CHECK_EQUAL(binomial_coefficient(13u, 5u), 1287u);
    BOOST_CHECK_EQUAL(binomial_coefficient(21u, 10u), 352716u);
    BOOST_CHECK_THROW(binomial_coefficient(10u, 21u), std::invalid_argument);
    BOOST_CHECK_THROW(binomial_coefficient(0u, 1u), std::invalid_argument);
    BOOST_CHECK_THROW(binomial_coefficient(4u, 7u), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(kNN_test)
{
    // Corner cases
    {
        std::vector<vector_double> points = {};
        std::vector<std::vector<vector_double::size_type>> res = {};
        BOOST_CHECK(kNN(points, 2u) == res);
    }
    {
        std::vector<vector_double> points = {{1.2, 1.2}};
        std::vector<std::vector<vector_double::size_type>> res = {{}};
        BOOST_CHECK(kNN(points, 2u) == res);
    }
    {
        std::vector<vector_double> points = {{1.2, 1.2}, {1.2, 1.2}};
        std::vector<std::vector<vector_double::size_type>> res = {{1u}, {0u}};
        BOOST_CHECK(kNN(points, 2u) == res);
    }
    {
        std::vector<vector_double> points = {{1, 1}, {2, 2}, {3.1, 3.1}};
        std::vector<std::vector<vector_double::size_type>> res = {{1u, 2u}, {0u, 2u}, {1u, 0u}};
        BOOST_CHECK(kNN(points, 2u) == res);
    }

    // Some test cases
    {
        std::vector<vector_double> points = {{1, 1}, {2, 2}, {3.1, 3.1}, {4.2, 4.2}, {5.4, 5.4}};
        std::vector<std::vector<vector_double::size_type>> res
            = {{1u, 2u, 3u}, {0u, 2u, 3u}, {1u, 3u, 0u}, {2u, 4u, 1u}, {3u, 2u, 1u}};
        BOOST_CHECK(kNN(points, 3u) == res);
    }
    // throws
    {
        std::vector<vector_double> points = {{1, 1}, {2, 2}, {2, 3, 4}};
        BOOST_CHECK_THROW(kNN(points, 3u), std::invalid_argument);
    }
}
