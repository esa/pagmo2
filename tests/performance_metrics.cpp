/* Copyright 2017-2026 PaGMO development team

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

#define BOOST_TEST_MODULE performance_metrics_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <pagmo/types.hpp>
#include <pagmo/utils/hypervolume.hpp>
#include <pagmo/utils/performance_metrics.hpp>

using namespace pagmo;

namespace
{

const double nan_value = std::numeric_limits<double>::quiet_NaN();
const double inf_value = std::numeric_limits<double>::infinity();
const double max_value = std::numeric_limits<double>::max();
const double denorm_min_value = std::numeric_limits<double>::denorm_min();

// Random objective vectors with components uniformly distributed in [0, 1).
std::vector<vector_double> random_points(std::mt19937 &r_engine, unsigned n_points, unsigned n_obj)
{
    std::uniform_real_distribution<double> drng(0., 1.);
    std::vector<vector_double> retval(n_points, vector_double(n_obj));
    for (auto &p : retval) {
        for (auto &value : p) {
            value = drng(r_engine);
        }
    }
    return retval;
}

// Each point of the returned set is weakly dominated by the corresponding point of points.
std::vector<vector_double> worsened_points(std::mt19937 &r_engine, const std::vector<vector_double> &points)
{
    std::uniform_real_distribution<double> drng(0., 0.2);
    auto retval = points;
    for (auto &p : retval) {
        for (auto &value : p) {
            value += drng(r_engine);
        }
    }
    return retval;
}

} // namespace

// The expected values reproduce Tables 1-8 of Ishibuchi et al., "Modified distance calculation in
// generational distance and inverted generational distance", EMO 2015.
BOOST_AUTO_TEST_CASE(igd_plus_paper_examples_test)
{
    // Table 1 (Fig. 1): IGD+(A) = 3.550, IGD+(B) = 6.110.
    std::vector<vector_double> z = {{1, 0}, {0, 10}};
    BOOST_CHECK_SMALL(igd_plus({{2, 5}}, z) - (std::sqrt(26.) + 2.) / 2., 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{3, 9}}, z) - (std::sqrt(85.) + 3.) / 2., 1e-12);
    // Table 3 (Fig. 3): IGD+(D) = 1.707.
    BOOST_CHECK_SMALL(igd_plus({{2, 1}}, z) - (std::sqrt(2.) + 2.) / 2., 1e-12);

    // Table 2 (Fig. 2): IGD+(A) = 3.550, IGD+(B) = 7.171.
    z = {{0, 1}, {10, 0}};
    BOOST_CHECK_SMALL(igd_plus({{5, 2}}, z) - (std::sqrt(26.) + 2.) / 2., 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{11, 3}}, z) - (std::sqrt(125.) + std::sqrt(10.)) / 2., 1e-12);
    // Table 5 (Fig. 5): IGD+(B) = 4.854.
    BOOST_CHECK_SMALL(igd_plus({{6, 4}, {10, 3}}, z) - (std::sqrt(45.) + 3.) / 2., 1e-12);

    // Table 4 (Fig. 4): IGD+(A) = 1.483, IGD+(B) = 2.260.
    z = {{0, 10}, {1, 6}, {2, 2}, {6, 1}, {10, 0}};
    BOOST_CHECK_SMALL(igd_plus({{2, 4}, {3, 3}, {4, 2}}, z) - (6. + std::sqrt(2.)) / 5., 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{2, 8}, {4, 4}, {8, 2}}, z) - (4. + 2. * std::sqrt(5.) + 2. * std::sqrt(2.)) / 5.,
                      1e-12);

    // Table 6 (Fig. 6 (b)): only the objectives in which a solution is worse than the
    // reference point count, so the Euclidean distance sqrt(10) of A becomes 1.
    z = {{4, 4}};
    BOOST_CHECK_SMALL(igd_plus({{1, 5}}, z) - 1., 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{5, 6}}, z) - std::sqrt(5.), 1e-12);

    // Tables 7 and 8 (Figs. 8 and 9): IGD+ is weakly, but not strictly, Pareto compliant.
    z = {{0, 0}};
    const std::vector<vector_double> a = {{1, 8}, {2, 2}, {8, 1}};
    BOOST_CHECK_SMALL(igd_plus(a, z) - std::sqrt(8.), 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{4, 3}}, z) - 5., 1e-12);
    BOOST_CHECK_EQUAL(igd_plus(a, z), igd_plus({{2, 2}}, z));
}

BOOST_AUTO_TEST_CASE(igd_plus_test)
{
    // The distance goes from the reference front to the points, and it is averaged over the reference front.
    BOOST_CHECK_SMALL(igd_plus({{-1, 3}}, {{0, 0}}) - 3., 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{0, 0}}, {{-1, 3}}) - 1., 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{0, 0}, {10, 10}}, {{0, 0}}), 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{0, 0}}, {{0, 0}, {10, 10}}), 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{1, 1}}, {{0, 0}, {10, 10}}) - std::sqrt(2.) / 2., 1e-12);

    // Zero if and only if every point of the reference front is weakly dominated by some point.
    const std::vector<vector_double> z = {{0, 10}, {1, 6}, {2, 2}, {6, 1}, {10, 0}};
    BOOST_CHECK_EQUAL(igd_plus(z, z), 0.);
    BOOST_CHECK_EQUAL(igd_plus({{0, 0}}, z), 0.);
    BOOST_CHECK(igd_plus({{0, 10}, {1, 6}, {2, 2}, {6, 1}}, z) > 0.);

    // One and three objectives.
    BOOST_CHECK_SMALL(igd_plus({{2}}, {{1}, {3}}) - 0.5, 1e-12);
    const std::vector<vector_double> z3 = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
    BOOST_CHECK_SMALL(igd_plus({{0.5, 0.5, 0.5}}, z3) - std::sqrt(0.5), 1e-12);
    BOOST_CHECK_SMALL(igd_plus({{1, 0, 0}}, z3) - 2. / 3., 1e-12);

    // Duplicates and the order of the points do not matter.
    std::vector<vector_double> a = {{2, 4}, {3, 3}, {4, 2}};
    const auto value = igd_plus(a, z);
    BOOST_CHECK_EQUAL(igd_plus({{2, 4}, {3, 3}, {3, 3}, {4, 2}, {2, 4}}, z), value);
    std::reverse(a.begin(), a.end());
    BOOST_CHECK_EQUAL(igd_plus(a, z), value);
    auto z_rev = z;
    std::reverse(z_rev.begin(), z_rev.end());
    BOOST_CHECK_SMALL(igd_plus(a, z_rev) - value, 1e-12);

    // Very large and very small objective values do not overflow or underflow.
    BOOST_CHECK_EQUAL(igd_plus({{1e200}}, {{0}}), 1e200);
    BOOST_CHECK_EQUAL(igd_plus({{1e-200}}, {{0}}), 1e-200);
    BOOST_CHECK_CLOSE(igd_plus({{1e200, 1e200}}, {{0, 0}}), std::sqrt(2.) * 1e200, 1e-10);
    BOOST_CHECK_CLOSE(igd_plus({{1e-200, 1e-200}}, {{0, 0}}), std::sqrt(2.) * 1e-200, 1e-10);
    // Neither the sum of the distances nor the differences between the objective values overflow.
    BOOST_CHECK_EQUAL(igd_plus({{1e308, 0}}, {{0, 0}, {0, 1}}), 1e308);
    BOOST_CHECK_CLOSE(igd_plus({{1e308, 0}}, {{0, 0}, {0, 1}, {0, 2}}), 1e308, 1e-10);
    BOOST_CHECK_EQUAL(igd_plus({{1e308}}, {{-1e308}, {1e308}}), 1e308);
    BOOST_CHECK_EQUAL(igd_plus({{0, 0}}, {{-max_value, 0}, {0, -max_value}}), max_value);
    // The result overflows only if the IGD+ itself exceeds the largest finite double.
    BOOST_CHECK_EQUAL(igd_plus({{max_value}}, {{-max_value}}), inf_value);
    // The average does not exceed the largest distance because of rounding errors.
    BOOST_CHECK_EQUAL(igd_plus({{max_value}}, {{0}, {0}, {0}}), max_value);
    // Subnormal distances are rescaled, so that the result is rounded only once: 3 / 4 and sqrt(2) times the smallest
    // subnormal round to it.
    BOOST_CHECK_EQUAL(igd_plus({{denorm_min_value}}, {{0}, {0}, {0}, {denorm_min_value}}), denorm_min_value);
    BOOST_CHECK_EQUAL(igd_plus({{denorm_min_value, denorm_min_value}}, {{0, 0}}), denorm_min_value);
    // Large objective values which cancel out do not affect small distances.
    const auto tiny = std::ldexp(1., -1050);
    BOOST_CHECK_EQUAL(igd_plus({{tiny, max_value}}, {{0, max_value}, {0, max_value}, {0, max_value}}), tiny);

    // Weak Pareto compliance.
    std::mt19937 r_engine(42u);
    for (unsigned n_obj = 2u; n_obj <= 4u; ++n_obj) {
        const auto ref = random_points(r_engine, 30u, n_obj);
        const auto pa = random_points(r_engine, 20u, n_obj);
        const auto pb = worsened_points(r_engine, pa);
        BOOST_CHECK(igd_plus(pa, ref) <= igd_plus(pb, ref));
    }
}

BOOST_AUTO_TEST_CASE(igd_plus_throws_test)
{
    const std::vector<vector_double> z = {{0, 1}, {1, 0}};
    BOOST_CHECK_THROW(igd_plus({}, z), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{0, 0}}, {}), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({}, {}), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{}}, {{}}), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{0, 0}}, {{0, 1}, {1}}), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{0, 0}, {1, 1, 1}}, z), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{0, 0, 0}}, z), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{0, nan_value}}, z), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{inf_value, 0}}, z), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{0, 0}}, {{0, 1}, {-inf_value, 0}}), std::invalid_argument);
    BOOST_CHECK_THROW(igd_plus({{0, 0}}, {{nan_value, 1}}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(hypervolume_ratio_test)
{
    const vector_double r = {4, 4};
    const std::vector<vector_double> a = {{1, 3}, {2, 2}, {3, 1}};
    // HV(A, r) = 6 and HV(Z, r) = 9.
    BOOST_CHECK_SMALL(hypervolume_ratio(a, {{1, 1}}, r) - 6. / 9., 1e-12);
    BOOST_CHECK_SMALL(hypervolume_ratio(a, a, r) - 1., 1e-12);
    BOOST_CHECK_SMALL(hypervolume_ratio({{1, 1}}, a, r) - 9. / 6., 1e-12);
    // The ratio is not clamped when the points dominate a region not dominated by the reference front.
    BOOST_CHECK_SMALL(hypervolume_ratio({{0, 0}}, {{1, 1}}, {2, 2}) - 4., 1e-12);
    // Three objectives.
    BOOST_CHECK_SMALL(hypervolume_ratio({{1, 1, 1}}, {{0, 0, 0}}, {2, 2, 2}) - 1. / 8., 1e-12);

    // Points which are not strictly better than the reference point in every objective are discarded,
    // instead of raising an error as pagmo::hypervolume does.
    const std::vector<vector_double> outside = {{1, 1}, {5, 0}, {4, 0}, {0, 4}, {4, 4}, {5, 5}};
    BOOST_CHECK_THROW(hypervolume(outside).compute(r), std::invalid_argument);
    BOOST_CHECK_SMALL(hypervolume_ratio(outside, {{1, 1}}, r) - 1., 1e-12);
    BOOST_CHECK_SMALL(hypervolume_ratio({{1, 1}}, outside, r) - 1., 1e-12);
    BOOST_CHECK_EQUAL(hypervolume_ratio({{5, 0}, {4, 0}, {0, 4}, {4, 4}, {5, 5}}, {{1, 1}}, r), 0.);

    // The ratio is computed even if the hypervolumes themselves overflow or underflow.
    for (const auto scale : {1e300, 1e-300}) {
        auto scaled_a = a;
        for (auto &p : scaled_a) {
            for (auto &value : p) {
                value *= scale;
            }
        }
        BOOST_CHECK_SMALL(hypervolume_ratio(scaled_a, {{scale, scale}}, {4. * scale, 4. * scale}) - 6. / 9., 1e-12);
    }
    // The differences between the objective values do not overflow: HV(A, r) = 2 HV(Z, r).
    BOOST_CHECK_SMALL(hypervolume_ratio({{-1e308, 0}, {-1e308, 0}}, {{0, 0}}, {1e308, 1e-308}) - 2., 1e-12);
    BOOST_CHECK_SMALL(hypervolume_ratio({{-max_value, -max_value}}, {{0, 0}}, {max_value, max_value}) - 4., 1e-12);
    // Hypervolumes which do not overflow or underflow are not rescaled, even if the objective values span a wide range.
    BOOST_CHECK_CLOSE(hypervolume_ratio({{0, 0}}, {{-1e200, 0}, {0, -1e200}}, {1, 1}), 1. / 2e200, 1e-10);
    BOOST_CHECK_CLOSE(hypervolume_ratio({{-1e200, 0}, {0, -1e200}}, {{0, 0}}, {1, 1}), 2e200, 1e-10);
    BOOST_CHECK_SMALL(hypervolume_ratio({{-1e308, 0}}, {{0, -1e308}}, {1e-308, 1e-308}) - 1., 1e-12);
    const vector_double unit_box(22u, 1e15 - 1);
    BOOST_CHECK_SMALL(hypervolume_ratio({unit_box}, {unit_box}, vector_double(22u, 1e15)) - 1., 1e-12);
    // An underflowing hypervolume of the points is recovered by the rescaling: HV(A, r) = 1e-400, HV(Z, r) = 2e-200.
    BOOST_CHECK_CLOSE(hypervolume_ratio({{-1e-200, -1e-200}}, {{-1, -1e-200}, {-1e-200, -1}}, {0, 0}), 0.5e-200, 1e-10);
    // The rescaled hypervolumes are used if they are more accurate, even if they are subnormal.
    const auto u = std::ldexp(1., -536);
    BOOST_CHECK_SMALL(hypervolume_ratio({{-0.25, -0.25, -u, -u}, {-u, -u, -0.25, -0.25}},
                                        {{-0.25, -0.25, -4 * u, -4 * u}, {-4 * u, -4 * u, -0.25, -0.25}}, {0, 0, 0, 0})
                          - 0.0625,
                      1e-12);
    // Hypervolumes in 1100 objectives which overflow or underflow, but not after rescaling, whatever the widths.
    const vector_double big_box(1100u, -0.99);
    BOOST_CHECK_SMALL(hypervolume_ratio({big_box}, {big_box}, vector_double(1100u, 0.99)) - 1., 1e-12);
    BOOST_CHECK_SMALL(
        hypervolume_ratio({vector_double(1100u, -2.)}, {vector_double(1100u, -2.)}, vector_double(1100u, 0.)) - 1.,
        1e-12);
    BOOST_CHECK_SMALL(
        hypervolume_ratio({vector_double(1100u, 0.)}, {vector_double(1100u, 0.)}, vector_double(1100u, 0.5)) - 1.,
        1e-12);
    BOOST_CHECK_CLOSE(
        hypervolume_ratio({vector_double(1100u, -2.)}, {vector_double(1100u, -2.001)}, vector_double(1100u, 0.)),
        std::pow(2. / 2.001, 1100.), 1e-8);
    // The ratio is zero if no point is left, even if the hypervolume of the reference front, 0.5^1100, underflows.
    BOOST_CHECK_EQUAL(
        hypervolume_ratio({vector_double(1100u, 0.)}, {vector_double(1100u, -0.5)}, vector_double(1100u, 0.)), 0.);
    // Objective values equal to -DBL_MAX, which the default algorithm for three objectives uses as a sentinel.
    const std::vector<vector_double> sentinel = {{-1, -0.5, -2}, {-max_value, -0.5, -1}};
    BOOST_CHECK_SMALL(hypervolume_ratio(sentinel, sentinel, {0, 0, 0}) - 1., 1e-12);
    // HV(A, r) = DBL_MAX and HV(Z, r) = DBL_MAX / 2 + 0.5.
    BOOST_CHECK_CLOSE(hypervolume_ratio({{-max_value, -1, -1}}, sentinel, {0, 0, 0}), 2., 1e-10);

    // Same values as the ratio of the hypervolumes computed by pagmo::hypervolume (hv2d, hv3d and hvwfg).
    std::mt19937 r_engine(42u);
    for (unsigned n_obj = 2u; n_obj <= 4u; ++n_obj) {
        const auto ref = random_points(r_engine, 30u, n_obj);
        const auto pa = random_points(r_engine, 20u, n_obj);
        const vector_double ref_point(n_obj, 1.1);
        const auto expected = hypervolume(pa).compute(ref_point) / hypervolume(ref).compute(ref_point);
        BOOST_CHECK_SMALL(hypervolume_ratio(pa, ref, ref_point) - expected, 1e-12);
        // Weak Pareto compliance.
        const auto pb = worsened_points(r_engine, pa);
        BOOST_CHECK(hypervolume_ratio(pa, ref, ref_point) >= hypervolume_ratio(pb, ref, ref_point));
    }
}

BOOST_AUTO_TEST_CASE(hypervolume_ratio_throws_test)
{
    const vector_double r = {4, 4};
    const std::vector<vector_double> z = {{1, 1}};
    BOOST_CHECK_THROW(hypervolume_ratio({{1}}, {{1}}, {2}), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio({{}}, {{}}, {}), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio({}, z, r), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio(z, {}, r), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio({{1, 1, 1}}, z, r), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio(z, {{1, 1}, {1, 1, 1}}, r), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio(z, z, {4, 4, 4}), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio({{1, nan_value}}, z, r), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio(z, {{-inf_value, 1}}, r), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio(z, z, {4, inf_value}), std::invalid_argument);
    BOOST_CHECK_THROW(hypervolume_ratio(z, z, {nan_value, 4}), std::invalid_argument);

    // Zero hypervolume of the reference front.
    const auto zero_hv = [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), "The hypervolume of the reference front");
    };
    BOOST_CHECK_EXCEPTION(hypervolume_ratio(z, {{4, 0}}, r), std::invalid_argument, zero_hv);
    BOOST_CHECK_EXCEPTION(hypervolume_ratio(z, {{4, 4}, {5, 1}}, r), std::invalid_argument, zero_hv);

    // HV(A, r) = 4^1100 and HV(Z, r) = 1: the ratio exceeds the range of double precision, and the hypervolumes cannot
    // be represented even after rescaling the objectives.
    const auto unrepresentable_hv = [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), "cannot be represented in double precision");
    };
    BOOST_CHECK_EXCEPTION(
        hypervolume_ratio({vector_double(1100u, -4.)}, {vector_double(1100u, -1.)}, vector_double(1100u, 0.)),
        std::invalid_argument, unrepresentable_hv);
}

BOOST_AUTO_TEST_CASE(normalized_igd_plus_test)
{
    // IGD+({r}; Z) = 5.
    const std::vector<vector_double> z = {{0, 0}};
    const vector_double r = {3, 4};
    BOOST_CHECK_SMALL(normalized_igd_plus({{0, 2}}, z, r) - 0.4, 1e-12);
    BOOST_CHECK_SMALL(normalized_igd_plus({r}, z, r) - 1., 1e-12);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{0, 0}}, z, r), 0.);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{-1, -1}}, z, r), 0.);
    // The value is not clamped, and it may exceed one even if no point is dominated by the reference point.
    BOOST_CHECK_SMALL(normalized_igd_plus({{0, 2}}, z, {1, 1}) - std::sqrt(2.), 1e-12);
    BOOST_CHECK_SMALL(normalized_igd_plus({{6, 8}}, z, r) - 2., 1e-12);

    // Table 1 of Ishibuchi et al. (EMO 2015), normalized by the reference point (11, 11).
    BOOST_CHECK_SMALL(normalized_igd_plus({{2, 5}}, {{1, 0}, {0, 10}}, {11, 11})
                          - (std::sqrt(26.) + 2.) / (std::sqrt(221.) + std::sqrt(122.)),
                      1e-12);

    // The denominator is positive as long as the reference point is worse than some point of the reference front
    // in some objective: here IGD+({r}; Z) = sqrt(2) / 2.
    BOOST_CHECK_SMALL(normalized_igd_plus({{1, 1}}, {{0, 0}, {2, 2}}, {1, 1}) - 1., 1e-12);
    BOOST_CHECK_SMALL(normalized_igd_plus({{1, 2}}, {{0, 0}, {2, 2}}, {1, 1}) - std::sqrt(5.) / std::sqrt(2.), 1e-12);

    // One objective: IGD+({r}; Z) = 2 and IGD+(A; Z) = 0.5.
    BOOST_CHECK_SMALL(normalized_igd_plus({{2}}, {{1}, {3}}, {4}) - 0.25, 1e-12);

    // Very large and very small objective values do not overflow or underflow.
    BOOST_CHECK_SMALL(normalized_igd_plus({{1e200, 1e200}}, z, {2e200, 2e200}) - 0.5, 1e-12);
    BOOST_CHECK_SMALL(normalized_igd_plus({{1e-200, 1e-200}}, z, {2e-200, 2e-200}) - 0.5, 1e-12);
    // Neither the sums of the distances nor the differences between the objective values overflow, in the numerator
    // as well as in the denominator.
    BOOST_CHECK_EQUAL(normalized_igd_plus({{1e308, 0}}, {{0, 0}, {0, 1}}, {1e308, 0}), 1.);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{0}}, {{-1e308}}, {1e308}), 0.5);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{1e308}}, {{-1e308}}, {0}), 2.);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{max_value}}, {{-max_value}}, {max_value}), 1.);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{max_value}}, {{0}, {0}, {0}}, {max_value}), 1.);
    // The denominator does not underflow: both IGD+ values are 1 / 2 of the smallest subnormal.
    BOOST_CHECK_EQUAL(normalized_igd_plus({{denorm_min_value}}, {{0}, {denorm_min_value}}, {denorm_min_value}), 1.);
    // Subnormal distances are rescaled before being rounded, even if irrelevant large objective values are present.
    BOOST_CHECK_EQUAL(normalized_igd_plus({{denorm_min_value, denorm_min_value}}, {{0, 0}},
                                          {2. * denorm_min_value, 2. * denorm_min_value}),
                      0.5);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{denorm_min_value, denorm_min_value}, {max_value, max_value}}, {{0, 0}},
                                          {2. * denorm_min_value, 2. * denorm_min_value}),
                      0.5);
    BOOST_CHECK_SMALL(normalized_igd_plus({{denorm_min_value, denorm_min_value}, {max_value, max_value}}, {{0, 0}},
                                          {denorm_min_value, 0})
                          - std::sqrt(2.),
                      1e-12);
    // A ratio smaller than the smallest subnormal is rounded to the nearest double: 2 / 3 of it rounds to it.
    BOOST_CHECK_EQUAL(normalized_igd_plus({{denorm_min_value}}, {{0}}, {1.5}), denorm_min_value);
    // Large objective values which cancel out do not affect small distances.
    const auto tiny = std::ldexp(1., -1050);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{tiny / 2., max_value}}, {{0, max_value}}, {tiny, max_value}), 0.5);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{3. * tiny, max_value}}, {{0, max_value}, {0, max_value}, {0, max_value}},
                                          {4. * tiny, max_value}),
                      0.75);
    BOOST_CHECK_EQUAL(normalized_igd_plus({{0, max_value}}, {{0, max_value}}, {denorm_min_value, max_value}), 0.);

    std::mt19937 r_engine(42u);
    for (unsigned n_obj = 2u; n_obj <= 4u; ++n_obj) {
        const auto ref = random_points(r_engine, 30u, n_obj);
        auto pa = random_points(r_engine, 20u, n_obj);
        const vector_double ref_point(n_obj, 1.1);
        // Same values as the ratio of the IGD+ values.
        BOOST_CHECK_SMALL(normalized_igd_plus(pa, ref, ref_point) - igd_plus(pa, ref) / igd_plus({ref_point}, ref),
                          1e-12);
        // Weak Pareto compliance.
        const auto pb = worsened_points(r_engine, pa);
        BOOST_CHECK(normalized_igd_plus(pa, ref, ref_point) <= normalized_igd_plus(pb, ref, ref_point));
        // At most one if some point weakly dominates the reference point.
        pa.push_back(ref_point);
        BOOST_CHECK(normalized_igd_plus(pa, ref, ref_point) <= 1.);
    }
}

BOOST_AUTO_TEST_CASE(normalized_igd_plus_throws_test)
{
    const std::vector<vector_double> z = {{0, 1}, {1, 0}};
    const vector_double r = {2, 2};
    BOOST_CHECK_THROW(normalized_igd_plus({{}}, {{}}, {}), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus({}, z, r), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus(z, {}, r), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus({{1, 1, 1}}, z, r), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus(z, {{0, 1}, {1}}, r), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus(z, z, {2, 2, 2}), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus({{1, nan_value}}, z, r), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus(z, {{inf_value, 1}}, r), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus(z, z, {2, -inf_value}), std::invalid_argument);
    BOOST_CHECK_THROW(normalized_igd_plus(z, z, {nan_value, 2}), std::invalid_argument);

    // Zero denominator: the reference point weakly dominates every point of the reference front.
    const auto zero_igd = [](const std::invalid_argument &ia) {
        return boost::contains(ia.what(), "the reference point weakly dominates every objective vector");
    };
    BOOST_CHECK_EXCEPTION(normalized_igd_plus(z, {{1, 1}}, {1, 1}), std::invalid_argument, zero_igd);
    BOOST_CHECK_EXCEPTION(normalized_igd_plus(z, {{2, 3}}, {1, 1}), std::invalid_argument, zero_igd);
    BOOST_CHECK_EXCEPTION(normalized_igd_plus(z, {{1, 1}, {2, 3}}, {1, 1}), std::invalid_argument, zero_igd);
    BOOST_CHECK_EXCEPTION(normalized_igd_plus(z, z, {0, 0}), std::invalid_argument, zero_igd);
}
