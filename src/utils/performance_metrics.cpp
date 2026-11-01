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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_hvwfg.hpp>
#include <pagmo/utils/hypervolume.hpp>
#include <pagmo/utils/performance_metrics.hpp>

namespace pagmo
{

namespace
{

// Checks that a set of objective vectors is not empty and that it contains only
// finite objective vectors of dimension n_obj.
void check_point_set(const std::vector<vector_double> &points, const std::string &set_name,
                     const std::string &func_name, vector_double::size_type n_obj)
{
    if (points.empty()) {
        pagmo_throw(std::invalid_argument, "The " + set_name + " passed to " + func_name
                                               + "() is empty: at least one objective vector is required");
    }
    for (decltype(points.size()) i = 0u; i < points.size(); ++i) {
        if (points[i].size() != n_obj) {
            pagmo_throw(std::invalid_argument, "The objective vector at index " + std::to_string(i) + " of the "
                                                   + set_name + " passed to " + func_name + "() has dimension "
                                                   + std::to_string(points[i].size()) + ", while a dimension of "
                                                   + std::to_string(n_obj) + " was expected");
        }
        for (auto value : points[i]) {
            if (!std::isfinite(value)) {
                pagmo_throw(std::invalid_argument, "The objective vector at index " + std::to_string(i) + " of the "
                                                       + set_name + " passed to " + func_name
                                                       + "() contains non-finite values");
            }
        }
    }
}

// Checks that the reference point has at least min_n_obj components, all of them finite.
void check_ref_point(const vector_double &ref_point, vector_double::size_type min_n_obj, const std::string &func_name)
{
    if (ref_point.size() < min_n_obj) {
        pagmo_throw(std::invalid_argument, "The reference point passed to " + func_name + "() has dimension "
                                               + std::to_string(ref_point.size()) + ", while a dimension of at least "
                                               + std::to_string(min_n_obj) + " is required");
    }
    for (auto value : ref_point) {
        if (!std::isfinite(value)) {
            pagmo_throw(std::invalid_argument,
                        "The reference point passed to " + func_name + "() contains non-finite values");
        }
    }
}

// 2^exp max(a - z, 0). Scaling up is exact (the difference itself is exact if it is subnormal), and the difference
// is halved before scaling if it overflows.
double scaled_difference(double a, double z, int exp)
{
    const auto diff = a - z;
    if (exp == 0 || diff <= 0.) {
        return std::max(diff, 0.);
    }
    if (std::isfinite(diff)) {
        return std::ldexp(diff, exp);
    }
    return std::ldexp(std::ldexp(a, -1) - std::ldexp(z, -1), exp + 1);
}

// Modified distance d+(z, a) of Ishibuchi et al. (minimization), i.e., the Euclidean distance from z to the region
// weakly dominated by a, multiplied by 2^exp. std::hypot() avoids the spurious overflow and underflow of the squared
// components for very large or very small objectives.
double modified_distance(const vector_double &z, const vector_double &a, int exp)
{
    assert(z.size() == a.size());
    double retval = 0.;
    for (decltype(z.size()) i = 0u; i < z.size(); ++i) {
        retval = std::hypot(retval, scaled_difference(a[i], z[i], exp));
    }
    return retval;
}

// Sum, over the points z of reference_front, of the modified distance from z to the nearest point of points (first),
// and largest of these distances (second), multiplied by 2^exp, without input validation. The sum is infinite if it
// overflows.
std::pair<double, double> igd_plus_sum(const std::vector<vector_double> &points,
                                       const std::vector<vector_double> &reference_front, int exp)
{
    assert(!points.empty());
    assert(!reference_front.empty());
    double sum = 0., max_distance = 0.;
    for (const auto &z : reference_front) {
        auto min_distance = std::numeric_limits<double>::infinity();
        for (const auto &a : points) {
            min_distance = std::min(min_distance, modified_distance(z, a, exp));
        }
        sum += min_distance;
        max_distance = std::max(max_distance, min_distance);
    }
    return {sum, max_distance};
}

// Whether a sum of distances, which is either zero or positive, overflows or is subnormal.
bool is_out_of_range(double sum)
{
    return sum != 0. && !std::isnormal(sum);
}

// Largest absolute value in a set of objective vectors.
double max_abs_value(const std::vector<vector_double> &points)
{
    double retval = 0.;
    for (const auto &p : points) {
        for (auto value : p) {
            retval = std::max(retval, std::abs(value));
        }
    }
    return retval;
}

// Exponent of the power of two by which the differences between the objective values are multiplied when the sums of
// the modified distances overflow or are subnormal. If the sums overflow, bound is the largest absolute objective
// value, so that the differences are smaller than 2^(e + 1), where bound < 2^e. If the sums are subnormal, bound is
// the largest of the (rounded) distances, so that the exact distances are smaller than n_obj 2^e. In both cases, after
// scaling, the distances are smaller than n_obj 2^(1021 - k), where 2^k > n_ref n_obj, and their sums smaller than
// 2^1021.
int distance_scale_exp(double bound, vector_double::size_type n_ref, vector_double::size_type n_obj)
{
    int k = 0, exp = 0;
    std::frexp(static_cast<double>(n_ref) * static_cast<double>(n_obj), &k);
    std::frexp(bound, &exp);
    return 1020 - k - exp;
}

// Copy of points in which the i-th objective value is multiplied by 2^exps[i].
std::vector<vector_double> scale_objectives(std::vector<vector_double> points, const std::vector<int> &exps)
{
    for (auto &p : points) {
        assert(p.size() == exps.size());
        for (decltype(p.size()) i = 0u; i < p.size(); ++i) {
            p[i] = std::ldexp(p[i], exps[i]);
        }
    }
    return points;
}

// Objective vectors of points which are strictly better than ref_point in every objective, i.e., those having
// a nonzero contribution to the hypervolume.
std::vector<vector_double> strictly_better_points(const std::vector<vector_double> &points,
                                                  const vector_double &ref_point)
{
    std::vector<vector_double> retval;
    std::copy_if(points.begin(), points.end(), std::back_inserter(retval), [&ref_point](const vector_double &p) {
        return std::equal(p.begin(), p.end(), ref_point.begin(), std::less<double>());
    });
    return retval;
}

// Hypervolume of points with respect to ref_point, or zero if points is empty. The default algorithm for three
// objectives (hv3d) uses -DBL_MAX as a sentinel coordinate, hence hvwfg is used instead if an objective value equals
// it.
double hypervolume_or_zero(const std::vector<vector_double> &points, const vector_double &ref_point)
{
    if (points.empty()) {
        return 0.;
    }
    const hypervolume hv(points);
    if (ref_point.size() == 3u && std::any_of(points.begin(), points.end(), [](const vector_double &p) {
            return std::find(p.begin(), p.end(), std::numeric_limits<double>::lowest()) != p.end();
        })) {
        hvwfg algo;
        return hv.compute(ref_point, algo);
    }
    return hv.compute(ref_point);
}

// Base-2 logarithm of the largest width ref_point[i] - p[i] over the points p of a nonempty set whose objective
// vectors are all strictly better than ref_point.
double log2_width(const std::vector<vector_double> &points, const vector_double &ref_point, vector_double::size_type i)
{
    assert(!points.empty());
    auto min_value = points[0][i];
    for (const auto &p : points) {
        min_value = std::min(min_value, p[i]);
    }
    const auto width = ref_point[i] - min_value;
    // The width may overflow, but its half does not.
    return std::isfinite(width) ? std::log2(width)
                                : std::log2(std::ldexp(ref_point[i], -1) - std::ldexp(min_value, -1)) + 1.;
}

} // namespace

/// Hypervolume ratio
/**
 * Computes the hypervolume ratio (HVR) of a set of objective vectors \f$A\f$ with respect to a reference front
 * \f$Z\f$ (e.g., the Pareto front of the problem, or the best known approximation of it) and to a reference point
 * \f$\mathbf r\f$:
 *
 * \f[
 *   \mathrm{HVR}(A; Z, \mathbf r) = \frac{\mathrm{HV}(A, \mathbf r)}{\mathrm{HV}(Z, \mathbf r)},
 * \f]
 *
 * where \f$\mathrm{HV}(S, \mathbf r)\f$ is the hypervolume of the region dominated by the set \f$S\f$ and bounded by
 * \f$\mathbf r\f$, as computed by pagmo::hypervolume::compute(). The denominator is the hypervolume of the reference
 * front, and the same reference point \f$\mathbf r\f$ is shared by the numerator and the denominator. Hence, ratios
 * are comparable only if they are computed with the same \f$Z\f$ and \f$\mathbf r\f$.
 *
 * The objective vectors which are not strictly better than \f$\mathbf r\f$ in every objective (i.e., those having
 * at least one component \f$a_i \geq r_i\f$) do not contribute to the hypervolume. They are discarded from both
 * \f$A\f$ and \f$Z\f$ before computing the hypervolumes, rather than raising an error as pagmo::hypervolume does for
 * points lying outside the reference point boundary. If no point of \f$A\f$ is left, the ratio is zero.
 *
 * The hypervolumes are computed in double precision by pagmo::hypervolume. If they overflow or underflow, they are
 * recomputed after multiplying each objective by a power of two, chosen so that the volumes of the boxes spanned by the
 * largest widths \f$r_i - a_i\f$ of \f$A\f$ and \f$r_i - z_i\f$ of \f$Z\f$ are balanced around one (as far as this
 * is possible without the objective values overflowing): this multiplies both hypervolumes by the same factor and
 * leaves their ratio unchanged, and the rescaled hypervolumes are used if they are more accurate. The ratio may still
 * be inaccurate (or zero) if the hypervolumes cannot be represented as normal double precision numbers even after
 * rescaling, or if intermediate results of the hypervolume algorithms overflow or underflow (e.g., when the widths
 * \f$r_i - a_i\f$ of the dominated boxes span many orders of magnitude), even though the final hypervolumes are normal
 * numbers.
 *
 * Larger values are better. The ratio is not clamped: it is one when \f$A\f$ and \f$Z\f$ dominate the same region,
 * and it may exceed one when \f$A\f$ dominates a region which is not dominated by \f$Z\f$ (this may happen when
 * \f$Z\f$ is only an approximation of the Pareto front).
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    All objectives are assumed to be minimized, consistently with the rest of pagmo. Objectives to be maximized
 *    must be negated in ``points``, ``reference_front`` and ``ref_point`` before calling this function.
 *
 * \endverbatim
 *
 * @param points the objective vectors \f$A\f$ to be evaluated.
 * @param reference_front the objective vectors of the reference front \f$Z\f$.
 * @param ref_point the reference point \f$\mathbf r\f$, shared by both hypervolume computations.
 *
 * @return the hypervolume ratio of \p points with respect to \p reference_front and \p ref_point.
 *
 * @throws std::invalid_argument if \p ref_point has fewer than two components, if \p points or \p reference_front is
 * empty, if the dimension of an objective vector differs from the dimension of \p ref_point, or if an objective
 * vector or \p ref_point contains non-finite values
 * @throws std::invalid_argument if no point of \p reference_front is strictly better than \p ref_point in every
 * objective, or if the hypervolumes cannot be represented in double precision even after rescaling the objectives
 * @throws unspecified any exception thrown by pagmo::hypervolume
 */
double hypervolume_ratio(const std::vector<vector_double> &points, const std::vector<vector_double> &reference_front,
                         const vector_double &ref_point)
{
    check_ref_point(ref_point, 2u, "hypervolume_ratio");
    check_point_set(points, "set of points", "hypervolume_ratio", ref_point.size());
    check_point_set(reference_front, "reference front", "hypervolume_ratio", ref_point.size());

    // The objective vectors which do not contribute to the hypervolumes are discarded.
    const auto inside_points = strictly_better_points(points, ref_point);
    const auto inside_front = strictly_better_points(reference_front, ref_point);
    if (inside_front.empty()) {
        pagmo_throw(std::invalid_argument,
                    "The hypervolume of the reference front passed to hypervolume_ratio() is zero, as none of its "
                    "objective vectors is strictly better than the reference point in every objective");
    }
    if (inside_points.empty()) {
        return 0.;
    }

    // The ratio can be computed if the hypervolume of the reference front is positive and finite and the hypervolume of
    // the points is finite. Both hypervolumes are positive in exact arithmetic, and the accuracy of their computed
    // values is measured by the smaller of their binary exponents: they are not affected by overflow or underflow of
    // the final values if both are normal numbers (intermediate results of the hypervolume algorithms may still
    // overflow or underflow).
    const auto computable = [](double points_hv, double front_hv) {
        return std::isfinite(points_hv) && std::isfinite(front_hv) && front_hv > 0.;
    };
    const auto accuracy = [&computable](double points_hv, double front_hv) {
        if (!computable(points_hv, front_hv)) {
            return std::numeric_limits<int>::min();
        }
        return points_hv > 0. ? std::min(std::ilogb(points_hv), std::ilogb(front_hv))
                              : std::numeric_limits<int>::min() + 1;
    };
    auto points_hv = hypervolume_or_zero(inside_points, ref_point);
    auto front_hv = hypervolume_or_zero(inside_front, ref_point);
    if (accuracy(points_hv, front_hv) < std::numeric_limits<double>::min_exponent - 1) {
        // The hypervolumes overflow or underflow: they are recomputed after multiplying each objective by a power of
        // two, close to the reciprocal of the geometric mean of the largest widths r_i - x_i of the points and of the
        // reference front, while the largest absolute value stays below 2^1022. The exponents are rounded
        // cumulatively, so that their sum, i.e., the exponent of the factor multiplying both hypervolumes, is within
        // one half of the exact one: the volumes of the boxes spanned by the largest widths of the two sets are then
        // balanced around one, whatever the number of objectives, unless the cap on the largest absolute values lowers
        // some exponents and the following objectives cannot compensate. Multiplying by a power of two is exact unless
        // the result is subnormal.
        std::vector<int> exps(ref_point.size());
        double exact_exp_sum = 0.;
        long exp_sum = 0;
        for (decltype(ref_point.size()) i = 0u; i < ref_point.size(); ++i) {
            exact_exp_sum -= (log2_width(inside_points, ref_point, i) + log2_width(inside_front, ref_point, i)) / 2.;
            exps[i] = static_cast<int>(std::lround(exact_exp_sum) - exp_sum);
            auto max_value = std::abs(ref_point[i]);
            for (const auto &p : inside_points) {
                max_value = std::max(max_value, std::abs(p[i]));
            }
            for (const auto &p : inside_front) {
                max_value = std::max(max_value, std::abs(p[i]));
            }
            int max_exp = 0;
            std::frexp(max_value, &max_exp);
            exps[i] = std::min(exps[i], 1022 - max_exp);
            exp_sum += exps[i];
        }
        const auto scaled_ref_point = scale_objectives({ref_point}, exps)[0];
        // Subnormal results may turn some objective vectors into vectors no longer strictly better than the
        // reference point, hence the filtering is repeated.
        const auto scaled_points_hv = hypervolume_or_zero(
            strictly_better_points(scale_objectives(inside_points, exps), scaled_ref_point), scaled_ref_point);
        const auto scaled_front_hv = hypervolume_or_zero(
            strictly_better_points(scale_objectives(inside_front, exps), scaled_ref_point), scaled_ref_point);
        // The rescaled hypervolumes are used if they are more accurate than the original ones.
        if (accuracy(scaled_points_hv, scaled_front_hv) > accuracy(points_hv, front_hv)) {
            points_hv = scaled_points_hv;
            front_hv = scaled_front_hv;
        }
    }
    if (!computable(points_hv, front_hv)) {
        pagmo_throw(std::invalid_argument,
                    "The hypervolumes computed by hypervolume_ratio() cannot be represented in double precision, "
                    "even after rescaling the objectives");
    }
    return points_hv / front_hv;
}

/// Modified inverted generational distance (IGD+)
/**
 * Computes the modified inverted generational distance (IGD+) of a set of objective vectors \f$A\f$ with respect to a
 * reference front \f$Z\f$ (e.g., the Pareto front of the problem, or the best known approximation of it):
 *
 * \f[
 *   \mathrm{IGD}^+(A; Z) = \frac{1}{|Z|} \sum_{\mathbf z \in Z} \min_{\mathbf a \in A} d^+(\mathbf z, \mathbf a),
 * \f]
 *
 * where the modified distance from \f$\mathbf z\f$ to \f$\mathbf a\f$ is
 *
 * \f[
 *   d^+(\mathbf z, \mathbf a) = \sqrt{\sum_{i=1}^{m} \left(\max\{a_i - z_i, 0\}\right)^2},
 * \f]
 *
 * and \f$m\f$ is the number of objectives. The modified distance \f$d^+(\mathbf z, \mathbf a)\f$ goes from the
 * point \f$\mathbf z\f$ of the reference front to the objective vector \f$\mathbf a\f$, and it accounts only for the
 * objectives in which \f$\mathbf a\f$ is worse than \f$\mathbf z\f$: it is the Euclidean distance from
 * \f$\mathbf z\f$ to the region weakly dominated by \f$\mathbf a\f$. Each point of the reference front is matched to
 * its nearest objective vector in \f$A\f$, and the distances are averaged over the points of the reference front (not
 * over the points of \f$A\f$).
 *
 * Smaller values are better. In exact arithmetic, the IGD+ is zero if and only if every point of \f$Z\f$ is weakly
 * dominated by some point of \f$A\f$.
 *
 * If the sum of the distances overflows or is subnormal in double precision, the distances are recomputed after
 * multiplying the differences between the objective values by a power of two, chosen from the magnitude of the
 * objective values if the sum overflows, and from the magnitude of the (subnormal) distances otherwise. Hence, up to
 * rounding errors, the result is infinite only if the IGD+ exceeds the largest finite double, and it is zero only if
 * the IGD+ is zero or smaller than the smallest positive double.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    All objectives are assumed to be minimized, consistently with the rest of pagmo. Objectives to be maximized
 *    must be negated in both ``points`` and ``reference_front`` before calling this function.
 *
 * .. note::
 *
 *    Unlike the original IGD, IGD+ is weakly Pareto compliant: if every point of a set :math:`B` is weakly
 *    dominated by some point of a set :math:`A`, then :math:`\mathrm{IGD}^+(A; Z) \leq \mathrm{IGD}^+(B; Z)`.
 *
 * \endverbatim
 *
 * See: H. Ishibuchi, H. Masuda, Y. Tanigaki and Y. Nojima, Modified distance calculation in generational distance and
 * inverted generational distance, Proceedings of the 8th International Conference on Evolutionary Multi-Criterion
 * Optimization (EMO 2015), Part II, pp. 110-125, 2015, https://doi.org/10.1007/978-3-319-15892-1_8.
 *
 * Complexity is \f$ O(M N_A N_Z)\f$ where \f$M\f$ is the number of objectives, and \f$N_A\f$ and \f$N_Z\f$ are the
 * number of points in \f$A\f$ and \f$Z\f$.
 *
 * @param points the objective vectors \f$A\f$ to be evaluated.
 * @param reference_front the objective vectors of the reference front \f$Z\f$.
 *
 * @return the IGD+ of \p points with respect to \p reference_front.
 *
 * @throws std::invalid_argument if \p points or \p reference_front is empty, if the objective vectors do not all
 * have the same nonzero dimension, or if an objective vector contains non-finite values
 */
double igd_plus(const std::vector<vector_double> &points, const std::vector<vector_double> &reference_front)
{
    const vector_double::size_type n_obj = reference_front.empty() ? 0u : reference_front[0].size();
    check_point_set(reference_front, "reference front", "igd_plus", n_obj);
    if (n_obj == 0u) {
        pagmo_throw(std::invalid_argument, "The objective vectors passed to igd_plus() have dimension 0, while a "
                                           "dimension of at least 1 is required");
    }
    check_point_set(points, "set of points", "igd_plus", n_obj);

    auto sum = igd_plus_sum(points, reference_front, 0);
    int exp = 0;
    if (is_out_of_range(sum.first)) {
        // The differences are scaled down if the sum overflows, and scaled up (exactly) if it is subnormal.
        const auto bound
            = std::isfinite(sum.first) ? sum.second : std::max(max_abs_value(points), max_abs_value(reference_front));
        exp = distance_scale_exp(bound, reference_front.size(), n_obj);
        sum = igd_plus_sum(points, reference_front, exp);
    }
    // The average is capped at the largest distance, which it could exceed only because of rounding errors.
    return std::ldexp(std::min(sum.first / static_cast<double>(reference_front.size()), sum.second), -exp);
}

/// Normalized modified inverted generational distance (NIGD+)
/**
 * Computes the IGD+ of a set of objective vectors \f$A\f$ with respect to a reference front \f$Z\f$ (see
 * pagmo::igd_plus()), normalized by the IGD+ attained, with respect to the same reference front, by the set
 * \f$\{\mathbf r\}\f$ whose only point is the reference point \f$\mathbf r\f$:
 *
 * \f[
 *   \mathrm{NIGD}^+(A; Z, \mathbf r) = \frac{\mathrm{IGD}^+(A; Z)}{\mathrm{IGD}^+(\{\mathbf r\}; Z)},
 * \f]
 *
 * where the denominator is
 *
 * \f[
 *   \mathrm{IGD}^+(\{\mathbf r\}; Z) = \frac{1}{|Z|} \sum_{\mathbf z \in Z}
 *   \sqrt{\sum_{i=1}^{m} \left(\max\{r_i - z_i, 0\}\right)^2}.
 * \f]
 *
 * The reference point \f$\mathbf r\f$ (typically the same reference point used for the hypervolume, see
 * pagmo::hypervolume_ratio()) is the normalization bound: the denominator is the IGD+ of the baseline set
 * \f$\{\mathbf r\}\f$. The objectives are not rescaled individually, so they keep their original scales in both the
 * numerator and the denominator, and objectives with larger ranges weigh more in the distances. Since the denominator
 * depends only on \f$Z\f$ and \f$\mathbf r\f$, NIGD+ inherits the weak Pareto compliance of IGD+ when \f$Z\f$ and
 * \f$\mathbf r\f$ are fixed.
 *
 * The denominator is zero, and the normalization undefined, if and only if \f$\mathbf r\f$ weakly dominates every
 * point of the reference front (i.e., \f$r_i \leq z_i\f$ for every objective \f$i\f$ and every
 * \f$\mathbf z \in Z\f$). In that case an exception is raised.
 *
 * Smaller values are better. In exact arithmetic, the NIGD+ is zero if and only if every point of \f$Z\f$ is weakly
 * dominated by some point of \f$A\f$ (in double precision, a sufficiently small positive ratio can be rounded to
 * zero), and it is one when \f$A = \{\mathbf r\}\f$. The result is not clamped: it is at most one
 * whenever some point of \f$A\f$ weakly dominates \f$\mathbf r\f$, but otherwise it may exceed one, even if no point
 * of \f$A\f$ is dominated by \f$\mathbf r\f$ (e.g., \f$Z = \{(0, 0)\}\f$, \f$\mathbf r = (1, 1)\f$ and
 * \f$A = \{(0, 2)\}\f$ yield \f$\sqrt{2}\f$).
 *
 * The ratio is computed as the ratio of the sums of the distances, since the factor \f$1 / |Z|\f$ cancels out. As in
 * pagmo::igd_plus(), if one of the sums overflows or is subnormal in double precision, both are recomputed after
 * multiplying the differences between the objective values by the same power of two.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    All objectives are assumed to be minimized, consistently with the rest of pagmo. Objectives to be maximized
 *    must be negated in ``points``, ``reference_front`` and ``ref_point`` before calling this function.
 *
 * \endverbatim
 *
 * Complexity is \f$ O(M N_A N_Z)\f$ where \f$M\f$ is the number of objectives, and \f$N_A\f$ and \f$N_Z\f$ are the
 * number of points in \f$A\f$ and \f$Z\f$.
 *
 * @param points the objective vectors \f$A\f$ to be evaluated.
 * @param reference_front the objective vectors of the reference front \f$Z\f$.
 * @param ref_point the reference point \f$\mathbf r\f$ used as normalization bound.
 *
 * @return the NIGD+ of \p points with respect to \p reference_front and \p ref_point.
 *
 * @throws std::invalid_argument if \p ref_point is empty, if \p points or \p reference_front is empty, if the
 * dimension of an objective vector differs from the dimension of \p ref_point, or if an objective vector or
 * \p ref_point contains non-finite values
 * @throws std::invalid_argument if \p ref_point weakly dominates every point of \p reference_front
 */
double normalized_igd_plus(const std::vector<vector_double> &points, const std::vector<vector_double> &reference_front,
                           const vector_double &ref_point)
{
    check_ref_point(ref_point, 1u, "normalized_igd_plus");
    check_point_set(points, "set of points", "normalized_igd_plus", ref_point.size());
    check_point_set(reference_front, "reference front", "normalized_igd_plus", ref_point.size());

    if (std::all_of(reference_front.begin(), reference_front.end(), [&ref_point](const vector_double &z) {
            return std::equal(ref_point.begin(), ref_point.end(), z.begin(), std::less_equal<double>());
        })) {
        pagmo_throw(std::invalid_argument,
                    "The IGD+ of the reference point with respect to the reference front passed to "
                    "normalized_igd_plus() is zero, i.e., the reference point weakly dominates every objective "
                    "vector of the reference front. The reference point must be worse than at least one objective "
                    "vector of the reference front in at least one objective");
    }

    auto points_sum = igd_plus_sum(points, reference_front, 0);
    auto reference_sum = igd_plus_sum({ref_point}, reference_front, 0);
    if (is_out_of_range(points_sum.first) || is_out_of_range(reference_sum.first)) {
        // As in igd_plus(), the differences are scaled down if a sum overflows, and scaled up (exactly) if a sum is
        // subnormal, unless the largest distance is too large for that.
        int exp = 0;
        if (!std::isfinite(points_sum.first) || !std::isfinite(reference_sum.first)) {
            exp = distance_scale_exp(
                std::max({max_abs_value(points), max_abs_value(reference_front), max_abs_value({ref_point})}),
                reference_front.size(), ref_point.size());
        } else {
            exp = std::max(0, distance_scale_exp(std::max(points_sum.second, reference_sum.second),
                                                 reference_front.size(), ref_point.size()));
        }
        if (exp != 0) {
            points_sum = igd_plus_sum(points, reference_front, exp);
            reference_sum = igd_plus_sum({ref_point}, reference_front, exp);
        }
    }
    // Without scaling, or when scaling up, the sum of the reference point is positive, since the difference between
    // two distinct doubles is never zero. When scaling down, it can underflow to zero only if points_sum overflowed and
    // reference_sum did not: the ratio then exceeds the largest finite double, and infinity is returned.
    return points_sum.first / reference_sum.first;
}

} // namespace pagmo
