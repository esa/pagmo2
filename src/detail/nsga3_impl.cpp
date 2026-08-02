/* Copyright 2017-2021 PaGMO development team

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
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp> // fast_non_dominated_sorting

namespace
{

/*  Relative tolerance used when deciding whether two extreme points coincide.
 */
constexpr double extreme_point_tol = 1e-12;

/*  Two extreme points are duplicates only when *every* coordinate matches. The
 *  comparison is relative to the magnitude of the coordinates, so that it does
 *  not depend on the scale of the objectives, with an absolute floor of tol for
 *  coordinates close to zero.
 */
bool close_vectors(const std::vector<double> &lhs, const std::vector<double> &rhs, double tol)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0u; i < lhs.size(); ++i) {
        const double scale = std::max(1.0, std::max(std::abs(lhs[i]), std::abs(rhs[i])));
        if (!(std::abs(lhs[i] - rhs[i]) <= tol * scale)) {
            return false;
        }
    }
    return true;
}

} // namespace

namespace pagmo
{

namespace detail
{

/// Gaussian Elimination
/**
 * Solves the linear system Ax = b by performing Gaussian elimination with partial
 * pivoting on an augmented matrix A|b.
 *
 * A singular, or numerically singular, system is not an error: it is reported by
 * returning an empty optional so that callers may apply their own fallback. The
 * singularity threshold scales with the magnitude of the entries of A, so that the
 * test is invariant under a uniform rescaling of the system.
 *
 * @param A the square coefficient matrix, taken by value as it is modified in place
 * @param b the right hand side vector
 * @return the solution x, or std::nullopt if A is (numerically) singular
 * @throws std::invalid_argument if A is empty or not square, or if the dimension of
 * b does not match that of A
 */
std::optional<vector_double> gaussian_elimination(std::vector<std::vector<double>> A, const vector_double &b)
{
    // Validate dimensions
    const std::size_t N = A.size();
    if (N == 0u) {
        pagmo_throw(std::invalid_argument, "The coefficient matrix of a linear system cannot be empty");
    }
    for (std::size_t i = 0u; i < N; ++i) {
        if (A[i].size() != N) {
            pagmo_throw(std::invalid_argument, "The coefficient matrix of a linear system must be square, while row "
                                                   + std::to_string(i) + " has " + std::to_string(A[i].size())
                                                   + " entries and the matrix has " + std::to_string(N) + " rows");
        }
    }
    if (b.size() != N) {
        pagmo_throw(std::invalid_argument, "The right hand side of a linear system must match the dimension of the "
                                           "coefficient matrix, while a vector of size "
                                               + std::to_string(b.size()) + " was detected for a matrix of size "
                                               + std::to_string(N));
    }

    /*  Singularity is declared when a pivot cannot be distinguished from zero at the
     *  scale of the matrix entries. The floor keeps an all-zero matrix singular.
     */
    double max_entry = 0.0;
    for (std::size_t i = 0u; i < N; ++i) {
        for (std::size_t j = 0u; j < N; ++j) {
            if (!std::isfinite(A[i][j])) {
                return std::nullopt;
            }
            max_entry = std::max(max_entry, std::abs(A[i][j]));
        }
    }
    const double tol = std::max(std::numeric_limits<double>::epsilon() * static_cast<double>(N) * max_entry,
                                std::numeric_limits<double>::min());

    // Build augmented matrix
    for (std::size_t i = 0u; i < N; ++i) {
        A[i].push_back(b[i]);
    }

    // Eliminate subordinate entries, selecting the largest available pivot in each column
    for (std::size_t p = 0u; p < N; ++p) {
        std::size_t pivot_row = p;
        for (std::size_t i = p + 1u; i < N; ++i) {
            if (std::abs(A[i][p]) > std::abs(A[pivot_row][p])) {
                pivot_row = i;
            }
        }
        if (std::abs(A[pivot_row][p]) <= tol) {
            return std::nullopt;
        }
        std::swap(A[p], A[pivot_row]);
        for (std::size_t i = p + 1u; i < N; ++i) {
            const double quot = A[i][p] / A[p][p];
            for (std::size_t j = p; j < A[p].size(); ++j) {
                A[i][j] -= A[p][j] * quot;
            }
        }
    }

    // Back substitution
    vector_double x(N);
    std::size_t i = N - 1u;
    while (true) {
        for (std::size_t var = i + 1u; var < N; ++var) {
            A[i][N] -= A[i][var] * x[var];
        }
        x[i] = A[i][N] / A[i][i];
        if (!std::isfinite(x[i])) {
            return std::nullopt;
        }
        if (i == 0u) {
            break;
        }
        --i;
    }

    return x;
}

/// Achievement Scalarization Function
/**
 * Computes the weighted Chebyshev scalarization max_i(objs[i]/weights[i]) used by
 * NSGA-III to identify the extreme point of S_t along a given axis direction.
 * Weights below a small floor are replaced by that floor, so that a zero weight does
 * not produce a division by zero.
 *
 * @param objs the (translated) objective vector
 * @param weights the weight vector, of the same dimension as objs
 * @return the largest weighted ratio over all the components
 */
double achievement(const vector_double &objs, const vector_double &weights)
{
    const double default_weight = 1e-5;
    double max_ratio = -std::numeric_limits<double>::max();
    double w = 0.0;

    for (std::size_t i = 0u; i < objs.size(); ++i) {
        w = weights[i] > default_weight ? weights[i] : default_weight;
        max_ratio = std::max(max_ratio, objs[i] / w);
    }

    return max_ratio;
}

/// Distance from objective point to perpendicular intersection with reference point vector
/**
 * Computes the distance from obj_point to its orthogonal projection onto the line
 * spanned by ref_point. This is the association measure of NSGA-III: each individual
 * is associated to the reference point whose line it lies closest to.
 *
 * @param ref_point the reference point defining the direction
 * @param obj_point the (normalized) objective vector
 * @return the perpendicular distance between obj_point and the ref_point direction
 */
double perpendicular_distance(const std::vector<double> &ref_point, const std::vector<double> &obj_point)
{
    double num = 0.0, denom = 0.0, sq_dist = 0.0;
    for (std::size_t i = 0u; i < ref_point.size(); ++i) {
        num += ref_point[i] * obj_point[i];
        denom += ref_point[i] * ref_point[i];
    }
    const double coeff = num / denom;
    for (std::size_t i = 0u; i < ref_point.size(); ++i) {
        const double term = coeff * ref_point[i] - obj_point[i];
        sq_dist += term * term;
    }
    return std::sqrt(sq_dist);
}

/*  The ideal point used to translate the objectives, Algorithm 2 line 2.
 *
 *  The minimum is taken over S_t, given by `selected`, and not over the whole
 *  combined population. When running_ideal is not null this is instead the best
 *  value found for each objective since the start of the run, as described in
 *  Deb & Jain Section IV.C, and it is updated in place so that it is retained
 *  across generations.
 */
std::vector<double> nsga3_compute_ideal(const std::vector<vector_double> &objs, const std::vector<pop_size_t> &selected,
                                        std::vector<double> *running_ideal)
{
    if (selected.empty()) {
        pagmo_throw(std::invalid_argument, "The selected set S_t of NSGA-III cannot be empty");
    }

    const std::size_t nobj = objs[selected[0]].size();
    std::vector<double> p_ideal(nobj, std::numeric_limits<double>::max());
    for (auto idx : selected) {
        for (std::size_t obj = 0u; obj < nobj; ++obj) {
            p_ideal[obj] = std::min(p_ideal[obj], objs[idx][obj]);
        }
    }

    if (running_ideal != nullptr) {
        if (running_ideal->size() == p_ideal.size()) { // i.e. not first gen
            for (std::size_t i = 0u; i < p_ideal.size(); ++i) {
                p_ideal[i] = std::min(p_ideal[i], (*running_ideal)[i]);
            }
        }
        *running_ideal = p_ideal;
    }

    return p_ideal;
}

/*  Algorithm 2 line 3. Every objective vector is translated, not only those of S_t,
 *  so that the result stays indexed by the position an individual has in the
 *  combined population. Only the entries addressed by `selected` are ever read by
 *  the rest of the pipeline.
 */
std::vector<std::vector<double>> nsga3_translate_objectives(const std::vector<vector_double> &objs,
                                                            const std::vector<double> &ideal_point)
{
    const std::size_t NP = objs.size();
    const std::size_t nobj = ideal_point.size();
    std::vector<std::vector<double>> translated_objs(NP, std::vector<double>(nobj));

    for (std::size_t obj = 0u; obj < nobj; ++obj) {
        for (std::size_t i = 0u; i < NP; ++i) {
            translated_objs[i][obj] = objs[i][obj] - ideal_point[obj];
        }
    }

    return translated_objs;
}

/*  Algorithm 2 line 4: for each axis, the member of S_t which minimises the
 *  achievement scalarization along that axis.
 *
 *  The search runs over the whole of S_t, that is over the accepted fronts *and*
 *  the splitting front, and not over the first non-dominated front alone.
 */
std::vector<std::vector<double>> nsga3_find_extreme_points(const std::vector<pop_size_t> &selected,
                                                           const std::vector<std::vector<double>> &translated_objs,
                                                           const std::vector<double> &ideal_point,
                                                           std::vector<std::vector<double>> *retained_extremes)
{
    if (selected.empty()) {
        pagmo_throw(std::invalid_argument, "The selected set S_t of NSGA-III cannot be empty");
    }

    std::vector<std::vector<double>> points;
    const std::size_t nobj = ideal_point.size();
    points.reserve(nobj);

    if (retained_extremes != nullptr && retained_extremes->size() != nobj) {
        retained_extremes->assign(nobj, std::vector<double>{});
    }

    for (std::size_t i = 0u; i < nobj; ++i) {
        std::vector<double> weights(nobj, 1e-6);
        weights[i] = 1.0;
        double min_asf = std::numeric_limits<double>::max();
        std::vector<double> min_obj{};

        /*  Extreme points retained from previous generations are stored in the
         *  original objective coordinates: each must be translated by the
         *  *current* ideal point before it can be compared, on the same footing,
         *  with the candidates of this generation.
         */
        if (retained_extremes != nullptr) {
            for (std::size_t p = 0u; p < retained_extremes->size(); ++p) {
                if ((*retained_extremes)[p].size() != nobj) {
                    continue; // Nothing retained for this objective yet
                }
                std::vector<double> retained(nobj);
                for (std::size_t obj = 0u; obj < nobj; ++obj) {
                    retained[obj] = (*retained_extremes)[p][obj] - ideal_point[obj];
                }
                const double asf = achievement(retained, weights);
                if (asf < min_asf) {
                    min_asf = asf;
                    min_obj = retained;
                }
            }
        }

        for (auto idx : selected) {
            // Calculate ASF value for translated objectives
            const double asf = achievement(translated_objs[idx], weights);
            if (asf < min_asf) {
                min_asf = asf;
                min_obj = translated_objs[idx];
            }
        }
        if (min_obj.empty()) { // Only reachable if every ASF value was NaN
            min_obj = translated_objs[selected[0]];
        }
        points.push_back(min_obj);
        if (retained_extremes != nullptr) {
            // Retain in the original coordinates, so a moving ideal point does not invalidate it
            std::vector<double> original(nobj);
            for (std::size_t obj = 0u; obj < nobj; ++obj) {
                original[obj] = min_obj[obj] + ideal_point[obj];
            }
            (*retained_extremes)[i] = original;
        }
    }

    return points;
}

std::vector<double> nsga3_translated_maxima(const std::vector<std::vector<double>> &translated_objs,
                                            const std::vector<pop_size_t> &selected)
{
    if (selected.empty()) {
        pagmo_throw(std::invalid_argument, "The selected set S_t of NSGA-III cannot be empty");
    }

    const std::size_t nobj = translated_objs[selected[0]].size();
    std::vector<double> maxima(nobj, -std::numeric_limits<double>::max());
    for (auto idx : selected) {
        for (std::size_t obj = 0u; obj < nobj; ++obj) {
            maxima[obj] = std::max(maxima[obj], translated_objs[idx][obj]);
        }
    }

    return maxima;
}

std::vector<double> nsga3_find_intercepts(const std::vector<std::vector<double>> &ext_points,
                                          const std::vector<std::vector<double>> &translated_objs,
                                          const std::vector<pop_size_t> &selected)
{
    /*  Algorithm 2 line 6.
     *
     *  1. Check duplicate extreme points
     *  2. A = translated objectives of extreme points;  b = [1,1,...] to n_objs
     *  3. Solve Ax = b via Gaussian elimination
     *  4. Return reciprocals as intercepts
     *
     *  Duplicate extreme points, a singular system, and a solved coefficient which
     *  is not strictly positive and finite all mean that no usable hyperplane
     *  passes through the extreme points. NSGA-III then falls back on the
     *  componentwise maximum of the translated objectives over S_t. Both the
     *  extreme points and the returned intercepts are expressed in the translated
     *  coordinate system.
     */

    const std::size_t n_obj = ext_points.size();
    std::vector<double> intercepts(n_obj, 1.0);
    bool fallback_to_maxima = false;

    for (std::size_t p = 0u; !fallback_to_maxima && p < n_obj; ++p) {
        if (ext_points[p].size() != n_obj) {
            fallback_to_maxima = true;
            break;
        }
        for (std::size_t q = p + 1u; !fallback_to_maxima && q < n_obj; ++q) {
            // Extreme points coincide only when the *complete* vectors match
            fallback_to_maxima = close_vectors(ext_points[p], ext_points[q], extreme_point_tol);
        }
    }

    if (!fallback_to_maxima) {
        const std::vector<double> b(n_obj, 1.0);

        // Ax = b
        std::optional<vector_double> x = gaussian_elimination(ext_points, b);

        if (x.has_value()) {
            // Express as intercepts, 1/x
            for (std::size_t i = 0u; i < n_obj; ++i) {
                // A zero, negative or non-finite coefficient has no usable reciprocal
                if (!std::isfinite((*x)[i]) || (*x)[i] <= 0.0) {
                    fallback_to_maxima = true;
                    break;
                }
                intercepts[i] = 1.0 / (*x)[i];
            }
        } else {
            fallback_to_maxima = true; // Singular, or numerically singular, system
        }
    }

    if (fallback_to_maxima) {
        const std::vector<double> maxima = nsga3_translated_maxima(translated_objs, selected);
        for (std::size_t i = 0u; i < n_obj && i < maxima.size(); ++i) {
            intercepts[i] = maxima[i];
        }
    }

    /*  A degenerate objective, identical across the whole of S_t, has zero extent.
     *  Its translated coordinate is zero everywhere, so dividing by one keeps it at
     *  zero instead of producing an infinity or a NaN. The same guard catches an
     *  intercept which the solver returned finite but which the reciprocal turned
     *  into something unusable.
     */
    for (std::size_t i = 0u; i < n_obj; ++i) {
        if (!std::isfinite(intercepts[i]) || intercepts[i] <= 0.0) {
            intercepts[i] = 1.0;
        }
    }

    return intercepts;
}

std::vector<std::vector<double>> nsga3_normalize_objectives(const std::vector<std::vector<double>> &translated_objs,
                                                            const std::vector<double> &intercepts)
{
    /*  Algorithm 2 line 7 and Equation 5.
     *  The objectives, and therefore the intercepts, are already translated by the
     *  ideal point, so the subtraction of Equation 5 has already been applied.
     */

    if (translated_objs.empty()) {
        return {};
    }

    const std::size_t nobj = translated_objs[0].size();
    std::vector<std::vector<double>> norm_objs(translated_objs.size(), std::vector<double>(nobj));

    for (std::size_t i = 0u; i < translated_objs.size(); ++i) {
        for (std::size_t idx = 0u; idx < nobj; ++idx) {
            const double intercept_or_eps = std::max(intercepts[idx], std::numeric_limits<double>::epsilon());
            norm_objs[i][idx] = translated_objs[i][idx] / intercept_or_eps;
        }
    }

    return norm_objs;
}

/*  Selects members of a population for survival into the next generation.
 *  arguments:
 *    objs:  The objective vectors of the combined parent and offspring
 *           populations, of size 2*N_pop
 *    N_pop: The target population size to return
 *    directions: The immutable reference direction set of the algorithm
 */
std::vector<pop_size_t> nsga3_selection(const std::vector<vector_double> &objs, pop_size_t N_pop,
                                        const std::vector<reference_point> &directions,
                                        std::vector<double> *running_ideal,
                                        std::vector<std::vector<double>> *retained_extremes, random_engine_type &reng)
{
    if (N_pop == 0u) {
        pagmo_throw(std::invalid_argument, "NSGA-III cannot select an empty population");
    }
    if (N_pop > objs.size()) {
        pagmo_throw(std::invalid_argument, "NSGA-III cannot select " + std::to_string(N_pop)
                                               + " individuals out of a combined population of only "
                                               + std::to_string(objs.size()));
    }
    if (directions.empty()) {
        pagmo_throw(std::invalid_argument, "NSGA-III requires a non-empty set of reference directions");
    }

    fnds_return_type nds = fast_non_dominated_sorting(objs);
    auto fronts = std::move(std::get<0>(nds));

    /*  Algorithm 1 lines 1 to 8: accumulate whole fronts into the selected set S_t
     *  until it reaches the target size. The front which takes it there is the
     *  splitting front F_l, and is the last one retained.
     */
    std::size_t last_front = 0u;
    std::size_t next_size = 0u;
    while (next_size < N_pop) {
        next_size += fronts[last_front++].size();
    }
    fronts.erase(fronts.begin() + static_cast<decltype(fronts)::difference_type>(last_front), fronts.end());
    std::vector<pop_size_t> selected; // S_t
    selected.reserve(next_size);
    for (const auto &front : fronts) {
        selected.insert(selected.end(), front.begin(), front.end());
    }

    // Algorithm 1 lines 9 to 12: accept all members of the first l-1 fronts
    std::vector<pop_size_t> next;
    next.reserve(N_pop);
    for (std::size_t f = 0u; f + 1u < fronts.size(); ++f) {
        next.insert(next.end(), fronts[f].begin(), fronts[f].end());
    }

    if (next.size() == N_pop) {
        // |S_t| == N: the splitting front is absorbed whole and no niching is needed
        return next;
    }

    /*  Algorithm 1 line 14, Algorithm 2. A null memory pointer means the
     *  corresponding quantity is recomputed from scratch every generation instead
     *  of being retained across them.
     */
    const auto ideal_point = nsga3_compute_ideal(objs, selected, running_ideal);
    const auto translated_objectives = nsga3_translate_objectives(objs, ideal_point);
    const auto ext_points = nsga3_find_extreme_points(selected, translated_objectives, ideal_point, retained_extremes);
    const auto intercepts = nsga3_find_intercepts(ext_points, translated_objectives, selected);
    const auto norm_objs = nsga3_normalize_objectives(translated_objectives, intercepts);

    // Algorithm 1 lines 15 and 16, Algorithm 3
    std::vector<reference_point> rps(directions);
    for (auto &rp : rps) {
        rp.reset();
    }
    associate_with_reference_points(rps, norm_objs, fronts);

    // Algorithm 1 line 17, Algorithm 4
    while (next.size() < N_pop) {
        const std::size_t min_rp_idx = identify_niche_point(rps, reng);
        const std::optional<std::size_t> selected_idx = rps[min_rp_idx].select_member(reng);
        if (selected_idx.has_value()) {
            rps[min_rp_idx].increment_members();
            rps[min_rp_idx].remove_candidate(selected_idx.value());
            next.push_back(selected_idx.value());
        } else {
            rps.erase(rps.begin() + static_cast<decltype(rps)::difference_type>(min_rp_idx));
        }
    }

    return next;
}

} // namespace detail

} // namespace pagmo
