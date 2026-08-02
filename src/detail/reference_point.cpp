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
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/detail/nsga3_impl.hpp> // perpendicular_distance, choose_random_element
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/exceptions.hpp>

namespace
{

/*  Relative tolerance used when deciding whether two reference directions, or two
 *  reference direction coordinates, coincide.
 */
constexpr double direction_tol = 1e-12;

bool close_coordinates(double lhs, double rhs)
{
    const double scale = std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
    return std::abs(lhs - rhs) <= direction_tol * scale;
}

bool close_directions(const std::vector<double> &lhs, const std::vector<double> &rhs)
{
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0u; i < lhs.size(); ++i) {
        if (!close_coordinates(lhs[i], rhs[i])) {
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

reference_point::reference_point(std::size_t nobjs) : coeffs(nobjs, 0.0) {}

double &reference_point::operator[](std::size_t idx)
{
    return coeffs[idx];
}

double reference_point::operator[](std::size_t idx) const
{
    return coeffs[idx];
}

std::size_t reference_point::dim() const
{
    return coeffs.size();
}

std::ostream &operator<<(std::ostream &ostr, const reference_point &rp)
{
    std::ostringstream oss;
    oss << "[";
    if (!rp.coeffs.empty()) {
        std::copy(rp.coeffs.begin(), rp.coeffs.end() - 1, std::ostream_iterator<double>(oss, ", "));
        oss << rp.coeffs.back();
    }
    oss << "]";
    ostr << oss.str();
    return ostr;
}

/*  Exact binomial coefficient.
 *
 *  The multiplicative form is evaluated left to right, so that the running value is
 *  always the exact binomial coefficient C(n - k + i, i) and never a partial
 *  product which would have to be divided later. The common factor between the
 *  numerator and the denominator of each step is cancelled first, which keeps the
 *  intermediate values no larger than the final result: the overflow check
 *  therefore fires only for arguments whose true value does not fit.
 */
std::size_t n_choose_k(std::size_t n, std::size_t k)
{
    if (k > n) {
        return 0u;
    }
    // C(n, k) == C(n, n - k); the smaller of the two needs fewer steps
    if (k > n - k) {
        k = n - k;
    }
    std::size_t result = 1u;
    for (std::size_t i = 1u; i <= k; ++i) {
        std::size_t factor = n - k + i;
        const std::size_t common = std::gcd(factor, i);
        factor /= common;
        const std::size_t divisor = i / common;
        /*  result*(n - k + i) is divisible by i, and the divisor left after
         *  cancelling is coprime with the factor, so it must divide result.
         */
        result /= divisor;
        if (result > std::numeric_limits<std::size_t>::max() / factor) {
            pagmo_throw(std::overflow_error, "The binomial coefficient of " + std::to_string(n) + " and "
                                                 + std::to_string(k) + " is too large to be represented");
        }
        result *= factor;
    }
    return result;
}

std::size_t reference_point_count(std::size_t nobjs, std::size_t divisions)
{
    if (nobjs == 0u) {
        pagmo_throw(std::invalid_argument, "The number of objectives of a reference direction set must be positive");
    }
    if (divisions == 0u) {
        pagmo_throw(std::invalid_argument,
                    "The number of divisions of a reference direction layer must be positive, while a value of 0 "
                    "was detected");
    }
    return n_choose_k(nobjs + divisions - 1u, divisions);
}

std::vector<reference_point> generate_uniform_reference_points(std::size_t nobjs, std::size_t divisions)
{
    const std::size_t count = reference_point_count(nobjs, divisions);
    if (count > max_reference_directions) {
        pagmo_throw(std::invalid_argument,
                    "A Das and Dennis layer with " + std::to_string(divisions) + " divisions over "
                        + std::to_string(nobjs) + " objectives would contain " + std::to_string(count)
                        + " reference directions, which exceeds the limit of "
                        + std::to_string(max_reference_directions)
                        + ". Consider fewer divisions, or the two layer construction of Deb and Jain");
    }

    /*  Das and Dennis: enumerate every way of writing the number of divisions as an
     *  ordered sum of nobjs non-negative integers, and divide by the number of
     *  divisions to land on the unit simplex.
     *
     *  The counts are visited in lexicographic order of their first nobjs - 1
     *  entries, the last entry being whatever remains. This reproduces exactly the
     *  order of the recursive formulation it replaces, without building and
     *  concatenating a vector of points at every level of the recursion.
     */
    std::vector<reference_point> points;
    points.reserve(count);

    const double total = static_cast<double>(divisions);
    std::vector<std::size_t> counts(nobjs, 0u);
    counts.back() = divisions;

    while (true) {
        reference_point rp(nobjs);
        for (std::size_t i = 0u; i < nobjs; ++i) {
            rp[i] = static_cast<double>(counts[i]) / total;
        }
        points.push_back(std::move(rp));

        if (nobjs == 1u) {
            break;
        }
        /*  Advance to the next composition: increment the rightmost of the leading
         *  nobjs - 1 entries which still has slack, and return everything to its
         *  right to the pool.
         */
        std::size_t prefix = 0u;
        std::size_t bumped = nobjs - 1u; // nobjs - 1u means "no entry could be bumped"
        for (std::size_t i = 0u; i + 1u < nobjs; ++i) {
            prefix += counts[i];
        }
        for (std::size_t i = nobjs - 1u; i > 0u; --i) {
            const std::size_t j = i - 1u;
            if (prefix < divisions) {
                bumped = j;
                break;
            }
            prefix -= counts[j];
        }
        if (bumped == nobjs - 1u) {
            break;
        }
        ++counts[bumped];
        std::size_t head = 0u;
        for (std::size_t i = 0u; i <= bumped; ++i) {
            head += counts[i];
        }
        for (std::size_t i = bumped + 1u; i + 1u < nobjs; ++i) {
            counts[i] = 0u;
        }
        counts.back() = divisions - head;
    }

    return points;
}

std::vector<reference_point> generate_reference_directions(std::size_t nobjs, std::size_t divisions_outer,
                                                           std::size_t divisions_inner)
{
    auto directions = generate_uniform_reference_points(nobjs, divisions_outer);
    if (divisions_inner == 0u) {
        return directions;
    }

    const std::size_t outer_count = directions.size();
    const std::size_t inner_count = reference_point_count(nobjs, divisions_inner);
    if (inner_count > max_reference_directions || outer_count > max_reference_directions - inner_count) {
        pagmo_throw(std::invalid_argument,
                    "A two layer reference direction set with " + std::to_string(divisions_outer) + " outer and "
                        + std::to_string(divisions_inner) + " inner divisions over " + std::to_string(nobjs)
                        + " objectives would contain " + std::to_string(outer_count) + " + "
                        + std::to_string(inner_count) + " reference directions, which exceeds the limit of "
                        + std::to_string(max_reference_directions));
    }

    auto inner = generate_uniform_reference_points(nobjs, divisions_inner);

    /*  Shrink the inner layer by one half about the centroid of the simplex. The
     *  coordinates of a Das and Dennis layer sum to one over nobjs entries, so the
     *  image sums to (1 + 1)/2 == 1 and stays on the same simplex.
     */
    const double centre = 1.0 / static_cast<double>(nobjs);
    for (auto &rp : inner) {
        for (std::size_t i = 0u; i < nobjs; ++i) {
            rp[i] = (rp[i] + centre) / 2.0;
        }
    }

    /*  An inner direction can only coincide with an outer one if every one of its
     *  coordinates is also a coordinate of the outer grid. Deciding that from the
     *  two grids alone costs O(divisions_outer * divisions_inner) and, for every
     *  configuration in which the grids do not meet, removes the need to compare
     *  the layers point by point at all.
     */
    bool grids_meet = false;
    for (std::size_t i = 0u; !grids_meet && i <= divisions_inner; ++i) {
        const double inner_coord = (static_cast<double>(i) / static_cast<double>(divisions_inner) + centre) / 2.0;
        for (std::size_t j = 0u; j <= divisions_outer; ++j) {
            const double outer_coord = static_cast<double>(j) / static_cast<double>(divisions_outer);
            if (close_coordinates(inner_coord, outer_coord)) {
                grids_meet = true;
                break;
            }
        }
    }

    directions.reserve(outer_count + inner.size());
    for (auto &rp : inner) {
        if (grids_meet) {
            bool duplicate = false;
            for (std::size_t i = 0u; i < outer_count; ++i) {
                if (close_directions(directions[i].get_coeffs(), rp.get_coeffs())) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }
        }
        directions.push_back(std::move(rp));
    }

    return directions;
}

void reference_point::add_candidate(std::size_t index, double distance)
{
    candidates.emplace_back(index, distance);
}

void reference_point::remove_candidate(std::size_t index)
{
    for (std::size_t idx = 0u; idx < candidates.size(); ++idx) {
        if (candidates[idx].first == index) {
            candidates.erase(candidates.begin()
                             + static_cast<std::vector<std::pair<std::size_t, double>>::difference_type>(idx));
            break; // Candidate indices are unique
        }
    }
}

void associate_with_reference_points(std::vector<reference_point> &rps,
                                     const std::vector<std::vector<double>> &norm_objs,
                                     const std::vector<std::vector<pop_size_t>> &fronts)
{
    for (std::size_t f = 0u; f < fronts.size(); ++f) {
        for (std::size_t i = 0u; i < fronts[f].size(); ++i) {
            std::size_t nearest = 0u;
            double min_dist = std::numeric_limits<double>::max();
            for (std::size_t p = 0u; p < rps.size(); ++p) {
                const double dist = perpendicular_distance(rps[p].get_coeffs(), norm_objs[fronts[f][i]]);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = p;
                }
            }
            /*  Deb & Jain Algorithm 1 line 16: the niche count of a reference point
             *  is the number of members of S_t \ F_l associated with it. Members of
             *  the splitting front F_l are candidates for selection instead.
             */
            if (f != fronts.size() - 1u) {
                rps[nearest].increment_members();
            } else {
                rps[nearest].add_candidate(fronts[f][i], min_dist);
            }
        }
    }
}

std::size_t identify_niche_point(std::vector<reference_point> &rps, random_engine_type &reng)
{
    std::size_t min_size = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> minimal_set;
    for (const auto &rp : rps) {
        min_size = std::min(min_size, rp.member_count());
    }
    for (std::size_t idx = 0u; idx < rps.size(); ++idx) {
        if (rps[idx].member_count() == min_size) {
            minimal_set.push_back(idx);
        }
    }
    // Return a random element from the minimal set
    return choose_random_element<std::size_t>(minimal_set, reng);
}

// Section IV.E
std::optional<std::size_t> reference_point::select_member(random_engine_type &reng) const
{
    std::optional<std::size_t> selected = std::nullopt;
    if (candidate_count() != 0u) {
        if (member_count() == 0u) { // Candidates but no members: rho == 0
            selected = nearest_candidate();
        } else {
            selected = random_candidate(reng); // Candidates and members: rho >= 1
        }
    }
    return selected;
}

std::optional<std::size_t> reference_point::nearest_candidate() const
{
    double min_dist = std::numeric_limits<double>::max();
    std::optional<std::size_t> min_idx = std::nullopt;
    for (std::size_t idx = 0u; idx < candidates.size(); ++idx) {
        if (candidates[idx].second < min_dist) {
            min_dist = candidates[idx].second;
            min_idx = candidates[idx].first;
        }
    }
    return min_idx;
}

std::optional<std::size_t> reference_point::random_candidate(random_engine_type &reng) const
{
    if (candidates.empty()) {
        return std::nullopt;
    }
    return choose_random_element<std::pair<std::size_t, double>>(candidates, reng).first;
}

} // namespace detail

} // namespace pagmo
