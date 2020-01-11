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

#include <algorithm>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>

namespace pagmo
{

/// Default constructor.
hv_algorithm::hv_algorithm() = default;

/// Destructor.
hv_algorithm::~hv_algorithm() {}

/// Default copy constructor
hv_algorithm::hv_algorithm(const hv_algorithm &) = default;

/// Default move constructor
hv_algorithm::hv_algorithm(hv_algorithm &&) noexcept = default;

/// Compute volume between two points
/**
 * Calculates the volume between points a and b (as defined for n-dimensional Euclidean spaces).
 *
 * @param a first point defining the hypercube
 * @param b second point defining the hypercube
 * @param dim_bound dimension boundary for the volume. If equal to 0 (default value), then compute the volume of
 * whole vector. Any positive number limits the computation from dimension 1 to dim_bound INCLUSIVE.
 *
 * @return volume of hypercube defined by points a and b
 */
double hv_algorithm::volume_between(const vector_double &a, const vector_double &b, vector_double::size_type dim_bound)
{
    if (dim_bound == 0) {
        dim_bound = a.size();
    }
    double volume = 1.0;
    for (vector_double::size_type idx = 0u; idx < dim_bound; ++idx) {
        volume *= (a[idx] - b[idx]);
    }
    return (volume < 0. ? -volume : volume);
}

/// Compute volume between two points
/**
 * Calculates the volume between points a and b (as defined for n-dimensional Euclidean spaces).
 *
 * @param a first point defining the hypercube
 * @param b second point defining the hypercube
 * @param size dimension of the vectors.
 *
 * @return volume of hypercube defined by points a and b
 */
double hv_algorithm::volume_between(double *a, double *b, vector_double::size_type size)
{
    double volume = 1.0;
    while (size--) {
        volume *= (b[size] - a[size]);
    }
    return (volume < 0 ? -volume : volume);
}

/// Exclusive hypervolume method
/**
 * This method computes the exclusive hypervolume for given individual.
 * It accepts a list of points as an input, and the distinguished "reference point".
 * Hypervolume is then computed as a joint hypervolume of hypercubes, generated pairwise with the reference point
 * and each point from the set.
 *
 * @param p_idx index of the individual
 * @param points vector of vector_doubles for which the hypervolume is computed
 * @param r_point distinguished "reference point".
 *
 * @return exlusive hypervolume contributed by the individual at index p_idx
 */
double hv_algorithm::exclusive(unsigned p_idx, std::vector<vector_double> &points, const vector_double &r_point) const
{
    if (points.size() == 1) {
        return compute(points, r_point);
    }
    std::vector<vector_double> points_less;
    points_less.reserve(points.size() - 1);
    std::copy(points.begin(), points.begin() + p_idx, std::back_inserter(points_less));
    std::copy(points.begin() + p_idx + 1, points.end(), std::back_inserter(points_less));

    return compute(points, r_point) - compute(points_less, r_point);
}

/// Least contributor method
/**
 * This method establishes the individual that contributes the least to the hypervolume.
 * By default it computes each individual contribution, and chooses the one with the lowest contribution.
 * Other algorithms may overload this method for a more efficient solution.
 *
 * @param points vector of vector_doubles for which the hypervolume is computed
 * @param r_point distinguished "reference point".
 *
 * @return index of the least contributor
 */
unsigned long long hv_algorithm::least_contributor(std::vector<vector_double> &points,
                                                   const vector_double &r_point) const
{
    return extreme_contributor(points, r_point, [](double a, double b) { return a < b; });
}

/// Greatest contributor method
/**
 * This method establishes the individual that contributes the most to the hypervolume.
 * By default it computes each individual contribution, and chooses the one with the highest contribution.
 * Other algorithms may overload this method for a more efficient solution.
 *
 * @param points vector of vector_doubles for which the hypervolume is computed
 * @param r_point distinguished "reference point".
 *
 * @return index of the greatest contributor
 */
unsigned long long hv_algorithm::greatest_contributor(std::vector<vector_double> &points,
                                                      const vector_double &r_point) const
{
    return extreme_contributor(points, r_point, [](double a, double b) { return a > b; });
}

/// Contributions method
/**
 * This methods return the exclusive contribution to the hypervolume for each point.
 * Main reason for this method is the fact that in most cases the explicit request for all contributions
 * can be done more efficiently (may vary depending on the provided hv_algorithm) than executing "exclusive" method
 * in a loop.
 *
 * This base method uses a very naive approach, which in fact is only slightly more efficient than calling
 * "exclusive" method successively.
 *
 * @param points vector of vector_doubles for which the contributions are computed
 * @param r_point distinguished "reference point".
 * @return vector of exclusive contributions by every point
 */
std::vector<double> hv_algorithm::contributions(std::vector<vector_double> &points, const vector_double &r_point) const
{
    std::vector<double> c;
    c.reserve(points.size());

    // Trivial case
    if (points.size() == 1) {
        c.push_back(volume_between(points[0], r_point));
        return c;
    }

    // Compute the total hypervolume for the reference
    std::vector<vector_double> points_cpy(points.begin(), points.end());
    double hv_total = compute(points_cpy, r_point);

    // Points[0] as a first candidate
    points_cpy = std::vector<vector_double>(points.begin() + 1, points.end());
    c.push_back(hv_total - compute(points_cpy, r_point));

    // Check the remaining ones using the provided comparison function
    for (unsigned idx = 1u; idx < points.size(); ++idx) {
        std::vector<vector_double> points_less;
        points_less.reserve(points.size() - 1);
        std::copy(points.begin(), points.begin() + idx, std::back_inserter(points_less));
        std::copy(points.begin() + idx + 1, points.end(), std::back_inserter(points_less));
        double delta = hv_total - compute(points_less, r_point);
        c.push_back(delta);
    }

    return c;
}

/// Get algorithm's name.
/**
 * Default implementation will return the algorithm's mangled C++ name.
 *
 * @return name of the algorithm.
 */
std::string hv_algorithm::get_name() const
{
    return typeid(*this).name();
}

/// Assert that reference point dominates every other point from the set.
/**
 * This is a method that can be referenced from verify_before_compute method.
 * The method checks whether the provided reference point fits the minimisation assumption, e.g.,
 * reference point must be "no worse" and in at least one objective and "better" for each of the points from the
 * set.
 *
 * @param points - vector of vector_doubles for which the hypervolume is computed
 * @param r_point - distinguished "reference point".
 */
void hv_algorithm::assert_minimisation(const std::vector<vector_double> &points, const vector_double &r_point) const
{
    for (std::vector<vector_double>::size_type idx = 0; idx < points.size(); ++idx) {
        bool outside_bounds = false;
        bool all_equal = true;

        for (vector_double::size_type f_idx = 0; f_idx < points[idx].size(); ++f_idx) {
            outside_bounds |= (r_point[f_idx] < points[idx][f_idx]);
            all_equal &= (r_point[f_idx] == points[idx][f_idx]);
        }
        if (all_equal || outside_bounds) {
            // Prepare error message.
            std::stringstream ss;
            std::string str_p("("), str_r("(");
            for (vector_double::size_type f_idx = 0; f_idx < points[idx].size(); ++f_idx) {
                str_p += std::to_string(points[idx][f_idx]);
                str_r += std::to_string(r_point[f_idx]);
                if (f_idx < points[idx].size() - 1) {
                    str_p += ", ";
                    str_r += ", ";
                } else {
                    str_p += ")";
                    str_r += ")";
                }
            }
            ss << "Reference point is invalid: another point seems to be outside the reference point boundary, or "
                  "be equal to it:"
               << std::endl;
            ss << " P[" << idx << "]\t= " << str_p << std::endl;
            ss << " R\t= " << str_r << std::endl;
            pagmo_throw(std::invalid_argument, ss.str());
        }
    }
}

/// Dominance comparison method
/**
 * Establishes the domination relationship between two points (overloaded for double*);
 *
 * returns DOM_CMP_B_DOMINATES_A if point 'b' DOMINATES point 'a'
 * returns DOM_CMP_A_DOMINATES_B if point 'a' DOMINATES point 'b'
 * returns DOM_CMP_A_B_EQUAL if point 'a' IS EQUAL TO 'b'
 * returns DOM_CMP_INCOMPARABLE otherwise
 *
 * @param a first point
 * @param b second point
 * @param size size of the points
 *
 * @return the comparison result (1 - b dom a,2 - a dom b, 3 - a == b,4 - not comparable)
 */
int hv_algorithm::dom_cmp(double *a, double *b, vector_double::size_type size)
{
    for (vector_double::size_type i = 0; i < size; ++i) {
        if (a[i] > b[i]) {
            for (vector_double::size_type j = i + 1; j < size; ++j) {
                if (a[j] < b[j]) {
                    return DOM_CMP_INCOMPARABLE;
                }
            }
            return DOM_CMP_B_DOMINATES_A;
        } else if (a[i] < b[i]) {
            for (vector_double::size_type j = i + 1; j < size; ++j) {
                if (a[j] > b[j]) {
                    return DOM_CMP_INCOMPARABLE;
                }
            }
            return DOM_CMP_A_DOMINATES_B;
        }
    }
    return DOM_CMP_A_B_EQUAL;
}

// Dominance comparison method
int hv_algorithm::dom_cmp(const vector_double &a, const vector_double &b, vector_double::size_type dim_bound)
{
    if (dim_bound == 0u) {
        dim_bound = a.size();
    }
    for (vector_double::size_type i = 0u; i < dim_bound; ++i) {
        if (a[i] > b[i]) {
            for (vector_double::size_type j = i + 1; j < dim_bound; ++j) {
                if (a[j] < b[j]) {
                    return DOM_CMP_INCOMPARABLE;
                }
            }
            return DOM_CMP_B_DOMINATES_A;
        } else if (a[i] < b[i]) {
            for (vector_double::size_type j = i + 1; j < dim_bound; ++j) {
                if (a[j] > b[j]) {
                    return DOM_CMP_INCOMPARABLE;
                }
            }
            return DOM_CMP_A_DOMINATES_B;
        }
    }
    return DOM_CMP_A_B_EQUAL;
}

// Compute the extreme contributor
/**
 * Computes the index of the individual that contributes the most or the least to the
 * hypervolume (depending on the  prodivded comparison function)
 */
unsigned hv_algorithm::extreme_contributor(std::vector<vector_double> &points, const vector_double &r_point,
                                           bool (*cmp_func)(double, double)) const
{
    // Trivial case
    if (points.size() == 1u) {
        return 0u;
    }

    std::vector<double> c = contributions(points, r_point);

    unsigned idx_extreme = 0u;

    // Check the remaining ones using the provided comparison function
    for (unsigned idx = 1u; idx < c.size(); ++idx) {
        if (cmp_func(c[idx], c[idx_extreme])) {
            idx_extreme = idx;
        }
    }

    return idx_extreme;
}

} // namespace pagmo
