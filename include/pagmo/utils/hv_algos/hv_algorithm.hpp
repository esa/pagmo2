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

#ifndef PAGMO_UTIL_HV_ALGORITHM_H
#define PAGMO_UTIL_HV_ALGORITHM_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Base hypervolume algorithm class.
/**
 * This class represents the abstract hypervolume algorithm used for computing
 * the hypervolume indicator (also known as Lebesgue measure, or S-metric), and other
 * measures that can derive from it, e.g. exclusive contribution by given point
 *
 * There are the following public methods available:
 *
 * - 'compute' - returns the total hypervolume of the set of points
 * - 'exclusive' - returns the exclusive volume contributed by a given point
 * - 'least_contributor' - returns the index of the point contributing the least volume
 * - 'greatest_contributor' - returns the index of the point contributing the most volume
 * - 'contributions' - returns the vector of exclusive contributions for each of the points.
 *
 * Additionally, the private method extreme_contributor can be overloaded:
 * - 'extreme_contributor' - returns an index of a single individual that contributes either the least or the greatest
 *  amount of the volume. The distinction between the extreme contributors is made using a comparator function.
 *  Purpose of this method is to avoid repeating a similar code for the least and the greatest contributor methods.
 *  In many cases it's just a matter of a single alteration in a comparison sign '<' to '>'.
 *
 * This class provides a base for an interface for any hv_algorithm that may be added.
 * The most important method to implement is the 'compute' method, as the remaining methods can be derived from it.
 * If no other method than 'compute' is implemented, the base class will use a naive approach to provide the other
 * functionalities:
 *
 * 'hv_algorithm::exclusive' method relies on 'compute' method, by computing the hypervolume twice (e.g. ExclusiveHV =
 * HV(P) -
 * HV(P/{p}))
 * 'hv_algorithm::contributions' method relies on 'compute' method as well, by computing the exclusive volume for each
 * point
 * using the approach above.
 * 'hv_algorithm::extreme_contributor' (private method) relies on the 'hv_algorithm::contributions' method in order to
 * elicit the correct
 * extreme individual.
 * 'hv_algorithm::least_contributor' and 'hv_algorithm::greatest_contributor' methods rely on
 * 'hv_algorithm::extreme_contributor' method by
 * providing the correct comparator.
 *
 * Thanks to that, any newly implemented hypervolume algorithm that overloads the 'compute' method, gets the
 * functionalities above as well.
 * It is often the case that the algorithm may provide a better solution for each of the features above, e.g.
 * overloading the 'hv_algorithm::extreme_contributor' method with an efficient implementation will automatically speed
 * up the 'least_contributor' and the 'greatest_contributor' methods as well.
 *
 * Additionally, any newly implemented hypervolume algorithm should overload the 'hv_algorithm::verify_before_compute'
 * method in
 * order to prevent
 * the computation in case of incompatible data.
 *
 */
class hv_algorithm
{
public:
    /// Destructor required for pure virtual methods
    hv_algorithm() = default;
    virtual ~hv_algorithm() {}
    /// Default copy constructor
    hv_algorithm(const hv_algorithm &) = default;
    /// Default move constructor
    hv_algorithm(hv_algorithm &&) = default;

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
    static double volume_between(const vector_double &a, const vector_double &b,
                                 vector_double::size_type dim_bound = 0u)
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
    static double volume_between(double *a, double *b, vector_double::size_type size)
    {
        double volume = 1.0;
        while (size--) {
            volume *= (b[size] - a[size]);
        }
        return (volume < 0 ? -volume : volume);
    }

    /// Compute method
    /**
     * This method computes the hypervolume.
     * It accepts a list of points as an input, and the distinguished "reference point".
     * Hypervolume is then computed as a joint hypervolume of hypercubes, generated pairwise with the reference point
     * and each point from the set.
     *
     * @param points - vector of points for which the hypervolume is computed
     * @param r_point - reference point.
     *
     * @return The value of the hypervolume
     */
    virtual double compute(std::vector<vector_double> &points, const vector_double &r_point) const = 0;

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
    virtual double exclusive(unsigned int p_idx, std::vector<vector_double> &points, const vector_double &r_point) const
    {
        if (points.size() == 1) {
            return compute(points, r_point);
        }
        std::vector<vector_double> points_less;
        points_less.reserve(points.size() - 1);
        copy(points.begin(), points.begin() + p_idx, back_inserter(points_less));
        copy(points.begin() + p_idx + 1, points.end(), back_inserter(points_less));

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
    virtual unsigned long long least_contributor(std::vector<vector_double> &points, const vector_double &r_point) const
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
    virtual unsigned long long greatest_contributor(std::vector<vector_double> &points,
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
    virtual std::vector<double> contributions(std::vector<vector_double> &points, const vector_double &r_point) const
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
        for (unsigned int idx = 1u; idx < points.size(); ++idx) {
            std::vector<vector_double> points_less;
            points_less.reserve(points.size() - 1);
            copy(points.begin(), points.begin() + idx, back_inserter(points_less));
            copy(points.begin() + idx + 1, points.end(), back_inserter(points_less));
            double delta = hv_total - compute(points_less, r_point);
            c.push_back(delta);
        }

        return c;
    }

    /// Verification of input
    /**
     * This method serves as a verification method.
     * Not every algorithm is suited of every type of problem.
     *
     * @param points - vector of vector_doubles for which the hypervolume is computed
     * @param r_point - distinguished "reference point".
     */
    virtual void verify_before_compute(const std::vector<vector_double> &points,
                                       const vector_double &r_point) const = 0;

    /// Clone method.
    /**
     * @return ptr to a copy of this.
     */
    virtual std::shared_ptr<hv_algorithm> clone() const = 0;

    /// Get algorithm's name.
    /**
     * Default implementation will return the algorithm's mangled C++ name.
     *
     * @return name of the algorithm.
     */
    virtual std::string get_name() const
    {
        return typeid(*this).name();
    }

protected:
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
    void assert_minimisation(const std::vector<vector_double> &points, const vector_double &r_point) const
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

    /*! Possible result of a comparison between points */
    enum {
        DOM_CMP_B_DOMINATES_A = 1, ///< second argument dominates the first one
        DOM_CMP_A_DOMINATES_B = 2, ///< first argument dominates the second one
        DOM_CMP_A_B_EQUAL = 3,     ///< both points are equal
        DOM_CMP_INCOMPARABLE = 4   ///< points are incomparable
    };

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
    static int dom_cmp(double *a, double *b, vector_double::size_type size)
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

    /// Dominance comparison method
    /**
     * Establishes the domination relationship between two points.
     *
     * returns DOM_CMP_B_DOMINATES_A if point 'b' DOMINATES point 'a'
     * returns DOM_CMP_A_DOMINATES_B if point 'a' DOMINATES point 'b'
     * returns DOM_CMP_A_B_EQUAL if point 'a' IS EQUAL TO 'b'
     * returns DOM_CMP_INCOMPARABLE otherwise
     *
     * @param a first point
     * @param b second point
     * @param dim_bound maximum dimension to be considered
     *
     * @return the comparison result (1 - b dom a,2 - a dom b, 3 - a == b,4 - not comparable)
     */
    static int dom_cmp(const vector_double &a, const vector_double &b, vector_double::size_type dim_bound = 0u)
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

private:
    /// Compute the extreme contributor
    /**
     * Computes the index of the individual that contributes the most or the least to the
     * hypervolume (depending on the  prodivded comparison function)
     */
    unsigned int extreme_contributor(std::vector<vector_double> &points, const vector_double &r_point,
                                     bool (*cmp_func)(double, double)) const
    {
        // Trivial case
        if (points.size() == 1u) {
            return 0u;
        }

        std::vector<double> c = contributions(points, r_point);

        unsigned int idx_extreme = 0u;

        // Check the remaining ones using the provided comparison function
        for (unsigned int idx = 1u; idx < c.size(); ++idx) {
            if (cmp_func(c[idx], c[idx_extreme])) {
                idx_extreme = idx;
            }
        }

        return idx_extreme;
    }
};

} // namespace pagmo

#endif
