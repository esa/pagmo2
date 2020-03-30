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

#ifndef PAGMO_UTILS_HV_ALGORITHM_HPP
#define PAGMO_UTILS_HV_ALGORITHM_HPP

#include <memory>
#include <string>
#include <vector>

#include <pagmo/detail/visibility.hpp>
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
class PAGMO_DLL_PUBLIC hv_algorithm
{
public:
    // Default ctor.
    hv_algorithm();

    // Destructor.
    virtual ~hv_algorithm();

    // Copy constructor
    hv_algorithm(const hv_algorithm &);

    // Move constructor
    hv_algorithm(hv_algorithm &&) noexcept;

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
                                 vector_double::size_type dim_bound = 0u);

    // Compute volume between two points
    static double volume_between(double *, double *, vector_double::size_type);

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

    // Exclusive hypervolume method
    virtual double exclusive(unsigned p_idx, std::vector<vector_double> &, const vector_double &) const;

    // Least contributor method
    virtual unsigned long long least_contributor(std::vector<vector_double> &, const vector_double &) const;

    // Greatest contributor method
    virtual unsigned long long greatest_contributor(std::vector<vector_double> &, const vector_double &) const;

    // Contributions method
    virtual std::vector<double> contributions(std::vector<vector_double> &, const vector_double &) const;

#if !defined(PAGMO_DOXYGEN_INVOKED)
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
#endif

    /// Clone method.
    /**
     * @return ptr to a copy of this.
     */
    virtual std::shared_ptr<hv_algorithm> clone() const = 0;

    // Get algorithm's name.
    virtual std::string get_name() const;

protected:
    // Assert that reference point dominates every other point from the set.
    void assert_minimisation(const std::vector<vector_double> &, const vector_double &) const;

    /*! Possible result of a comparison between points */
    enum {
        DOM_CMP_B_DOMINATES_A = 1, ///< second argument dominates the first one
        DOM_CMP_A_DOMINATES_B = 2, ///< first argument dominates the second one
        DOM_CMP_A_B_EQUAL = 3,     ///< both points are equal
        DOM_CMP_INCOMPARABLE = 4   ///< points are incomparable
    };

    // Dominance comparison method
    static int dom_cmp(double *, double *, vector_double::size_type);

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
    static int dom_cmp(const vector_double &a, const vector_double &b, vector_double::size_type dim_bound = 0u);

private:
    // Compute the extreme contributor
    PAGMO_DLL_LOCAL unsigned extreme_contributor(std::vector<vector_double> &, const vector_double &,
                                                 bool (*)(double, double)) const;
};

} // namespace pagmo

#endif
