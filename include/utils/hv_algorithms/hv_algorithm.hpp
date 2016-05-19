/*****************************************************************************
*   Copyright (C) 2004-2015 The PaGMO development team,                     *
*   Advanced Concepts Team (ACT), European Space Agency (ESA)               *
*                                                                           *
*   https://github.com/esa/pagmo                                            *
*                                                                           *
*   act@esa.int                                                             *
*                                                                           *
*   This program is free software; you can redistribute it and/or modify    *
*   it under the terms of the GNU General Public License as published by    *
*   the Free Software Foundation; either version 2 of the License, or       *
*   (at your option) any later version.                                     *
*                                                                           *
*   This program is distributed in the hope that it will be useful,         *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
*   GNU General Public License for more details.                            *
*                                                                           *
*   You should have received a copy of the GNU General Public License       *
*   along with this program; if not, write to the                           *
*   Free Software Foundation, Inc.,                                         *
*   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.               *
*****************************************************************************/

#ifndef PAGMO_UTIL_HV_ALGORITHMS_HV_ALGORITHM_H
#define PAGMO_UTIL_HV_ALGORITHMS_HV_ALGORITHM_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <stdexcept>
#include <memory>
#include <string>

#include "../../io.hpp"
#include "../../types.hpp"

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wshorten-64-to-32"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

namespace pagmo {

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
	* Additionally, a private method 'base::extreme_contributor' can be overloaded:
	* - 'extreme_contributor' - returns an index of a single individual that contributes either the least or the greatest
	*  amount of the volume. The distinction between the extreme contributors is made using a comparator function.
	*  Purpose of this method is to avoid repeating a similar code for the least and the greatest contributor methods.
	*  In many cases it's just a matter of a single alteration in a comparison sign '<' to '>'.
	*
	* This base class provides an interface for any hv_algorithm that may be added.
	* The most important method to implement is the 'compute' method, as the remaining methods can be derived from it.
	* If no other method than 'compute' is implemented, the base class will use a naive approach to provide the other functionalities:
	*
	* 'base::exclusive' method relies on 'compute' method, by computing the hypervolume twice (e.g. ExclusiveHV = HV(P) - HV(P/{p}))
	* 'base::contributions' method relies on 'compute' method as well, by computing the exclusive volume for each point using the approach above.
	* 'base::extreme_contributor' (private method) relies on the 'base::contributions' method in order to elicit the correct extreme individual.
	* 'base::least_contributor' and 'base::greatest_contributor' methods rely on 'base::extreme_contributor' method by providing the correct comparator.
	*
	* Thanks to that, any newly implemented hypervolume algorithm that overloads the 'compute' method, gets the functionalities above as well.
	* It is often the case that the algorithm may provide a better solution for each of the features above, e.g. overloading the 'base::extreme_contributor'
	* method with an efficient implementation will automatically speed up the 'least_contributor' and the 'greatest_contributor' methods as well.
	*
	* Additionally, any newly implemented hypervolume algorithm should overload the 'base::verify_before_compute' method in order to prevent
	* the computation in case of incompatible data.
	*
	* @author Krzysztof Nowak (kn@linux.net)
	* @author Marcus Maertens (mmarcusx@gmail.com)
	*/
	class hv_algorithm
	{
	public:
		/// Destructor required for pure virtual methods
		hv_algorithm() = default;
		virtual ~hv_algorithm() {}
		hv_algorithm(const hv_algorithm &) = default;
		hv_algorithm(hv_algorithm &&) = default;


		/// Compute volume between two points
		/**
		* Calculates the volume between points a and b (as defined for n-dimensional Euclidean spaces).
		*
		* @param[in] a first point defining the hypercube
		* @param[in] b second point defining the hypercube
		* @param[in] dim_bound dimension boundary for the volume. If equal to 0 (default value), then compute the volume of whole vector. Any positive number limits the computation from dimension 1 to dim_bound INCLUSIVE.
		*
		* @return volume of hypercube defined by points a and b
		*/
		static double volume_between(const vector_double &a, const vector_double &b, unsigned int dim_bound = 0)
		{
			if (dim_bound == 0) {
				dim_bound = a.size();
			}
			double volume = 1.0;
			for (vector_double::size_type idx = 0; idx < dim_bound; ++idx) {
				volume *= (a[idx] - b[idx]);
			}
			return (volume < 0 ? -volume : volume);
		}


		/// Compute volume between two points
		/**
		* Calculates the volume between points a and b (as defined for n-dimensional Euclidean spaces).
		*
		* @param[in] a first point defining the hypercube
		* @param[in] b second point defining the hypercube
		* @param[in] size dimension of the vectors.
		*
		* @return volume of hypercube defined by points a and b
		*/
		static double volume_between(double* a, double* b, unsigned int size)
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
		* Hypervolume is then computed as a joint hypervolume of hypercubes, generated pairwise with the reference point and each point from the set.
		*
		* @param[in] points - vector of points for which the hypervolume is computed
		* @param[in] r_point - reference point.
		*/
		virtual double compute(std::vector<vector_double> &points, const vector_double &r_point) const = 0;


		/// Exclusive hypervolume method
		/**
		* This method computes the exclusive hypervolume for given individual.
		* It accepts a list of points as an input, and the distinguished "reference point".
		* Hypervolume is then computed as a joint hypervolume of hypercubes, generated pairwise with the reference point and each point from the set.
		*
		* @param[in] p_idx index of the individual
		* @param[in] points vector of vector_doubles for which the hypervolume is computed
		* @param[in] r_point distinguished "reference point".
		*
		* @return exlusive hypervolume contributed by the individual at index p_idx
		*/
		virtual double exclusive(const unsigned int p_idx, std::vector<vector_double> &points, const vector_double &r_point) const
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
		* @param[in] points vector of vector_doubles for which the hypervolume is computed
		* @param[in] r_point distinguished "reference point".
		*
		* @return index of the least contributor
		*/
		virtual unsigned int least_contributor(std::vector<vector_double> &points, const vector_double &r_point) const
		{
			return extreme_contributor(points, r_point, cmp_least);
		}


		/// Greatest contributor method
		/**
		* This method establishes the individual that contributes the most to the hypervolume.
		* By default it computes each individual contribution, and chooses the one with the highest contribution.
		* Other algorithms may overload this method for a more efficient solution.
		*
		* @param[in] points vector of vector_doubles for which the hypervolume is computed
		* @param[in] r_point distinguished "reference point".
		*
		* @return index of the greatest contributor
		*/
		virtual unsigned int greatest_contributor(std::vector<vector_double> &points, const vector_double &r_point) const
		{
			return extreme_contributor(points, r_point, cmp_greatest);
		}


		/// Contributions method
		/**
		* This methods return the exclusive contribution to the hypervolume for each point.
		* Main reason for this method is the fact that in most cases the explicit request for all contributions
		* can be done more efficiently (may vary depending on the provided hv_algorithm) than executing "exclusive" method in a loop.
		*
		* This base method uses a very naive approach, which in fact is only slightly more efficient than calling "exclusive" method successively.
		*
		* @param[in] points vector of vector_doubles for which the contributions are computed
		* @param[in] r_point distinguished "reference point".
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
			for (unsigned int idx = 1; idx < points.size(); ++idx) {
				std::vector<vector_double> points_less;
				points_less.reserve(points.size() - 1);
				copy(points.begin(), points.begin() + idx, back_inserter(points_less));
				copy(points.begin() + idx + 1, points.end(), back_inserter(points_less));
				double delta = hv_total - compute(points_less, r_point);

				if (fabs(delta) < 1e-8) {
					delta = 0.0;
				}
				c.push_back(delta);
			}

			return c;
		}


		/// Verification of input
		/**
		* This method serves as a verification method.
		* Not every algorithm is suited of every type of problem.
		*
		* @param[in] points - vector of vector_doubles for which the hypervolume is computed
		* @param[in] r_point - distinguished "reference point".
		*/
		virtual void verify_before_compute(const std::vector<vector_double> &points, const vector_double &r_point) const = 0;


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
		* reference point must be "no worse" and in at least one objective and "better" for each of the points from the set.
		*
		* @param[in] points - vector of vector_doubles for which the hypervolume is computed
		* @param[in] r_point - distinguished "reference point".
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
						}
						else {
							str_p += ")";
							str_r += ")";
						}
					}
					ss << "Reference point is invalid: another point seems to be outside the reference point boundary, or be equal to it:" << std::endl;
					ss << " P[" << idx << "]\t= " << str_p << std::endl;
					ss << " R\t= " << str_r << std::endl;
					pagmo_throw(std::invalid_argument, ss.str());
				}
			}
		}


		/// compute the extreme contributor
		/**
		* Computes the index of the individual that contributes the most or the least to the hypervolume (depending on the prodivded comparison function)
		*/
		virtual unsigned int extreme_contributor(std::vector<vector_double> &points, const vector_double &r_point, bool(*cmp_func)(double, double)) const
		{
			// Trivial case
			if (points.size() == 1) {
				return 0;
			}

			std::vector<double> c = contributions(points, r_point);

			unsigned int idx_extreme = 0;

			// Check the remaining ones using the provided comparison function
			for (unsigned int idx = 1; idx < c.size(); ++idx) {
				if (cmp_func(c[idx], c[idx_extreme])) {
					idx_extreme = idx;
				}
			}

			return idx_extreme;
		}


		/// Comparison method for the least contributor
		/**
		* This method is used as a comparison function for the extreme contributor method which may be overloaded by hv algorithms.
		* In such case, this method can determine, given two contributions of points, which one is the "smaller" contributor.
		*
		* @param[in] a first contribution of a point
		* @param[in] b second contribution of a point
		*
		* @return true if contribution 'a' is lesser than contribution 'b'
		*/
		static bool cmp_least(const double a, const double b)
		{
			return a < b;
		}


		/**
		* This method is used as a comparison function for the extreme contributor method which may be overloaded by hv algorithms.
		* In such case, this method can determine, given two contributions of points, which one is the "greater" contributor.
		*
		* @param[in] a first contribution of a point
		* @param[in] b second contribution of a point
		*
		* @return true if contribution 'a' is greater than contribution 'b'
		*/
		static bool cmp_greatest(const double a, const double b)
		{
			return a > b;
		}


		// Domination results of the 'dom_cmp' methods
		enum {
			DOM_CMP_B_DOMINATES_A = 1, // second argument dominates the first one
			DOM_CMP_A_DOMINATES_B = 2, // first argument dominates the second one
			DOM_CMP_A_B_EQUAL = 3, // both points are equal
			DOM_CMP_INCOMPARABLE = 4 // points are incomparable
		};


		/// Dominance comparison method
		/**
		* Establishes the domination relationship between two points (overloaded for double*);
		*
		* returns DOM_CMP_B_DOMINATES_A if point 'b' DOMINATES point 'a'
		* returns DOM_CMP_A_DOMINATES_B if point 'a' DOMINATES point 'b'
		* returns DOM_CMP_A_B_EQUAL if point 'a' IS EQUAL TO 'b'
		* returns DOM_CMP_INCOMPARABLE otherwise
		*/
		static int dom_cmp(double* a, double* b, unsigned int size)
		{
			for (vector_double::size_type i = 0; i < size; ++i) {
				if (a[i] > b[i]) {
					for (vector_double::size_type j = i + 1; j < size; ++j) {
						if (a[j] < b[j]) {
							return DOM_CMP_INCOMPARABLE;
						}
					}
					return DOM_CMP_B_DOMINATES_A;
				}
				else if (a[i] < b[i]) {
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
		*/
		static int dom_cmp(const vector_double &a, const vector_double &b, unsigned int dim_bound)
		{
			if (dim_bound == 0) {
				dim_bound = a.size();
			}
			for (vector_double::size_type i = 0; i < dim_bound; ++i) {
				if (a[i] > b[i]) {
					for (vector_double::size_type j = i + 1; j < dim_bound; ++j) {
						if (a[j] < b[j]) {
							return DOM_CMP_INCOMPARABLE;
						}
					}
					return DOM_CMP_B_DOMINATES_A;
				}
				else if (a[i] < b[i]) {
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


	};


	/// Vector comparator class
	/**
	* This is a helper class that allows for the generation of comparator objects.
	* Many hypervolume algorithms use comparator functions for sorting, or data structures handling.
	* In most cases the difference between the comparator functions differ either by the dimension number, or the inequality sign ('>' or '<').
	* We provide a general comparator class for that purpose.
	*/
	class vector_double_cmp
	{
	public:
		///Constructor of the comparator object
		/**
		* Create a comparator object, that compares items by given dimension, according to given inequality function.
		*
		* @param[in] dim dimension index by which we compare the vectors
		* @param[in] cmp_type inequality expression used for comparison, either character '<' or '>'
		*/
		vector_double_cmp(int dim, char cmp_type)
		{
			if (cmp_type == '<') {
				m_cmp_obj = std::shared_ptr<cmp_fun>(new cmp_le(dim));
			}
			else {
				m_cmp_obj = std::shared_ptr<cmp_fun>(new cmp_ge(dim));
			}
		}


		///Overloaded operator()
		/**
		* Overloading call operator is required for all sorting and data structure key comparators in stl library.
		*
		* @param[in] lhs vector_double on the left hand side
		* @param[in] rhs vector_double on the right hand side
		*
		* @return Boolean variable stating whether given expression is true for vector_doubles.
		*/
		inline bool operator()(const vector_double &lhs, const vector_double &rhs)
		{
			return (*m_cmp_obj)(lhs, rhs);
		}
	private:
		struct cmp_fun
		{
			int m_dim;
			cmp_fun(int dim) : m_dim(dim) { }
			virtual ~cmp_fun() { };
			/// virtual operator() - It is never called anyway, so we could have gone with pure virtual, yet then we would not be able to use inline.
			virtual inline bool operator()(const vector_double &lhs, const vector_double &rhs)
			{
				return lhs[0] < rhs[0];
			}
		};

		struct cmp_le : cmp_fun
		{
			cmp_le(int dim) : cmp_fun(dim) { }
			inline bool operator()(const vector_double &lhs, const vector_double &rhs)
			{
				return lhs[m_dim] < rhs[m_dim];
			}
		};

		struct cmp_ge : cmp_fun
		{
			cmp_ge(int dim) : cmp_fun(dim) { }
			inline bool operator()(const vector_double &lhs, const vector_double &rhs)
			{
				return lhs[m_dim] > rhs[m_dim];
			}
		};

		std::shared_ptr<cmp_fun> m_cmp_obj;
	};

} // namespace pagmo

#if defined(__clang__) || defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#endif
