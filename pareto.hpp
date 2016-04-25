#ifndef PAGMO_PARETO_HPP
#define PAGMO_PARETO_HPP

/** \file pareto.hpp
 * \brief Pareto.
 *
 * This header contains utilities used to compute non-dominated fronts
 */

 #include <cassert>
 #include "../types.hpp"
namespace pagmo{
namespace utils {
/// Pareto-dominance
/**
 * Return true if f1 Pareto dominate f2, false otherwise. This default implementation will assume minimisation for each one of the v_f components
 * I.e., each pair of corresponding elements in f1 and f2 is compared: if all elements in v_f1 are less or equal to the corresponding
 * element in f2, true will be returned. Otherwise, false will be returned.
 *
 *
 * @param[in] f1 first fitness vector.
 * @param[in] f2 second fitness vector.
 *
 * @return true if f1 is dominating f2, false otherwise.
 */
bool pareto_dominates(const vector_double &f1, const vector_double &f2) 
{
	assert(f1.size() == f2.size()); 
	vector_double::size_type count1 = 0u; 
	vector_double::size_type count2 = 0u;
	for (decltype(f1.size()) i = 0u; i < f1.size(); ++i) {
		if (f1[i] < f2[i]) {
			++count1;
		}
		if (f1[i] == f2[i]) {
			++count2;
		}
	}
	return ( ( (count1+count2) == f1.size()) && (count1>0) );
}

}} // namespace pagmo, utils
 #endif