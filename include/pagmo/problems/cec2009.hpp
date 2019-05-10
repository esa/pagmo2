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

You should have received codetail::pi()es of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#ifndef PAGMO_PROBLEMS_CEC2009_HPP
#define PAGMO_PROBLEMS_CEC2009_HPP

#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Fwd declare for the type definition below.
class PAGMO_DLL_PUBLIC cec2009;

namespace detail
{

namespace cec2009_data
{

// Alias for pointer to cec2009 implementation member function.
typedef void (cec2009::*func_ptr)(vector_double &, const vector_double &) const;

} // namespace cec2009_data

} // namespace detail

/// The CEC 2009 problems: Competition on "Performance Assessment of Constrained / Bound
///  Constrained Multi-Objective Optimization Algorithms"
/**
 *
 * This class instantiates any of the problems from CEC2009's competition
 * on multi-objective optimization algorithms, commonly referred to by the literature
 * as UF1-UF10 (unconstrained) and CF1-CF10 (constrained).
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The three problems constructed by some transformation on DTLZ2, DTLZ3
 *    and WFG1 problems as described in the technical report are not included in
 *    this implementation.
 *
 * .. note::
 *
 *    All problems are continuous, multi objective problems.
 *
 * .. seealso:
 *
 *    http://www3.ntu.edu.sg/home/EPNSugan/index_files/CEC09-MOEA/CEC09-MOEA.htm
 *
 * \endverbatim
 *
 */
class PAGMO_DLL_PUBLIC cec2009
{
public:
    /// Constructor
    /**
     * Will construct one of the 20 multi-objective optimization problems from
     * the CEC2009 competition. There are two sets of problems, namely the set
     * with unconstrained problems (UF) and the set with constrained problems (CF).
     *
     * @param prob_id The problem id. One of [1,2,...10]
     * @param is_constrained Specify whether the problem is constrained. False will yield the UF problems, True will
     * yield the CF problems.
     * @param dim problem dimension. Default is 30, which is the setting used by the competition. But all the
     * problems are scalable in terms of decision variable's dimension.
     *
     * @see http://www3.ntu.edu.sg/home/EPNSugan/index_files/CEC09-MOEA/CEC09-MOEA.htm
     *
     */
    cec2009(unsigned prob_id = 1u, bool is_constrained = false, unsigned dim = 30u);
    // Inequality constraint dimension
    vector_double::size_type get_nic() const;
    // Number of objectives
    vector_double::size_type get_nobj() const;
    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;
    // Fitness computation
    vector_double fitness(const vector_double &) const;
    // Problem name
    std::string get_name() const;
    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Pointers to member functions are used
    PAGMO_DLL_LOCAL vector_double fitness_impl(detail::cec2009_data::func_ptr, const vector_double &) const;

    // For the coverage analysis we do not cover the code below as its derived from a third party source
    // LCOV_EXCL_START

    // -------------------------------------------
    PAGMO_DLL_LOCAL void UF1(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF2(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF3(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF4(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF5(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF6(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF7(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF8(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF9(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void UF10(vector_double &f, const vector_double &x) const;

    /****************************************************************************/
    // constraint test instances
    /****************************************************************************/
    PAGMO_DLL_LOCAL void CF1(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF2(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF3(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF4(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF5(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF6(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF7(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF8(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF9(vector_double &f, const vector_double &x) const;
    PAGMO_DLL_LOCAL void CF10(vector_double &f, const vector_double &x) const;
    // -------------------------------------------
    // LCOV_EXCL_STOP

    static const std::vector<detail::cec2009_data::func_ptr> s_u_ptr;
    static const std::vector<detail::cec2009_data::func_ptr> s_c_ptr;

    // problem id
    unsigned m_prob_id;
    bool m_is_constrained;
    unsigned m_dim;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::cec2009)

#endif
