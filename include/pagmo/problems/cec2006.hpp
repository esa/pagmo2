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

#ifndef PAGMO_PROBLEMS_CEC2006_HPP
#define PAGMO_PROBLEMS_CEC2006_HPP

#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// Fwd declare for the type definition below.
class PAGMO_DLL_PUBLIC cec2006;

namespace detail
{

namespace cec2006_data
{

// Alias for pointer to cec2006 implementation member function.
typedef void (cec2006::*func_ptr)(vector_double &, const vector_double &) const;

} // namespace cec2006_data

} // end namespace detail

/// The CEC 2006 problems: Constrained Real-Parameter Optimization
/**
 *
 * This class allows to instantiate any of the 24 problems of the competition
 * on constrained real-parameter optimization problems that was organized in the
 * framework of the 2006 IEEE Congress on Evolutionary Computation.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The code for these UDAs is adapted from the original C code distributed during the competition and linked below.
 *
 * .. note::
 *
 *    All problems are constrained, continuous, single objective problems.
 *
 * .. seealso:
 *
 *    http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC-06/CEC06.htm
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC cec2006
{
public:
    /// Constructor
    /**
     * Will construct one of the 24 CEC2006 problems
     *
     * @param prob_id The problem id. One of [1,2,...,24]
     *
     * @throws invalid_argument if \p prob_id is not in [1,24]
     */
    cec2006(unsigned prob_id = 1u);
    // Equality constraint dimension
    vector_double::size_type get_nec() const;
    // Inequality constraint dimension
    vector_double::size_type get_nic() const;
    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;
    // Fitness computation
    vector_double fitness(const vector_double &) const;
    // Optimal solution
    vector_double best_known() const;
    // Problem name
    std::string get_name() const;
    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    // Pointers to member functions are used
    PAGMO_DLL_LOCAL vector_double fitness_impl(detail::cec2006_data::func_ptr, detail::cec2006_data::func_ptr,
                                               const vector_double &) const;

    // For the coverage analysis we do not cover the code below as its derived from a third party source
    // LCOV_EXCL_START

    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g01_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g01_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g02_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g02_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g03_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g03_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g04_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g04_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g05_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g05_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g06_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g06_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g07_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g07_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g08_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g08_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    PAGMO_DLL_LOCAL vector_double g08_fitness_impl(const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g09_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g09_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g10_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g10_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g11_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g11_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g12_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g12_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g13_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g13_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g14_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g14_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g15_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g15_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g16_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g16_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g17_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g17_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g18_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g18_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g19_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g19_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g20_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g20_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g21_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g21_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g22_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g22_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g23_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g23_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // Implementation of the objective function.
    PAGMO_DLL_LOCAL void g24_objfun_impl(vector_double &f, const vector_double &x) const;
    // Implementation of the constraint function.
    PAGMO_DLL_LOCAL void g24_compute_constraints_impl(vector_double &c, const vector_double &x) const;
    // -------------------------------------------

    // LCOV_EXCL_STOP

    // Vectors of pointers to member functions for
    // objfun/constraints computation.
    static const std::vector<detail::cec2006_data::func_ptr> s_o_ptr;
    static const std::vector<detail::cec2006_data::func_ptr> s_c_ptr;

    // problem id
    unsigned m_prob_id;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::cec2006)

#endif
