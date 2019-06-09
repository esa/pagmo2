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

#ifndef PAGMO_PROBLEMS_WFG_HPP
#define PAGMO_PROBLEMS_WFG_HPP

#include <string>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// WFG problem test suite
//  This test suite was conceived to exceed the functionalities of previously implemented
//  test suites.
//  The WFG test suite was introduced by Simon Huband, Luigi Barone, Lyndon While, and Phil Hingston. In their paper the
//  authors
//  identify the absence of nonseparable multimodal problems in order to test multi-objective optimization algorithms.
//  In view of this, they propose a set of 9 different scalable multi-objective unconstrained problems (both in their
//  objectives and in their decision vectors).
//  See:
//  Huband, Simon, Hingston, Philip, Barone, Luigi and While Lyndon. "A Review of Multi-Objective Test Problems and a
//  Scalable Test Problem Toolkit". IEEE Transactions on Evolutionary Computation (2006), 10(5), 477-506. doi:
//  10.1109/TEVC.2005.861417.

class PAGMO_DLL_PUBLIC wfg
{
public:
    // Constructor
    // Will construct one problem from the Walking Fish Group (WFG) test-suite
    wfg(unsigned prob_id = 1u, vector_double::size_type dim_dvs = 5u, vector_double::size_type dim_obj = 3u,
        vector_double::size_type dim_k = 4u);
    // Fitness computation
    vector_double fitness(const vector_double &) const;

    // Number of objectives
    vector_double::size_type get_nobj() const;

    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;

    // Problem name
    std::string get_name() const;

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL double linear(const vector_double &, const vector_double::size_type) const;
    PAGMO_DLL_LOCAL double convex(const vector_double &, const vector_double::size_type) const;
    PAGMO_DLL_LOCAL double concave(const vector_double &, const vector_double::size_type) const;
    PAGMO_DLL_LOCAL double mixed(const double, const double, const double) const;
    PAGMO_DLL_LOCAL double disconnected(const double, const double, const double, const double) const;
    PAGMO_DLL_LOCAL double b_poly(const double, const double) const;
    PAGMO_DLL_LOCAL double b_flat(const double, const double, const double, const double) const;
    PAGMO_DLL_LOCAL double b_param(const double, const double, const double, const double, const double) const;
    PAGMO_DLL_LOCAL double s_linear(const double, const double) const;
    PAGMO_DLL_LOCAL double s_decept(const double, const double, const double, const double) const;
    PAGMO_DLL_LOCAL double s_multi(const double, const double, const double, const double) const;
    PAGMO_DLL_LOCAL double r_sum(const vector_double &, const vector_double &) const;
    PAGMO_DLL_LOCAL double r_nonsep(const vector_double &, const vector_double::size_type) const;
    PAGMO_DLL_LOCAL vector_double wfg1_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg2_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg3_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg4_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg5_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg6_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg7_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg8_fitness(const vector_double &) const;
    PAGMO_DLL_LOCAL vector_double wfg9_fitness(const vector_double &) const;

private:
    // Problem dimensions
    unsigned m_prob_id;
    vector_double::size_type m_dim_dvs;
    vector_double::size_type m_dim_obj;
    vector_double::size_type m_dim_k;
};
} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::wfg)

#endif
