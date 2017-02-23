/* Copyright 2017 PaGMO development team

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

#ifndef PYGMO_DOCSTRINGS_HPP
#define PYGMO_DOCSTRINGS_HPP

#include <string>

namespace pygmo
{
// population
std::string population_docstring();
std::string population_push_back_docstring();
std::string population_decision_vector_docstring();
std::string population_best_idx_docstring();
std::string population_worst_idx_docstring();
std::string population_size_docstring();
std::string population_set_xf_docstring();
std::string population_set_x_docstring();
std::string population_set_problem_seed_docstring();
std::string population_get_problem_docstring();
std::string population_get_f_docstring();
std::string population_get_x_docstring();
std::string population_get_ID_docstring();
std::string population_get_seed_docstring();

// problem
std::string problem_docstring();
std::string problem_get_best_docstring(const std::string &);
std::string problem_fitness_docstring();
std::string problem_get_bounds_docstring();
std::string problem_get_nec_docstring();
std::string problem_get_nic_docstring();
std::string problem_get_nobj_docstring();
std::string problem_get_nx_docstring();
std::string problem_get_nf_docstring();
std::string problem_get_nc_docstring();
std::string problem_c_tol_docstring();
std::string problem_get_fevals_docstring();
std::string problem_get_gevals_docstring();
std::string problem_get_hevals_docstring();
std::string problem_has_set_seed_docstring();
std::string problem_set_seed_docstring();
std::string problem_feasibility_x_docstring();
std::string problem_feasibility_f_docstring();
std::string problem_has_gradient_docstring();
std::string problem_gradient_docstring();
std::string problem_has_gradient_sparsity_docstring();
std::string problem_gradient_sparsity_docstring();
std::string problem_has_hessians_docstring();
std::string problem_hessians_docstring();
std::string problem_has_hessians_sparsity_docstring();
std::string problem_hessians_sparsity_docstring();
std::string problem_get_name_docstring();
std::string problem_get_extra_info_docstring();
std::string problem_get_thread_safety_docstring();

// translate
std::string translate_docstring();
std::string translate_translation_docstring();

// decompose
std::string decompose_docstring();
std::string decompose_decompose_fitness_docstring();
std::string decompose_original_fitness_docstring();
std::string decompose_z_docstring();

// algorithm
std::string algorithm_docstring();
std::string algorithm_evolve_docstring();
std::string algorithm_set_seed_docstring();
std::string algorithm_has_set_seed_docstring();
std::string algorithm_set_verbosity_docstring();
std::string algorithm_has_set_verbosity_docstring();
std::string algorithm_get_name_docstring();
std::string algorithm_get_extra_info_docstring();
std::string algorithm_get_thread_safety_docstring();

// mbh.
std::string mbh_docstring();
std::string mbh_get_seed_docstring();
std::string mbh_get_verbosity_docstring();
std::string mbh_set_perturb_docstring();
std::string mbh_get_log_docstring();
std::string mbh_get_perturb_docstring();

// user - problems
std::string null_problem_docstring();
std::string rosenbrock_docstring();
std::string dtlz_docstring();
std::string dtlz_p_distance_docstring();
std::string zdt_p_distance_docstring();
std::string cec2013_docstring();
std::string get_best_docstring(const std::string &);

// user - algorithms
std::string null_algorithm_docstring();
std::string cmaes_docstring();
std::string cmaes_get_log_docstring();
std::string compass_search_docstring();
std::string compass_search_get_log_docstring();
std::string de_docstring();
std::string de_get_log_docstring();
std::string de1220_docstring();
std::string de1220_get_log_docstring();
std::string moead_docstring();
std::string moead_get_log_docstring();
std::string nsga2_docstring();
std::string nsga2_get_log_docstring();
std::string pso_docstring();
std::string pso_get_log_docstring();
std::string sade_docstring();
std::string sade_get_log_docstring();
std::string simulated_annealing_docstring();
std::string simulated_annealing_get_log_docstring();
std::string generic_uda_get_seed_docstring();

// utilities
// hypervolume
std::string hv_init1_docstring();
std::string hv_init2_docstring();
std::string hv_compute_docstring();
std::string hv_contributions_docstring();
std::string hv_exclusive_docstring();
std::string hv_greatest_contributor_docstring();
std::string hv_least_contributor_docstring();
std::string hv_refpoint_docstring();
std::string hvwfg_docstring();
std::string hv2d_docstring();
std::string hv3d_docstring();
std::string bf_approx_docstring();
std::string bf_fpras_docstring();
// stand alone functions
std::string fast_non_dominated_sorting_docstring();
std::string ideal_docstring();
std::string nadir_docstring();
}

#endif
