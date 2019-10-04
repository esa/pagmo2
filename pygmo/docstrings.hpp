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

#ifndef PYGMO_DOCSTRINGS_HPP
#define PYGMO_DOCSTRINGS_HPP

#include <string>

namespace pygmo
{
// population
std::string population_docstring();
std::string population_push_back_docstring();
std::string population_random_decision_vector_docstring();
std::string population_best_idx_docstring();
std::string population_worst_idx_docstring();
std::string population_set_xf_docstring();
std::string population_set_x_docstring();
std::string population_get_f_docstring();
std::string population_get_x_docstring();
std::string population_get_ID_docstring();
std::string population_get_seed_docstring();
std::string population_champion_x_docstring();
std::string population_champion_f_docstring();
std::string population_problem_docstring();

// problem
std::string problem_docstring();
std::string problem_get_best_docstring(const std::string &);
std::string problem_fitness_docstring();
std::string problem_get_bounds_docstring();
std::string problem_batch_fitness_docstring();
std::string problem_has_batch_fitness_docstring();
std::string problem_get_lb_docstring();
std::string problem_get_ub_docstring();
std::string problem_get_nec_docstring();
std::string problem_get_nic_docstring();
std::string problem_get_nobj_docstring();
std::string problem_get_nx_docstring();
std::string problem_get_nix_docstring();
std::string problem_get_ncx_docstring();
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

// user - problems
std::string null_problem_docstring();
std::string rosenbrock_docstring();
std::string minlp_rastrigin_docstring();
std::string dtlz_docstring();
std::string dtlz_p_distance_docstring();
std::string zdt_p_distance_docstring();
std::string cec2006_docstring();
std::string cec2009_docstring();
std::string cec2013_docstring();
std::string cec2014_docstring();
std::string luksan_vlcek1_docstring();
std::string translate_docstring();
std::string translate_translation_docstring();
std::string decompose_docstring();
std::string decompose_decompose_fitness_docstring();
std::string decompose_original_fitness_docstring();
std::string decompose_z_docstring();
std::string unconstrain_docstring();
std::string get_best_docstring(const std::string &);
std::string generic_udp_inner_problem_docstring();
std::string wfg_docstring();

// user - algorithms
std::string null_algorithm_docstring();
std::string cmaes_docstring();
std::string cmaes_get_log_docstring();
std::string xnes_docstring();
std::string xnes_get_log_docstring();
std::string compass_search_docstring();
std::string compass_search_get_log_docstring();
std::string bee_colony_docstring();
std::string bee_colony_get_log_docstring();
std::string de_docstring();
std::string de_get_log_docstring();
std::string de1220_docstring();
std::string de1220_get_log_docstring();
std::string moead_docstring();
std::string moead_get_log_docstring();
std::string nsga2_set_bfe_docstring();
std::string nsga2_docstring();
std::string nsga2_get_log_docstring();
std::string nspso_set_bfe_docstring();
std::string nspso_docstring();
std::string nspso_get_log_docstring();
std::string gaco_set_bfe_docstring();
std::string gaco_docstring();
std::string gaco_get_log_docstring();
std::string gwo_docstring();
std::string gwo_get_log_docstring();
std::string pso_docstring();
std::string pso_get_log_docstring();
std::string pso_gen_set_bfe_docstring();
std::string pso_gen_docstring();
std::string pso_gen_get_log_docstring();
std::string sade_docstring();
std::string sade_get_log_docstring();
std::string simulated_annealing_docstring();
std::string simulated_annealing_get_log_docstring();
std::string cstrs_self_adaptive_docstring();
std::string cstrs_self_adaptive_get_log_docstring();
std::string mbh_docstring();
std::string mbh_get_seed_docstring();
std::string mbh_get_verbosity_docstring();
std::string mbh_set_perturb_docstring();
std::string mbh_get_log_docstring();
std::string mbh_get_perturb_docstring();
std::string sea_docstring();
std::string sea_get_log_docstring();
std::string ihs_docstring();
std::string ihs_get_log_docstring();
std::string sga_docstring();
std::string sga_get_log_docstring();
std::string nlopt_docstring();
std::string nlopt_stopval_docstring();
std::string nlopt_ftol_rel_docstring();
std::string nlopt_ftol_abs_docstring();
std::string nlopt_xtol_rel_docstring();
std::string nlopt_xtol_abs_docstring();
std::string nlopt_maxeval_docstring();
std::string nlopt_maxtime_docstring();
std::string nlopt_get_log_docstring();
std::string nlopt_get_last_opt_result_docstring();
std::string nlopt_get_solver_name_docstring();
std::string nlopt_local_optimizer_docstring();

std::string ipopt_docstring();
std::string ipopt_get_log_docstring();
std::string ipopt_get_last_opt_result_docstring();
std::string ipopt_set_string_option_docstring();
std::string ipopt_set_string_options_docstring();
std::string ipopt_get_string_options_docstring();
std::string ipopt_reset_string_options_docstring();
std::string ipopt_set_integer_option_docstring();
std::string ipopt_set_integer_options_docstring();
std::string ipopt_get_integer_options_docstring();
std::string ipopt_reset_integer_options_docstring();
std::string ipopt_set_numeric_option_docstring();
std::string ipopt_set_numeric_options_docstring();
std::string ipopt_get_numeric_options_docstring();
std::string ipopt_reset_numeric_options_docstring();

// base local solver common docstrings.
std::string bls_selection_docstring(const std::string &);
std::string bls_replacement_docstring(const std::string &);
std::string bls_set_random_sr_seed_docstring(const std::string &);

// common docstrings reusable by multiple udas, udps
std::string generic_uda_get_seed_docstring();
std::string generic_uda_inner_algorithm_docstring();

// utilities
// generic
std::string random_decision_vector_docstring();
std::string batch_random_decision_vector_docstring();
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
// multi-objective
std::string pareto_dominance_docstring();
std::string non_dominated_front_2d_docstring();
std::string crowding_distance_docstring();
std::string fast_non_dominated_sorting_docstring();
std::string sort_population_mo_docstring();
std::string select_best_N_mo_docstring();
std::string decomposition_weights_docstring();
std::string decompose_objectives_docstring();
std::string ideal_docstring();
std::string nadir_docstring();
// constrained
std::string compare_fc_docstring();
std::string sort_population_con_docstring();
// gradient and hessians
std::string estimate_sparsity_docstring();
std::string estimate_gradient_docstring();
std::string estimate_gradient_h_docstring();
// global rng
std::string set_global_rng_seed_docstring();

// island.
std::string island_docstring();
std::string island_evolve_docstring();
std::string island_wait_docstring();
std::string island_wait_check_docstring();
std::string island_status_docstring();
std::string island_get_algorithm_docstring();
std::string island_set_algorithm_docstring();
std::string island_get_population_docstring();
std::string island_set_population_docstring();
std::string island_get_name_docstring();
std::string island_get_extra_info_docstring();
std::string island_get_r_policy_docstring();
std::string island_get_s_policy_docstring();

// udi.
std::string thread_island_docstring();

// archipelago.
std::string archipelago_docstring();
std::string archipelago_evolve_docstring();
std::string archipelago_status_docstring();
std::string archipelago_wait_docstring();
std::string archipelago_wait_check_docstring();
std::string archipelago_getitem_docstring();
std::string archipelago_get_champions_f_docstring();
std::string archipelago_get_champions_x_docstring();
std::string archipelago_get_migrants_db_docstring();
std::string archipelago_get_migration_log_docstring();
std::string archipelago_get_topology_docstring();
std::string archipelago_get_migration_type_docstring();
std::string archipelago_set_migration_type_docstring();
std::string archipelago_get_migrant_handling_docstring();
std::string archipelago_set_migrant_handling_docstring();

// bfe.
std::string bfe_docstring();
std::string bfe_call_docstring();
std::string bfe_get_name_docstring();
std::string bfe_get_extra_info_docstring();
std::string bfe_get_thread_safety_docstring();

// udbfe.
std::string default_bfe_docstring();
std::string thread_bfe_docstring();
std::string member_bfe_docstring();

// topology.
std::string topology_docstring();
std::string topology_get_connections_docstring();
std::string topology_push_back_docstring();
std::string topology_get_name_docstring();
std::string topology_get_extra_info_docstring();

// udt.
std::string unconnected_docstring();
std::string base_bgl_num_vertices_docstring();
std::string base_bgl_are_adjacent_docstring();
std::string base_bgl_add_vertex_docstring();
std::string base_bgl_add_edge_docstring();
std::string base_bgl_remove_edge_docstring();
std::string base_bgl_set_weight_docstring();
std::string base_bgl_set_all_weights_docstring();
std::string ring_docstring();
std::string ring_get_weight_docstring();
std::string fully_connected_docstring();
std::string fully_connected_get_weight_docstring();
std::string fully_connected_num_vertices_docstring();

// r_policy.
std::string r_policy_docstring();
std::string r_policy_replace_docstring();
std::string r_policy_get_name_docstring();
std::string r_policy_get_extra_info_docstring();

// udrp.
std::string fair_replace_docstring();

// s_policy
std::string s_policy_docstring();
std::string s_policy_select_docstring();
std::string s_policy_get_name_docstring();
std::string s_policy_get_extra_info_docstring();

// udsp.
std::string select_best_docstring();

} // namespace pygmo

#endif
