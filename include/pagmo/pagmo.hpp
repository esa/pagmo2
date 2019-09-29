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

#ifndef PAGMO_PAGMO_HPP
#define PAGMO_PAGMO_HPP

#include <pagmo/config.hpp>

// Core.
#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

// Utils.
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/discrepancy.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/gradients_and_hessians.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>
#include <pagmo/utils/hv_algos/hv_bf_approx.hpp>
#include <pagmo/utils/hv_algos/hv_bf_fpras.hpp>
#include <pagmo/utils/hv_algos/hv_hv2d.hpp>
#include <pagmo/utils/hv_algos/hv_hv3d.hpp>
#include <pagmo/utils/hv_algos/hv_hvwfg.hpp>
#include <pagmo/utils/hypervolume.hpp>
#include <pagmo/utils/multi_objective.hpp>

// Algorithms.
#include <pagmo/algorithms/bee_colony.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/algorithms/gwo.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/algorithms/moead.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/null_algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>

#if defined(PAGMO_WITH_EIGEN3)
#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/algorithms/xnes.hpp>
#endif

#if defined(PAGMO_WITH_IPOPT)
#include <pagmo/algorithms/ipopt.hpp>
#endif

#if defined(PAGMO_WITH_NLOPT)
#include <pagmo/algorithms/nlopt.hpp>
#endif

// Islands.
#include <pagmo/islands/thread_island.hpp>

#if defined(PAGMO_WITH_FORK_ISLAND)
#include <pagmo/islands/fork_island.hpp>
#endif

// Problems.
#include <pagmo/problems/ackley.hpp>
#include <pagmo/problems/cec2006.hpp>
#include <pagmo/problems/cec2009.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/golomb_ruler.hpp>
#include <pagmo/problems/griewank.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/lennard_jones.hpp>
#include <pagmo/problems/luksan_vlcek1.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/problems/rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/problems/wfg.hpp>
#include <pagmo/problems/zdt.hpp>

// Enable conditionally the inclusion of these
// two problems in the global header. See
// the explanation in config.hpp.
#if defined(PAGMO_ENABLE_CEC2013)
#include <pagmo/problems/cec2013.hpp>
#endif

#if defined(PAGMO_ENABLE_CEC2014)
#include <pagmo/problems/cec2014.hpp>
#endif

// Batch evaluators.
#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>

// Replacement policies.
#include <pagmo/r_policies/fair_replace.hpp>

// Selection policies.
#include <pagmo/s_policies/select_best.hpp>

// Topologies.
#include <pagmo/topologies/base_bgl_topology.hpp>
#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/topologies/ring.hpp>
#include <pagmo/topologies/unconnected.hpp>

#endif
