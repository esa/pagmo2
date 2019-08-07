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

#if defined(_MSC_VER)

// Disable various warnings from MSVC.
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)
#pragma warning(disable : 4503)
#pragma warning(disable : 4244)

#endif

#include <pygmo/python_includes.hpp>

// See: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// In every cpp file we need to make sure this is included before everything else,
// with the correct #defines.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include <pygmo/numpy.hpp>

#include <string>
#include <tuple>

#include <boost/python/args.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/config.hpp>

#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/algorithms/gwo.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/null_algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>

#if defined(PAGMO_WITH_NLOPT)
#include <pagmo/algorithms/nlopt.hpp>
#endif

#include <pygmo/algorithm_exposition_suite.hpp>
#include <pygmo/common_utils.hpp>
#include <pygmo/docstrings.hpp>
#include <pygmo/expose_algorithms.hpp>

using namespace pagmo;
namespace bp = boost::python;

namespace pygmo
{

void expose_algorithms_1()
{
    // Null algo.
    auto na = expose_algorithm_pygmo<null_algorithm>("null_algorithm", null_algorithm_docstring().c_str());
    // PSO
    auto pso_ = expose_algorithm_pygmo<pso>("pso", pso_docstring().c_str());
    pso_.def(bp::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool>(
        (bp::arg("gen") = 1u, bp::arg("omega") = 0.7298, bp::arg("eta1") = 2.05, bp::arg("eta2") = 2.05,
         bp::arg("max_vel") = 0.5, bp::arg("variant") = 5u, bp::arg("neighb_type") = 2u, bp::arg("neighb_param") = 4u,
         bp::arg("memory") = false)));
    pso_.def(bp::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("omega") = 0.7298, bp::arg("eta1") = 2.05, bp::arg("eta2") = 2.05,
         bp::arg("max_vel") = 0.5, bp::arg("variant") = 5u, bp::arg("neighb_type") = 2u, bp::arg("neighb_param") = 4u,
         bp::arg("memory") = false, bp::arg("seed"))));
    expose_algo_log(pso_, pso_get_log_docstring().c_str());
    pso_.def("get_seed", &pso::get_seed, generic_uda_get_seed_docstring().c_str());

    // PSO (generational)
    auto pso_gen_ = expose_algorithm_pygmo<pso_gen>("pso_gen", pso_gen_docstring().c_str());
    pso_gen_.def(bp::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool>(
        (bp::arg("gen") = 1u, bp::arg("omega") = 0.7298, bp::arg("eta1") = 2.05, bp::arg("eta2") = 2.05,
         bp::arg("max_vel") = 0.5, bp::arg("variant") = 5u, bp::arg("neighb_type") = 2u, bp::arg("neighb_param") = 4u,
         bp::arg("memory") = false)));
    pso_gen_.def(bp::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("omega") = 0.7298, bp::arg("eta1") = 2.05, bp::arg("eta2") = 2.05,
         bp::arg("max_vel") = 0.5, bp::arg("variant") = 5u, bp::arg("neighb_type") = 2u, bp::arg("neighb_param") = 4u,
         bp::arg("memory") = false, bp::arg("seed"))));
    expose_algo_log(pso_gen_, pso_gen_get_log_docstring().c_str());
    pso_gen_.def("get_seed", &pso_gen::get_seed, generic_uda_get_seed_docstring().c_str());

    // SEA
    auto sea_ = expose_algorithm_pygmo<sea>("sea", sea_docstring().c_str());
    sea_.def(bp::init<unsigned>((bp::arg("gen") = 1u)));
    sea_.def(bp::init<unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("seed"))));
    expose_algo_log(sea_, sea_get_log_docstring().c_str());
    sea_.def("get_seed", &sea::get_seed, generic_uda_get_seed_docstring().c_str());

    // IHS
    auto ihs_ = expose_algorithm_pygmo<ihs>("ihs", ihs_docstring().c_str());
    ihs_.def(bp::init<unsigned, double, double, double, double, double>(
        (bp::arg("gen") = 1u, bp::arg("phmcr") = 0.85, bp::arg("ppar_min") = 0.35, bp::arg("ppar_max") = 0.99,
         bp::arg("bw_min") = 1E-5, bp::arg("bw_max") = 1.)));
    ihs_.def(bp::init<unsigned, double, double, double, double, double, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("phmcr") = 0.85, bp::arg("ppar_min") = 0.35, bp::arg("ppar_max") = 0.99,
         bp::arg("bw_min") = 1E-5, bp::arg("bw_max") = 1., bp::arg("seed"))));
    // ihs needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    ihs_.def("get_log", lcast([](const ihs &a) -> bp::list {
                 bp::list retval;
                 for (const auto &t : a.get_log()) {
                     retval.append(bp::make_tuple(std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                                  std::get<4>(t), std::get<5>(t), std::get<6>(t),
                                                  v_to_a(std::get<7>(t))));
                 }
                 return retval;
             }),
             ihs_get_log_docstring().c_str());
    ihs_.def("get_seed", &ihs::get_seed, generic_uda_get_seed_docstring().c_str());

    // SGA
    auto sga_ = expose_algorithm_pygmo<sga>("sga", sga_docstring().c_str());
    sga_.def(bp::init<unsigned, double, double, double, double, unsigned, std::string, std::string, std::string>(
        (bp::arg("gen") = 1u, bp::arg("cr") = 0.9, bp::arg("eta_c") = 1., bp::arg("m") = 0.02, bp::arg("param_m") = 1.,
         bp::arg("param_s") = 2u, bp::arg("crossover") = "exponential", bp::arg("mutation") = "polynomial",
         bp::arg("selection") = "tournament")));
    sga_.def(
        bp::init<unsigned, double, double, double, double, unsigned, std::string, std::string, std::string, unsigned>(
            (bp::arg("gen") = 1u, bp::arg("cr") = 0.9, bp::arg("eta_c") = 1., bp::arg("m") = 0.02,
             bp::arg("param_m") = 1., bp::arg("param_s") = 2u, bp::arg("crossover") = "exponential",
             bp::arg("mutation") = "polynomial", bp::arg("selection") = "tournament", bp::arg("seed"))));
    expose_algo_log(sga_, sga_get_log_docstring().c_str());
    sga_.def("get_seed", &sga::get_seed, generic_uda_get_seed_docstring().c_str());

    // SIMULATED ANNEALING
    auto simulated_annealing_
        = expose_algorithm_pygmo<simulated_annealing>("simulated_annealing", simulated_annealing_docstring().c_str());
    simulated_annealing_.def(bp::init<double, double, unsigned, unsigned, unsigned, double>(
        (bp::arg("Ts") = 10., bp::arg("Tf") = 0.1, bp::arg("n_T_adj") = 10u, bp::arg("n_range_adj") = 1u,
         bp::arg("bin_size") = 20u, bp::arg("start_range") = 1.)));
    simulated_annealing_.def(bp::init<double, double, unsigned, unsigned, unsigned, double, unsigned>(
        (bp::arg("Ts") = 10., bp::arg("Tf") = 0.1, bp::arg("n_T_adj") = 10u, bp::arg("n_range_adj") = 10u,
         bp::arg("bin_size") = 10u, bp::arg("start_range") = 1., bp::arg("seed"))));
    expose_algo_log(simulated_annealing_, simulated_annealing_get_log_docstring().c_str());
    simulated_annealing_.def("get_seed", &simulated_annealing::get_seed, generic_uda_get_seed_docstring().c_str());
    expose_not_population_based(simulated_annealing_, "simulated_annealing");

    // SADE
    auto sade_ = expose_algorithm_pygmo<sade>("sade", sade_docstring().c_str());
    sade_.def(bp::init<unsigned, unsigned, unsigned, double, double, bool>(
        (bp::arg("gen") = 1u, bp::arg("variant") = 2u, bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6,
         bp::arg("xtol") = 1e-6, bp::arg("memory") = false)));
    sade_.def(bp::init<unsigned, unsigned, unsigned, double, double, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("variant") = 2u, bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6,
         bp::arg("xtol") = 1e-6, bp::arg("memory") = false, bp::arg("seed"))));
    expose_algo_log(sade_, sade_get_log_docstring().c_str());
    sade_.def("get_seed", &sade::get_seed, generic_uda_get_seed_docstring().c_str());

    // NSGA2
    auto nsga2_ = expose_algorithm_pygmo<nsga2>("nsga2", nsga2_docstring().c_str());
    nsga2_.def(bp::init<unsigned, double, double, double, double>((bp::arg("gen") = 1u, bp::arg("cr") = 0.95,
                                                                   bp::arg("eta_c") = 10., bp::arg("m") = 0.01,
                                                                   bp::arg("eta_m") = 10.)));
    nsga2_.def(bp::init<unsigned, double, double, double, double, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("cr") = 0.95, bp::arg("eta_c") = 10., bp::arg("m") = 0.01, bp::arg("eta_m") = 10.,
         bp::arg("seed"))));
    // nsga2 needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    nsga2_.def("get_log", lcast([](const nsga2 &a) -> bp::list {
                   bp::list retval;
                   for (const auto &t : a.get_log()) {
                       retval.append(bp::make_tuple(std::get<0>(t), std::get<1>(t), v_to_a(std::get<2>(t))));
                   }
                   return retval;
               }),
               nsga2_get_log_docstring().c_str());

    nsga2_.def("get_seed", &nsga2::get_seed, generic_uda_get_seed_docstring().c_str());
    nsga2_.def("set_bfe", &nsga2::set_bfe, nsga2_set_bfe_docstring().c_str(), bp::arg("b"));

    // GACO
    auto gaco_ = expose_algorithm_pygmo<gaco>("gaco", gaco_docstring().c_str());
    gaco_.def(
        bp::init<unsigned, unsigned, double, double, double, unsigned, unsigned, unsigned, unsigned, double, bool>(
            (bp::arg("gen") = 100u, bp::arg("ker") = 63u, bp::arg("q") = 1.0, bp::arg("oracle") = 0.,
             bp::arg("acc") = 0.01, bp::arg("threshold") = 1u, bp::arg("n_gen_mark") = 7u, bp::arg("impstop") = 100000u,
             bp::arg("evalstop") = 100000u, bp::arg("focus") = 0., bp::arg("memory") = false)));
    gaco_.def(bp::init<unsigned, unsigned, double, double, double, unsigned, unsigned, unsigned, unsigned, double, bool,
                       unsigned>(
        (bp::arg("gen") = 100u, bp::arg("ker") = 63u, bp::arg("q") = 1.0, bp::arg("oracle") = 0., bp::arg("acc") = 0.01,
         bp::arg("threshold") = 1u, bp::arg("n_gen_mark") = 7u, bp::arg("impstop") = 100000u,
         bp::arg("evalstop") = 100000u, bp::arg("focus") = 0., bp::arg("memory") = false, bp::arg("seed"))));
    expose_algo_log(gaco_, gaco_get_log_docstring().c_str());
    gaco_.def("get_seed", &gaco::get_seed, generic_uda_get_seed_docstring().c_str());
    gaco_.def("set_bfe", &gaco::set_bfe, gaco_set_bfe_docstring().c_str(), bp::arg("b"));

    // GWO
    auto gwo_ = expose_algorithm_pygmo<gwo>("gwo", gwo_docstring().c_str());
    gwo_.def(bp::init<unsigned>((bp::arg("gen") = 1u)));
    gwo_.def(bp::init<unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("seed"))));
    expose_algo_log(gwo_, gwo_get_log_docstring().c_str());
    gwo_.def("get_seed", &gwo::get_seed, generic_uda_get_seed_docstring().c_str());

#if defined(PAGMO_WITH_NLOPT)
    // NLopt.
    auto nlopt_ = expose_algorithm_pygmo<nlopt>("nlopt", nlopt_docstring().c_str());
    nlopt_.def(bp::init<const std::string &>((bp::arg("solver"))));
    // Properties for the stopping criteria.
    add_property(nlopt_, "stopval", &nlopt::get_stopval, &nlopt::set_stopval, nlopt_stopval_docstring().c_str());
    add_property(nlopt_, "ftol_rel", &nlopt::get_ftol_rel, &nlopt::set_ftol_rel, nlopt_ftol_rel_docstring().c_str());
    add_property(nlopt_, "ftol_abs", &nlopt::get_ftol_abs, &nlopt::set_ftol_abs, nlopt_ftol_abs_docstring().c_str());
    add_property(nlopt_, "xtol_rel", &nlopt::get_xtol_rel, &nlopt::set_xtol_rel, nlopt_xtol_rel_docstring().c_str());
    add_property(nlopt_, "xtol_abs", &nlopt::get_xtol_abs, &nlopt::set_xtol_abs, nlopt_xtol_abs_docstring().c_str());
    add_property(nlopt_, "maxeval", &nlopt::get_maxeval, &nlopt::set_maxeval, nlopt_maxeval_docstring().c_str());
    add_property(nlopt_, "maxtime", &nlopt::get_maxtime, &nlopt::set_maxtime, nlopt_maxtime_docstring().c_str());
    expose_not_population_based(nlopt_, "nlopt");
    expose_algo_log(nlopt_, nlopt_get_log_docstring().c_str());
    nlopt_.def("get_last_opt_result", lcast([](const nlopt &n) { return static_cast<int>(n.get_last_opt_result()); }),
               nlopt_get_last_opt_result_docstring().c_str());
    nlopt_.def("get_solver_name", &nlopt::get_solver_name, nlopt_get_solver_name_docstring().c_str());
    add_property(nlopt_, "local_optimizer", bp::make_function(lcast([](nlopt &n) { return n.get_local_optimizer(); }),
                                                              bp::return_internal_reference<>()),
                 lcast([](nlopt &n, const nlopt *ptr) {
                     if (ptr) {
                         n.set_local_optimizer(*ptr);
                     } else {
                         n.unset_local_optimizer();
                     }
                 }),
                 nlopt_local_optimizer_docstring().c_str());
#endif
}
} // namespace pygmo
