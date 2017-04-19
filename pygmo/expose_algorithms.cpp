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

#include "python_includes.hpp"

// See: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// In every cpp file We need to make sure this is included before everything else,
// with the correct #defines.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include "numpy.hpp"

#if defined(_MSC_VER)

// Disable various warnings from MSVC.
#pragma warning(push, 0)
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)
#pragma warning(disable : 4503)

#endif

#include <boost/any.hpp>
#include <boost/python/args.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>
#include <string>
#include <tuple>

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_EIGEN3)
#include <pagmo/algorithms/cmaes.hpp>
#endif
#include <pagmo/algorithms/bee_colony.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/algorithms/moead.hpp>
#if defined(PAGMO_WITH_NLOPT)
#include <pagmo/algorithms/nlopt.hpp>
#endif
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/threading.hpp>

#include "algorithm_exposition_suite.hpp"
#include "common_utils.hpp"
#include "docstrings.hpp"

#if defined(_MSC_VER)

#pragma warning(pop)

#endif

using namespace pagmo;
namespace bp = boost::python;

namespace pygmo
{

// A test algo.
struct test_algorithm {
    population evolve(const population &pop) const
    {
        return pop;
    }
    // Set/get an internal value to test extraction semantics.
    void set_n(int n)
    {
        m_n = n;
    }
    int get_n() const
    {
        return m_n;
    }
    int m_n = 1;
};

// A thread unsafe test algo.
struct tu_test_algorithm {
    population evolve(const population &pop) const
    {
        return pop;
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

void expose_algorithms()
{
    // MBH meta-algo.
    auto mbh_ = expose_algorithm<mbh>("mbh", mbh_docstring().c_str());
    mbh_.def("__init__",
             bp::make_constructor(lcast([](const algorithm &a, unsigned stop, const bp::object &perturb,
                                           unsigned seed) { return ::new pagmo::mbh(a, stop, to_vd(perturb), seed); }),
                                  bp::default_call_policies()));
    mbh_.def("__init__", bp::make_constructor(lcast([](const algorithm &a, unsigned stop, const bp::object &perturb) {
                                                  return ::new pagmo::mbh(a, stop, to_vd(perturb),
                                                                          pagmo::random_device::next());
                                              }),
                                              bp::default_call_policies()));
    mbh_.def("get_seed", &mbh::get_seed, mbh_get_seed_docstring().c_str());
    mbh_.def("get_verbosity", &mbh::get_verbosity, mbh_get_verbosity_docstring().c_str());
    mbh_.def("set_perturb", lcast([](mbh &a, const bp::object &o) { a.set_perturb(to_vd(o)); }),
             mbh_set_perturb_docstring().c_str(), (bp::arg("perturb")));
    expose_algo_log(mbh_, mbh_get_log_docstring().c_str());
    mbh_.def("get_perturb", lcast([](const mbh &a) { return v_to_a(a.get_perturb()); }),
             mbh_get_perturb_docstring().c_str());
    mbh_.add_property("inner_algorithm",
                      bp::make_function(lcast([](mbh &uda) -> algorithm & { return uda.get_inner_algorithm(); }),
                                        bp::return_internal_reference<>()),
                      generic_uda_inner_algorithm_docstring().c_str());
    // cstrs_self_adaptive meta-algo.
    auto cstrs_sa
        = expose_algorithm<cstrs_self_adaptive>("cstrs_self_adaptive", cstrs_self_adaptive_docstring().c_str());
    cstrs_sa.def("__init__", bp::make_constructor(lcast([](unsigned iters, const algorithm &a, unsigned seed) {
                                                      return ::new pagmo::cstrs_self_adaptive(iters, a, seed);
                                                  }),
                                                  bp::default_call_policies()));
    cstrs_sa.def("__init__", bp::make_constructor(lcast([](unsigned iters, const algorithm &a) {
                                                      return ::new pagmo::cstrs_self_adaptive(
                                                          iters, a, pagmo::random_device::next());
                                                  }),
                                                  bp::default_call_policies()));
    expose_algo_log(cstrs_sa, cstrs_self_adaptive_get_log_docstring().c_str());
    cstrs_sa.add_property(
        "inner_algorithm",
        bp::make_function(lcast([](cstrs_self_adaptive &uda) -> algorithm & { return uda.get_inner_algorithm(); }),
                          bp::return_internal_reference<>()),
        generic_uda_inner_algorithm_docstring().c_str());

    // Test algo.
    auto test_a = expose_algorithm<test_algorithm>("_test_algorithm", "A test algorithm.");
    test_a.def("get_n", &test_algorithm::get_n);
    test_a.def("set_n", &test_algorithm::set_n);
    // Thread unsafe test algo.
    expose_algorithm<tu_test_algorithm>("_tu_test_algorithm", "A thread unsafe test algorithm.");
    // Null algo.
    auto na = expose_algorithm<null_algorithm>("null_algorithm", null_algorithm_docstring().c_str());
    // ARTIFICIAL BEE COLONY
    auto bee_colony_ = expose_algorithm<bee_colony>("bee_colony", bee_colony_docstring().c_str());
    bee_colony_.def(bp::init<unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("limit") = 1u)));
    bee_colony_.def(
        bp::init<unsigned, unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("limit") = 20u, bp::arg("seed"))));
    expose_algo_log(bee_colony_, bee_colony_get_log_docstring().c_str());
    bee_colony_.def("get_seed", &bee_colony::get_seed, generic_uda_get_seed_docstring().c_str());
    // DE
    auto de_ = expose_algorithm<de>("de", de_docstring().c_str());
    de_.def(bp::init<unsigned, double, double, unsigned, double, double>(
        (bp::arg("gen") = 1u, bp::arg("F") = .8, bp::arg("CR") = .9, bp::arg("variant") = 2u, bp::arg("ftol") = 1e-6,
         bp::arg("tol") = 1E-6)));
    de_.def(bp::init<unsigned, double, double, unsigned, double, double, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("F") = .8, bp::arg("CR") = .9, bp::arg("variant") = 2u, bp::arg("ftol") = 1e-6,
         bp::arg("tol") = 1E-6, bp::arg("seed"))));
    expose_algo_log(de_, de_get_log_docstring().c_str());
    de_.def("get_seed", &de::get_seed, generic_uda_get_seed_docstring().c_str());
    // COMPASS SEARCH
    auto compass_search_ = expose_algorithm<compass_search>("compass_search", compass_search_docstring().c_str());
    compass_search_.def(
        bp::init<unsigned, double, double, double>((bp::arg("max_fevals") = 1u, bp::arg("start_range") = .1,
                                                    bp::arg("stop_range") = .01, bp::arg("reduction_coeff") = .5)));
    expose_algo_log(compass_search_, compass_search_get_log_docstring().c_str());
    compass_search_.def("get_max_fevals", &compass_search::get_max_fevals);
    compass_search_.def("get_start_range", &compass_search::get_start_range);
    compass_search_.def("get_stop_range", &compass_search::get_stop_range);
    compass_search_.def("get_reduction_coeff", &compass_search::get_reduction_coeff);
    compass_search_.def("get_verbosity", &compass_search::get_verbosity);
    // PSO
    auto pso_ = expose_algorithm<pso>("pso", pso_docstring().c_str());
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
    // SEA
    auto sea_ = expose_algorithm<sea>("sea", sea_docstring().c_str());
    sea_.def(bp::init<unsigned>((bp::arg("gen") = 1u)));
    sea_.def(bp::init<unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("seed"))));
    expose_algo_log(sea_, "");
    sea_.def("get_seed", &sea::get_seed, generic_uda_get_seed_docstring().c_str());
    // SIMULATED ANNEALING
    auto simulated_annealing_
        = expose_algorithm<simulated_annealing>("simulated_annealing", simulated_annealing_docstring().c_str());
    simulated_annealing_.def(bp::init<double, double, unsigned, unsigned, unsigned, double>(
        (bp::arg("Ts") = 10., bp::arg("Tf") = 0.1, bp::arg("n_T_adj") = 10u, bp::arg("n_range_adj") = 1u,
         bp::arg("bin_size") = 20u, bp::arg("start_range") = 1.)));
    simulated_annealing_.def(bp::init<double, double, unsigned, unsigned, unsigned, double, unsigned>(
        (bp::arg("Ts") = 10., bp::arg("Tf") = 0.1, bp::arg("n_T_adj") = 10u, bp::arg("n_range_adj") = 10u,
         bp::arg("bin_size") = 10u, bp::arg("start_range") = 1., bp::arg("seed"))));
    expose_algo_log(simulated_annealing_, simulated_annealing_get_log_docstring().c_str());
    simulated_annealing_.def("get_seed", &simulated_annealing::get_seed, generic_uda_get_seed_docstring().c_str());
    // SADE
    auto sade_ = expose_algorithm<sade>("sade", sade_docstring().c_str());
    sade_.def(bp::init<unsigned, unsigned, unsigned, double, double, bool>(
        (bp::arg("gen") = 1u, bp::arg("variant") = 2u, bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6,
         bp::arg("xtol") = 1e-6, bp::arg("memory") = false)));
    sade_.def(bp::init<unsigned, unsigned, unsigned, double, double, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("variant") = 2u, bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6,
         bp::arg("xtol") = 1e-6, bp::arg("memory") = false, bp::arg("seed"))));
    expose_algo_log(sade_, sade_get_log_docstring().c_str());
    sade_.def("get_seed", &sade::get_seed, generic_uda_get_seed_docstring().c_str());
    // DE-1220
    auto de1220_ = expose_algorithm<de1220>("de1220", de1220_docstring().c_str());
    // Helper to get the list of default allowed variants for de1220.
    auto de1220_allowed_variants = []() -> bp::list {
        bp::list retval;
        for (const auto &n : de1220_statics<void>::allowed_variants) {
            retval.append(n);
        }
        return retval;
    };
    de1220_.def("__init__",
                bp::make_constructor(lcast([](unsigned gen, const bp::object &allowed_variants, unsigned variant_adptv,
                                              double ftol, double xtol, bool memory) -> de1220 * {
                                         auto av = to_vu(allowed_variants);
                                         return ::new de1220(gen, av, variant_adptv, ftol, xtol, memory);
                                     }),
                                     bp::default_call_policies(),
                                     (bp::arg("gen") = 1u, bp::arg("allowed_variants") = de1220_allowed_variants(),
                                      bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6,
                                      bp::arg("memory") = false)));
    de1220_.def("__init__",
                bp::make_constructor(lcast([](unsigned gen, const bp::object &allowed_variants, unsigned variant_adptv,
                                              double ftol, double xtol, bool memory, unsigned seed) -> de1220 * {
                                         auto av = to_vu(allowed_variants);
                                         return ::new de1220(gen, av, variant_adptv, ftol, xtol, memory, seed);
                                     }),
                                     bp::default_call_policies(),
                                     (bp::arg("gen") = 1u, bp::arg("allowed_variants") = de1220_allowed_variants(),
                                      bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6,
                                      bp::arg("memory") = false, bp::arg("seed"))));
    expose_algo_log(de1220_, de1220_get_log_docstring().c_str());
    de1220_.def("get_seed", &de1220::get_seed, generic_uda_get_seed_docstring().c_str());
// CMA-ES
#if defined(PAGMO_WITH_EIGEN3)
    auto cmaes_ = expose_algorithm<cmaes>("cmaes", cmaes_docstring().c_str());
    cmaes_.def(bp::init<unsigned, double, double, double, double, double, double, double, bool>(
        (bp::arg("gen") = 1u, bp::arg("cc") = -1., bp::arg("cs") = -1., bp::arg("c1") = -1., bp::arg("cmu") = -1.,
         bp::arg("sigma0") = 0.5, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false)));
    cmaes_.def(bp::init<unsigned, double, double, double, double, double, double, double, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("cc") = -1., bp::arg("cs") = -1., bp::arg("c1") = -1., bp::arg("cmu") = -1.,
         bp::arg("sigma0") = 0.5, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false,
         bp::arg("seed"))));
    expose_algo_log(cmaes_, cmaes_get_log_docstring().c_str());
    cmaes_.def("get_seed", &cmaes::get_seed, generic_uda_get_seed_docstring().c_str());
#endif
    // MOEA/D - DE
    auto moead_ = expose_algorithm<moead>("moead", moead_docstring().c_str());
    moead_.def(bp::init<unsigned, std::string, std::string, unsigned, double, double, double, double, unsigned, bool>(
        (bp::arg("gen") = 1u, bp::arg("weight_generation") = "grid", bp::arg("decomposition") = "tchebycheff",
         bp::arg("neighbours") = 20u, bp::arg("CR") = 1., bp::arg("F") = 0.5, bp::arg("eta_m") = 20,
         bp::arg("realb") = 0.9, bp::arg("limit") = 2u, bp::arg("preserve_diversity") = true)));
    moead_.def(bp::init<unsigned, std::string, std::string, unsigned, double, double, double, double, unsigned, bool,
                        unsigned>(
        (bp::arg("gen") = 1u, bp::arg("weight_generation") = "grid", bp::arg("decomposition") = "tchebycheff",
         bp::arg("neighbours") = 20u, bp::arg("CR") = 1., bp::arg("F") = 0.5, bp::arg("eta_m") = 20,
         bp::arg("realb") = 0.9, bp::arg("limit") = 2u, bp::arg("preserve_diversity") = true, bp::arg("seed"))));
    // moead needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    moead_.def("get_log", lcast([](const moead &a) -> bp::list {
                   bp::list retval;
                   for (const auto &t : a.get_log()) {
                       retval.append(
                           bp::make_tuple(std::get<0>(t), std::get<1>(t), std::get<2>(t), v_to_a(std::get<3>(t))));
                   }
                   return retval;
               }),
               moead_get_log_docstring().c_str());

    moead_.def("get_seed", &moead::get_seed, generic_uda_get_seed_docstring().c_str());
    // NSGA2
    auto nsga2_ = expose_algorithm<nsga2>("nsga2", nsga2_docstring().c_str());
    nsga2_.def(bp::init<unsigned, double, double, double, double, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("cr") = 0.95, bp::arg("eta_c") = 10., bp::arg("m") = 0.01, bp::arg("eta_m") = 10.,
         bp::arg("int_dim") = 0)));
    nsga2_.def(bp::init<unsigned, double, double, double, double, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("cr") = 0.95, bp::arg("eta_c") = 10., bp::arg("m") = 0.01, bp::arg("eta_m") = 10.,
         bp::arg("int_dim") = 0, bp::arg("seed"))));
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

#if defined(PAGMO_WITH_NLOPT)
    // NLopt.
    auto nlopt_ = expose_algorithm<nlopt>("nlopt", nlopt_docstring().c_str());
    nlopt_.def(bp::init<const std::string &>((bp::arg("solver"))));
    // Properties for the stopping criteria.
    nlopt_.add_property("stopval", &nlopt::get_stopval, &nlopt::set_stopval, nlopt_stopval_docstring().c_str());
    nlopt_.add_property("ftol_rel", &nlopt::get_ftol_rel, &nlopt::set_ftol_rel, nlopt_ftol_rel_docstring().c_str());
    nlopt_.add_property("ftol_abs", &nlopt::get_ftol_abs, &nlopt::set_ftol_abs, nlopt_ftol_abs_docstring().c_str());
    nlopt_.add_property("xtol_rel", &nlopt::get_xtol_rel, &nlopt::set_xtol_rel, nlopt_xtol_rel_docstring().c_str());
    nlopt_.add_property("xtol_abs", &nlopt::get_xtol_abs, &nlopt::set_xtol_abs, nlopt_xtol_abs_docstring().c_str());
    nlopt_.add_property("maxeval", &nlopt::get_maxeval, &nlopt::set_maxeval, nlopt_maxeval_docstring().c_str());
    nlopt_.add_property("maxtime", &nlopt::get_maxtime, &nlopt::set_maxtime, nlopt_maxtime_docstring().c_str());
    // Selection/replacement.
    nlopt_.add_property(
        "selection", lcast([](const nlopt &n) -> bp::object {
            auto s = n.get_selection();
            if (boost::any_cast<std::string>(&s)) {
                return bp::str(boost::any_cast<std::string>(s));
            }
            return bp::object(boost::any_cast<population::size_type>(s));
        }),
        lcast([](nlopt &n, const bp::object &o) {
            bp::extract<std::string> e_str(o);
            if (e_str.check()) {
                n.set_selection(e_str());
                return;
            }
            bp::extract<population::size_type> e_idx(o);
            if (e_idx.check()) {
                n.set_selection(e_idx());
                return;
            }
            pygmo_throw(::PyExc_TypeError,
                        ("cannot convert the input object '" + str(o) + "' of type '" + str(type(o))
                         + "' to either a selection policy (one of ['best', 'worst', 'random']) or an individual index")
                            .c_str());
        }),
        nlopt_selection_docstring().c_str());
    nlopt_.add_property(
        "replacement", lcast([](const nlopt &n) -> bp::object {
            auto s = n.get_replacement();
            if (boost::any_cast<std::string>(&s)) {
                return bp::str(boost::any_cast<std::string>(s));
            }
            return bp::object(boost::any_cast<population::size_type>(s));
        }),
        lcast([](nlopt &n, const bp::object &o) {
            bp::extract<std::string> e_str(o);
            if (e_str.check()) {
                n.set_replacement(e_str());
                return;
            }
            bp::extract<population::size_type> e_idx(o);
            if (e_idx.check()) {
                n.set_replacement(e_idx());
                return;
            }
            pygmo_throw(
                ::PyExc_TypeError,
                ("cannot convert the input object '" + str(o) + "' of type '" + str(type(o))
                 + "' to either a replacement policy (one of ['best', 'worst', 'random']) or an individual index")
                    .c_str());
        }),
        nlopt_replacement_docstring().c_str());
    nlopt_.def("set_random_sr_seed", &nlopt::set_random_sr_seed, nlopt_set_random_sr_seed_docstring().c_str());
    expose_algo_log(nlopt_, nlopt_get_log_docstring().c_str());
    nlopt_.def("get_last_opt_result", lcast([](const nlopt &n) { return static_cast<int>(n.get_last_opt_result()); }),
               nlopt_get_last_opt_result_docstring().c_str());
    nlopt_.def("get_solver_name", &nlopt::get_solver_name, nlopt_get_solver_name_docstring().c_str());
    nlopt_.add_property("local_optimizer", bp::make_function(lcast([](nlopt &n) { return n.get_local_optimizer(); }),
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
}
