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

#if defined(_MSC_VER)

// Disable various warnings from MSVC.
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)
#pragma warning(disable : 4503)
#pragma warning(disable : 4244)

#endif

#include <pygmo/python_includes.hpp>

// See: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// In every cpp file We need to make sure this is included before everything else,
// with the correct #defines.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include <pygmo/numpy.hpp>

#include <boost/any.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/str.hpp>
#include <boost/python/tuple.hpp>
#include <map>
#include <string>
#include <tuple>

#include <pagmo/config.hpp>

#include <pagmo/algorithm.hpp>
#if defined(PAGMO_WITH_EIGEN3)
#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/algorithms/xnes.hpp>
#endif
#include <pagmo/algorithms/bee_colony.hpp>
#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/de1220.hpp>
#if defined(PAGMO_WITH_IPOPT)
#include <IpTNLP.hpp>
#include <pagmo/algorithms/ipopt.hpp>
#endif
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/algorithms/moead.hpp>
#if defined(PAGMO_WITH_NLOPT)
#include <pagmo/algorithms/nlopt.hpp>
#endif
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/threading.hpp>

#include <pygmo/algorithm_exposition_suite.hpp>
#include <pygmo/common_utils.hpp>
#include <pygmo/docstrings.hpp>
#include <pygmo/pygmo_classes.hpp>

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

template <typename T>
static inline void expose_not_population_based(bp::class_<T> &c, const std::string &algo_name)
{
    // Selection/replacement.
    add_property(
        c, "selection", lcast([](const T &n) -> bp::object {
            auto s = n.get_selection();
            if (boost::any_cast<std::string>(&s)) {
                return bp::str(boost::any_cast<std::string>(s));
            }
            return bp::object(boost::any_cast<population::size_type>(s));
        }),
        lcast([](T &n, const bp::object &o) {
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
        bls_selection_docstring(algo_name).c_str());
    add_property(
        c, "replacement", lcast([](const T &n) -> bp::object {
            auto s = n.get_replacement();
            if (boost::any_cast<std::string>(&s)) {
                return bp::str(boost::any_cast<std::string>(s));
            }
            return bp::object(boost::any_cast<population::size_type>(s));
        }),
        lcast([](T &n, const bp::object &o) {
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
        bls_replacement_docstring(algo_name).c_str());
    c.def("set_random_sr_seed", &T::set_random_sr_seed, bls_set_random_sr_seed_docstring(algo_name).c_str());
}

// Main algorithm exposition function - for *internal* use by pygmo. The exposition function
// for APs needs to be different.
template <typename Algo>
static inline bp::class_<Algo> expose_algorithm_pygmo(const char *name, const char *descr)
{
    // We require all algorithms to be def-ctible at the bare minimum.
    bp::class_<Algo> c(name, descr, bp::init<>());

    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Get reference to the algorithm class.
    auto &algo = get_algorithm_class();

    // Expose the algorithm constructor from Algo.
    algo.def(bp::init<const Algo &>((bp::arg("uda"))));

    // Expose extract.
    algo.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>, bp::return_internal_reference<>());

    // Add the algorithm to the algorithms submodule.
    bp::scope().attr("algorithms").attr(name) = c;

    return c;
}

void expose_algorithms()
{
    // MBH meta-algo.
    auto mbh_ = expose_algorithm_pygmo<mbh>("mbh", mbh_docstring().c_str());
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
    add_property(mbh_, "inner_algorithm",
                 bp::make_function(lcast([](mbh &uda) -> algorithm & { return uda.get_inner_algorithm(); }),
                                   bp::return_internal_reference<>()),
                 generic_uda_inner_algorithm_docstring().c_str());
    // cstrs_self_adaptive meta-algo.
    auto cstrs_sa
        = expose_algorithm_pygmo<cstrs_self_adaptive>("cstrs_self_adaptive", cstrs_self_adaptive_docstring().c_str());
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
    add_property(
        cstrs_sa, "inner_algorithm",
        bp::make_function(lcast([](cstrs_self_adaptive &uda) -> algorithm & { return uda.get_inner_algorithm(); }),
                          bp::return_internal_reference<>()),
        generic_uda_inner_algorithm_docstring().c_str());

    // Test algo.
    auto test_a = expose_algorithm_pygmo<test_algorithm>("_test_algorithm", "A test algorithm.");
    test_a.def("get_n", &test_algorithm::get_n);
    test_a.def("set_n", &test_algorithm::set_n);
    // Thread unsafe test algo.
    expose_algorithm_pygmo<tu_test_algorithm>("_tu_test_algorithm", "A thread unsafe test algorithm.");
    // Null algo.
    auto na = expose_algorithm_pygmo<null_algorithm>("null_algorithm", null_algorithm_docstring().c_str());
    // ARTIFICIAL BEE COLONY
    auto bee_colony_ = expose_algorithm_pygmo<bee_colony>("bee_colony", bee_colony_docstring().c_str());
    bee_colony_.def(bp::init<unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("limit") = 1u)));
    bee_colony_.def(
        bp::init<unsigned, unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("limit") = 20u, bp::arg("seed"))));
    expose_algo_log(bee_colony_, bee_colony_get_log_docstring().c_str());
    bee_colony_.def("get_seed", &bee_colony::get_seed, generic_uda_get_seed_docstring().c_str());
    // DE
    auto de_ = expose_algorithm_pygmo<de>("de", de_docstring().c_str());
    de_.def(bp::init<unsigned, double, double, unsigned, double, double>(
        (bp::arg("gen") = 1u, bp::arg("F") = .8, bp::arg("CR") = .9, bp::arg("variant") = 2u, bp::arg("ftol") = 1e-6,
         bp::arg("tol") = 1E-6)));
    de_.def(bp::init<unsigned, double, double, unsigned, double, double, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("F") = .8, bp::arg("CR") = .9, bp::arg("variant") = 2u, bp::arg("ftol") = 1e-6,
         bp::arg("tol") = 1E-6, bp::arg("seed"))));
    expose_algo_log(de_, de_get_log_docstring().c_str());
    de_.def("get_seed", &de::get_seed, generic_uda_get_seed_docstring().c_str());
    // COMPASS SEARCH
    auto compass_search_ = expose_algorithm_pygmo<compass_search>("compass_search", compass_search_docstring().c_str());
    compass_search_.def(
        bp::init<unsigned, double, double, double>((bp::arg("max_fevals") = 1u, bp::arg("start_range") = .1,
                                                    bp::arg("stop_range") = .01, bp::arg("reduction_coeff") = .5)));
    expose_algo_log(compass_search_, compass_search_get_log_docstring().c_str());
    compass_search_.def("get_max_fevals", &compass_search::get_max_fevals);
    compass_search_.def("get_start_range", &compass_search::get_start_range);
    compass_search_.def("get_stop_range", &compass_search::get_stop_range);
    compass_search_.def("get_reduction_coeff", &compass_search::get_reduction_coeff);
    compass_search_.def("get_verbosity", &compass_search::get_verbosity);
    expose_not_population_based(compass_search_, "compass_search");
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
    // DE-1220
    auto de1220_ = expose_algorithm_pygmo<de1220>("de1220", de1220_docstring().c_str());
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
#if defined(PAGMO_WITH_EIGEN3)
    // CMA-ES
    auto cmaes_ = expose_algorithm_pygmo<cmaes>("cmaes", cmaes_docstring().c_str());
    cmaes_.def(bp::init<unsigned, double, double, double, double, double, double, double, bool, bool>(
        (bp::arg("gen") = 1u, bp::arg("cc") = -1., bp::arg("cs") = -1., bp::arg("c1") = -1., bp::arg("cmu") = -1.,
         bp::arg("sigma0") = 0.5, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false,
         bp::arg("force_bounds") = false)));
    cmaes_.def(bp::init<unsigned, double, double, double, double, double, double, double, bool, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("cc") = -1., bp::arg("cs") = -1., bp::arg("c1") = -1., bp::arg("cmu") = -1.,
         bp::arg("sigma0") = 0.5, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false,
         bp::arg("force_bounds") = false, bp::arg("seed"))));
    expose_algo_log(cmaes_, cmaes_get_log_docstring().c_str());
    cmaes_.def("get_seed", &cmaes::get_seed, generic_uda_get_seed_docstring().c_str());
    // xNES
    auto xnes_ = expose_algorithm_pygmo<xnes>("xnes", xnes_docstring().c_str());
    xnes_.def(bp::init<unsigned, double, double, double, double, double, double, bool, bool>(
        (bp::arg("gen") = 1u, bp::arg("eta_mu") = -1., bp::arg("eta_sigma") = -1., bp::arg("eta_b") = -1.,
         bp::arg("sigma0") = -1, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false,
         bp::arg("force_bounds") = false)));
    xnes_.def(bp::init<unsigned, double, double, double, double, double, double, bool, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("eta_mu") = -1., bp::arg("eta_sigma") = -1., bp::arg("eta_b") = -1.,
         bp::arg("sigma0") = -1, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false,
         bp::arg("force_bounds") = false, bp::arg("seed"))));
    expose_algo_log(xnes_, xnes_get_log_docstring().c_str());
    xnes_.def("get_seed", &xnes::get_seed, generic_uda_get_seed_docstring().c_str());
#endif
    // MOEA/D - DE
    auto moead_ = expose_algorithm_pygmo<moead>("moead", moead_docstring().c_str());
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
    add_property(
        nlopt_, "local_optimizer",
        bp::make_function(lcast([](nlopt &n) { return n.get_local_optimizer(); }), bp::return_internal_reference<>()),
        lcast([](nlopt &n, const nlopt *ptr) {
            if (ptr) {
                n.set_local_optimizer(*ptr);
            } else {
                n.unset_local_optimizer();
            }
        }),
        nlopt_local_optimizer_docstring().c_str());
#endif

#if defined(PAGMO_WITH_IPOPT)
    // Ipopt.
    auto ipopt_ = expose_algorithm_pygmo<ipopt>("ipopt", ipopt_docstring().c_str());
    expose_not_population_based(ipopt_, "ipopt");
    expose_algo_log(ipopt_, ipopt_get_log_docstring().c_str());
    ipopt_.def("get_last_opt_result", lcast([](const ipopt &ip) { return static_cast<int>(ip.get_last_opt_result()); }),
               ipopt_get_last_opt_result_docstring().c_str());
    // Options management.
    // String opts.
    ipopt_.def("set_string_option", &ipopt::set_string_option, ipopt_set_string_option_docstring().c_str(),
               (bp::arg("name"), bp::arg("value")));
    ipopt_.def("set_string_options", lcast([](ipopt &ip, const bp::dict &d) {
                   std::map<std::string, std::string> m;
                   bp::stl_input_iterator<std::string> begin(d), end;
                   for (; begin != end; ++begin) {
                       m[*begin] = bp::extract<std::string>(d[*begin])();
                   }
                   ip.set_string_options(m);
               }),
               ipopt_set_string_options_docstring().c_str(), bp::arg("opts"));
    ipopt_.def("get_string_options", lcast([](const ipopt &ip) -> bp::dict {
                   const auto opts = ip.get_string_options();
                   bp::dict retval;
                   for (const auto &p : opts) {
                       retval[p.first] = p.second;
                   }
                   return retval;
               }),
               ipopt_get_string_options_docstring().c_str());
    ipopt_.def("reset_string_options", &ipopt::reset_string_options, ipopt_reset_string_options_docstring().c_str());
    // Integer options.
    ipopt_.def("set_integer_option", &ipopt::set_integer_option, ipopt_set_integer_option_docstring().c_str(),
               (bp::arg("name"), bp::arg("value")));
    ipopt_.def("set_integer_options", lcast([](ipopt &ip, const bp::dict &d) {
                   std::map<std::string, Ipopt::Index> m;
                   bp::stl_input_iterator<std::string> begin(d), end;
                   for (; begin != end; ++begin) {
                       m[*begin] = bp::extract<Ipopt::Index>(d[*begin])();
                   }
                   ip.set_integer_options(m);
               }),
               ipopt_set_integer_options_docstring().c_str(), bp::arg("opts"));
    ipopt_.def("get_integer_options", lcast([](const ipopt &ip) -> bp::dict {
                   const auto opts = ip.get_integer_options();
                   bp::dict retval;
                   for (const auto &p : opts) {
                       retval[p.first] = p.second;
                   }
                   return retval;
               }),
               ipopt_get_integer_options_docstring().c_str());
    ipopt_.def("reset_integer_options", &ipopt::reset_integer_options, ipopt_reset_integer_options_docstring().c_str());
    // Numeric options.
    ipopt_.def("set_numeric_option", &ipopt::set_numeric_option, ipopt_set_numeric_option_docstring().c_str(),
               (bp::arg("name"), bp::arg("value")));
    ipopt_.def("set_numeric_options", lcast([](ipopt &ip, const bp::dict &d) {
                   std::map<std::string, double> m;
                   bp::stl_input_iterator<std::string> begin(d), end;
                   for (; begin != end; ++begin) {
                       m[*begin] = bp::extract<double>(d[*begin])();
                   }
                   ip.set_numeric_options(m);
               }),
               ipopt_set_numeric_options_docstring().c_str(), bp::arg("opts"));
    ipopt_.def("get_numeric_options", lcast([](const ipopt &ip) -> bp::dict {
                   const auto opts = ip.get_numeric_options();
                   bp::dict retval;
                   for (const auto &p : opts) {
                       retval[p.first] = p.second;
                   }
                   return retval;
               }),
               ipopt_get_numeric_options_docstring().c_str());
    ipopt_.def("reset_numeric_options", &ipopt::reset_numeric_options, ipopt_reset_numeric_options_docstring().c_str());
#endif
}
}
