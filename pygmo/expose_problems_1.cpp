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

#include <boost/python/args.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>

#include <pagmo/config.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#if defined(PAGMO_ENABLE_CEC2013)
#include <pagmo/problems/cec2013.hpp>
#endif
#include <pagmo/problems/golomb_ruler.hpp>
#include <pagmo/problems/luksan_vlcek1.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/problems/wfg.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/docstrings.hpp>
#include <pygmo/expose_problems.hpp>
#include <pygmo/problem_exposition_suite.hpp>

using namespace pagmo;
namespace bp = boost::python;

namespace pygmo
{

void expose_problems_1()
{
    // Exposition of C++ problems.
    // Rosenbrock.
    auto rb = expose_problem_pygmo<rosenbrock>("rosenbrock", rosenbrock_docstring().c_str());
    rb.def(bp::init<vector_double::size_type>((bp::arg("dim"))));
    rb.def("best_known", &best_known_wrapper<rosenbrock>, problem_get_best_docstring("Rosenbrock").c_str());

    // MINLP-Rastrigin.
    auto minlp_rastr = expose_problem_pygmo<minlp_rastrigin>("minlp_rastrigin", minlp_rastrigin_docstring().c_str());
    minlp_rastr.def(bp::init<unsigned, unsigned>((bp::arg("dim_c") = 1u, bp::arg("dim_i") = 1u)));

    // Rastrigin.
    auto rastr = expose_problem_pygmo<rastrigin>("rastrigin", "__init__(dim = 1)\n\nThe Rastrigin problem.\n\n"
                                                              "See :cpp:class:`pagmo::rastrigin`.\n\n");
    rastr.def(bp::init<unsigned>((bp::arg("dim") = 1)));
    rastr.def("best_known", &best_known_wrapper<rastrigin>, problem_get_best_docstring("Rastrigin").c_str());
    // Schwefel.
    auto sch = expose_problem_pygmo<schwefel>("schwefel", "__init__(dim = 1)\n\nThe Schwefel problem.\n\n"
                                                          "See :cpp:class:`pagmo::schwefel`.\n\n");
    sch.def(bp::init<unsigned>((bp::arg("dim"))));
    sch.def("best_known", &best_known_wrapper<schwefel>, problem_get_best_docstring("Schwefel").c_str());
    // ZDT.
    auto zdt_p = expose_problem_pygmo<zdt>("zdt", "__init__(prob_id = 1, param = 30)\n\nThe ZDT problem.\n\n"
                                                  "See :cpp:class:`pagmo::zdt`.\n\n");
    zdt_p.def(bp::init<unsigned, unsigned>((bp::arg("prob_id") = 1u, bp::arg("param") = 30u)));
    zdt_p.def("p_distance", lcast([](const zdt &z, const bp::object &x) { return z.p_distance(to_vd(x)); }));
    zdt_p.def("p_distance", lcast([](const zdt &z, const population &pop) { return z.p_distance(pop); }),
              zdt_p_distance_docstring().c_str());

    // Golomb Ruler
    auto gr = expose_problem_pygmo<golomb_ruler>("golomb_ruler",
                                                 "__init__(order, upper_bound)\n\nThe Golomb Ruler Problem.\n\n"
                                                 "See :cpp:class:`pagmo::golomb_ruler`.\n\n");
    gr.def(bp::init<unsigned, unsigned>((bp::arg("order"), bp::arg("upper_bound"))));

#if defined(PAGMO_ENABLE_CEC2013)
    // See the explanation in pagmo/config.hpp.
    auto cec2013_ = expose_problem_pygmo<cec2013>("cec2013", cec2013_docstring().c_str());
    cec2013_.def(bp::init<unsigned, unsigned>((bp::arg("prob_id") = 1, bp::arg("dim") = 2)));
#endif

    // Luksan Vlcek 1
    auto lv_ = expose_problem_pygmo<luksan_vlcek1>("luksan_vlcek1", luksan_vlcek1_docstring().c_str());
    lv_.def(bp::init<unsigned>(bp::arg("dim")));

    // Translate meta-problem
    auto translate_ = expose_problem_pygmo<translate>("translate", translate_docstring().c_str());
    // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
    // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
    translate_.def("__init__", bp::make_constructor(lcast([](const problem &p, const bp::object &tv) {
                                                        return ::new translate(p, to_vd(tv));
                                                    }),
                                                    bp::default_call_policies()));
    add_property(translate_, "translation", lcast([](const translate &t) { return v_to_a(t.get_translation()); }),
                 translate_translation_docstring().c_str());
    add_property(translate_, "inner_problem",
                 bp::make_function(lcast([](translate &udp) -> problem & { return udp.get_inner_problem(); }),
                                   bp::return_internal_reference<>()),
                 generic_udp_inner_problem_docstring().c_str());
    // Unconstrain meta-problem.
    auto unconstrain_ = expose_problem_pygmo<unconstrain>("unconstrain", unconstrain_docstring().c_str());
    // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
    // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
    unconstrain_.def("__init__", bp::make_constructor(
                                     lcast([](const problem &p, const std::string &method, const bp::object &weights) {
                                         return ::new unconstrain(p, method, to_vd(weights));
                                     }),
                                     bp::default_call_policies()));
    add_property(unconstrain_, "inner_problem",
                 bp::make_function(lcast([](unconstrain &udp) -> problem & { return udp.get_inner_problem(); }),
                                   bp::return_internal_reference<>()),
                 generic_udp_inner_problem_docstring().c_str());
    // WFG.
    auto wfg_p = expose_problem_pygmo<wfg>("wfg", wfg_docstring().c_str());
    wfg_p.def(bp::init<unsigned, vector_double::size_type, vector_double::size_type, vector_double::size_type>(
        (bp::arg("prob_id") = 1u, bp::arg("dim_dvs") = 5u, bp::arg("dim_obj") = 3u, bp::arg("dim_k") = 4u)));
}
} // namespace pygmo
