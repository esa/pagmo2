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
#pragma warning(disable : 4244)

#endif

#include <pygmo/python_includes.hpp>

// See: https://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// In every cpp file We need to make sure this is included before everything else,
// with the correct #defines.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include <pygmo/numpy.hpp>

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/problems/ackley.hpp>
#include <pagmo/problems/cec2006.hpp>
#include <pagmo/problems/cec2009.hpp>
#if !defined(_MSC_VER)
#include <pagmo/problems/cec2013.hpp>
#endif
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/griewank.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/luksan_vlcek1.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_utils.hpp>
#include <pygmo/docstrings.hpp>
#include <pygmo/problem_exposition_suite.hpp>
#include <pygmo/pygmo_classes.hpp>

using namespace pagmo;
namespace bp = boost::python;

namespace pygmo
{

// A test problem.
struct test_problem {
    test_problem(unsigned nobj = 1) : m_nobj(nobj)
    {
    }
    vector_double fitness(const vector_double &) const
    {
        return {1.};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
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
    vector_double::size_type get_nobj() const
    {
        return m_nobj;
    }
    int m_n = 1;
    unsigned m_nobj;
};

// A thread-unsafe test problem.
struct tu_test_problem {
    vector_double fitness(const vector_double &) const
    {
        return {1.};
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }
};

// C++ UDP exposition function - for *internal* pygmo use. This needs to be different
// from the exposition function used for APs.
template <typename Prob>
static inline bp::class_<Prob> expose_problem_pygmo(const char *name, const char *descr)
{
    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name, descr, bp::init<>());

    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Get reference to the problem class.
    auto &prob = get_problem_class();

    // Expose the problem constructor from Prob.
    prob.def(bp::init<const Prob &>((bp::arg("udp"))));

    // Expose extract.
    prob.def("_cpp_extract", &generic_cpp_extract<pagmo::problem, Prob>, bp::return_internal_reference<>());

    // Add the problem to the problems submodule.
    bp::scope().attr("problems").attr(name) = c;

    return c;
}

void expose_problems()
{
    // Exposition of C++ problems.
    // Test problem.
    auto test_p = expose_problem_pygmo<test_problem>("_test_problem", "A test problem.");
    test_p.def(bp::init<unsigned>((bp::arg("nobj"))));
    test_p.def("get_n", &test_problem::get_n);
    test_p.def("set_n", &test_problem::set_n);
    // Thread unsafe test problem.
    expose_problem_pygmo<tu_test_problem>("_tu_test_problem", "A thread unsafe test problem.");
    // Null problem.
    auto np = expose_problem_pygmo<null_problem>("null_problem", null_problem_docstring().c_str());
    np.def(bp::init<vector_double::size_type, vector_double::size_type, vector_double::size_type>(
        (bp::arg("nobj") = 1, bp::arg("nec") = 0, bp::arg("nic") = 0)));
    // Rosenbrock.
    auto rb = expose_problem_pygmo<rosenbrock>("rosenbrock", rosenbrock_docstring().c_str());
    rb.def(bp::init<vector_double::size_type>((bp::arg("dim"))));
    rb.def("best_known", &best_known_wrapper<rosenbrock>, problem_get_best_docstring("Rosenbrock").c_str());
    // Hock-Schittkowsky 71
    auto hs71 = expose_problem_pygmo<hock_schittkowsky_71>("hock_schittkowsky_71",
                                                           "__init__()\n\nThe Hock-Schittkowsky 71 problem.\n\n"
                                                           "See :cpp:class:`pagmo::hock_schittkowsky_71`.\n\n");
    hs71.def("best_known", &best_known_wrapper<hock_schittkowsky_71>,
             problem_get_best_docstring("Hock-Schittkowsky 71").c_str());
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
    // Ackley.
    auto ack = expose_problem_pygmo<ackley>("ackley", "__init__(dim = 1)\n\nThe Ackley problem.\n\n"
                                                      "See :cpp:class:`pagmo::ackley`.\n\n");
    ack.def(bp::init<unsigned>((bp::arg("dim"))));
    ack.def("best_known", &best_known_wrapper<ackley>, problem_get_best_docstring("Ackley").c_str());
    // Griewank.
    auto griew = expose_problem_pygmo<griewank>("griewank", "__init__(dim = 1)\n\nThe Griewank problem.\n\n"
                                                            "See :cpp:class:`pagmo::griewank`.\n\n");
    griew.def(bp::init<unsigned>((bp::arg("dim"))));
    griew.def("best_known", &best_known_wrapper<griewank>, problem_get_best_docstring("Griewank").c_str());
    // ZDT.
    auto zdt_p = expose_problem_pygmo<zdt>("zdt", "__init__(prob_id = 1, param = 30)\n\nThe ZDT problem.\n\n"
                                                  "See :cpp:class:`pagmo::zdt`.\n\n");
    zdt_p.def(bp::init<unsigned, unsigned>((bp::arg("prob_id") = 1u, bp::arg("param") = 30u)));
    zdt_p.def("p_distance", lcast([](const zdt &z, const bp::object &x) { return z.p_distance(to_vd(x)); }));
    zdt_p.def("p_distance", lcast([](const zdt &z, const population &pop) { return z.p_distance(pop); }),
              zdt_p_distance_docstring().c_str());
    // DTLZ.
    auto dtlz_p = expose_problem_pygmo<dtlz>("dtlz", dtlz_docstring().c_str());
    dtlz_p.def(bp::init<unsigned, unsigned, unsigned, unsigned>(
        (bp::arg("prob_id") = 1u, bp::arg("dim") = 5u, bp::arg("fdim") = 3u, bp::arg("alpha") = 100u)));
    dtlz_p.def("p_distance", lcast([](const dtlz &z, const bp::object &x) { return z.p_distance(to_vd(x)); }));
    dtlz_p.def("p_distance", lcast([](const dtlz &z, const population &pop) { return z.p_distance(pop); }),
               dtlz_p_distance_docstring().c_str());
    // Inventory.
    auto inv = expose_problem_pygmo<inventory>(
        "inventory", "__init__(weeks = 4,sample_size = 10,seed = random)\n\nThe inventory problem.\n\n"
                     "See :cpp:class:`pagmo::inventory`.\n\n");
    inv.def(bp::init<unsigned, unsigned>((bp::arg("weeks") = 4u, bp::arg("sample_size") = 10u)));
    inv.def(
        bp::init<unsigned, unsigned, unsigned>((bp::arg("weeks") = 4u, bp::arg("sample_size") = 10u, bp::arg("seed"))));
// excluded in MSVC (Dec. - 2016) because of troubles to deal with the big static array defining the problem data. To be
// reassesed in future versions of the compiler
#if !defined(_MSC_VER)
    // CEC 2013.
    auto cec2013_ = expose_problem_pygmo<cec2013>("cec2013", cec2013_docstring().c_str());
    cec2013_.def(bp::init<unsigned, unsigned>((bp::arg("prob_id") = 1, bp::arg("dim") = 2)));
#endif

    // CEC 2006
    auto cec2006_ = expose_problem_pygmo<cec2006>("cec2006", cec2006_docstring().c_str());
    cec2006_.def(bp::init<unsigned>((bp::arg("prob_id"))));
    cec2006_.def("best_known", &best_known_wrapper<cec2006>, problem_get_best_docstring("CEC 2006").c_str());

    // CEC 2009
    auto cec2009_ = expose_problem_pygmo<cec2009>("cec2009", cec2009_docstring().c_str());
    cec2009_.def(bp::init<unsigned, bool, unsigned>(
        (bp::arg("prob_id") = 1u, bp::arg("is_constrained") = false, bp::arg("dim") = 30u)));

    // Luksan Vlcek 1
    auto lv_ = expose_problem_pygmo<luksan_vlcek1>("luksan_vlcek1", luksan_vlcek1_docstring().c_str());
    lv_.def(bp::init<unsigned>(bp::arg("dim")));

    // Translate meta-problem
    auto translate_ = expose_problem_pygmo<translate>("translate", translate_docstring().c_str());
    // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
    // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
    translate_.def("__init__", bp::make_constructor(lcast([](const problem &p, const bp::object &tv) {
                                                        return ::new pagmo::translate(p, to_vd(tv));
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
                                         return ::new pagmo::unconstrain(p, method, to_vd(weights));
                                     }),
                                     bp::default_call_policies()));
    add_property(unconstrain_, "inner_problem",
                 bp::make_function(lcast([](unconstrain &udp) -> problem & { return udp.get_inner_problem(); }),
                                   bp::return_internal_reference<>()),
                 generic_udp_inner_problem_docstring().c_str());
    // Decompose meta-problem.
    auto decompose_ = expose_problem_pygmo<decompose>("decompose", decompose_docstring().c_str());
    // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
    // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
    decompose_.def("__init__", bp::make_constructor(
                                   lcast([](const problem &p, const bp::object &weight, const bp::object &z,
                                            const std::string &method, bool adapt_ideal) {
                                       return ::new pagmo::decompose(p, to_vd(weight), to_vd(z), method, adapt_ideal);
                                   }),
                                   bp::default_call_policies()));
    decompose_.def("original_fitness", lcast([](const pagmo::decompose &p, const bp::object &x) {
                       return v_to_a(p.original_fitness(to_vd(x)));
                   }),
                   decompose_original_fitness_docstring().c_str(), (bp::arg("x")));
    add_property(decompose_, "z", lcast([](const pagmo::decompose &p) { return v_to_a(p.get_z()); }),
                 decompose_z_docstring().c_str());
    add_property(decompose_, "inner_problem",
                 bp::make_function(lcast([](decompose &udp) -> problem & { return udp.get_inner_problem(); }),
                                   bp::return_internal_reference<>()),
                 generic_udp_inner_problem_docstring().c_str());
}
}
