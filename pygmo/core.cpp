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
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include "numpy.hpp"

#if defined(_MSC_VER)

// Disable various warnings from MSVC.
#pragma warning(push, 0)
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)

#endif

#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/docstring_options.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/self.hpp>
#include <boost/python/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>
#include <pagmo/utils/hv_algos/hv_bf_approx.hpp>
#include <pagmo/utils/hv_algos/hv_bf_fpras.hpp>
#include <pagmo/utils/hv_algos/hv_hv2d.hpp>
#include <pagmo/utils/hv_algos/hv_hv3d.hpp>
#include <pagmo/utils/hv_algos/hv_hvwfg.hpp>
#include <pagmo/utils/hypervolume.hpp>

#include "algorithm.hpp"
#include "algorithm_exposition_suite.hpp"
#include "common_utils.hpp"
#include "docstrings.hpp"
#include "expose_algorithms.hpp"
#include "expose_problems.hpp"
#include "island.hpp"
#include "island_exposition_suite.hpp"
#include "object_serialization.hpp"
#include "problem.hpp"
#include "pygmo_classes.hpp"

#if defined(_MSC_VER)

#pragma warning(pop)

#endif

// This is necessary because the NumPy macro import_array() has different return values
// depending on the Python version.
#if PY_MAJOR_VERSION < 3
static inline void wrap_import_array()
{
    import_array();
}
#else
static inline void *wrap_import_array()
{
    import_array();
    return nullptr;
}
#endif

namespace bp = boost::python;
using namespace pagmo;

// Test that the cereal serialization of BP objects works as expected.
// The object returned by this function should be identical to the input
// object.
static inline bp::object test_object_serialization(const bp::object &o)
{
    std::ostringstream oss;
    {
        cereal::PortableBinaryOutputArchive oarchive(oss);
        oarchive(o);
    }
    const std::string tmp = oss.str();
    std::istringstream iss;
    iss.str(tmp);
    bp::object retval;
    {
        cereal::PortableBinaryInputArchive iarchive(iss);
        iarchive(retval);
    }
    return retval;
}

// Instances of the classes in pygmo_classes.hpp.
namespace pygmo
{

// Problem and meta-problem classes.
std::unique_ptr<bp::class_<problem>> problem_ptr{};

// Algorithm and meta-algorithm classes.
std::unique_ptr<bp::class_<algorithm>> algorithm_ptr{};

// Island.
std::unique_ptr<bp::class_<island>> island_ptr{};
}

// The cleanup function.
// This function will be registered to be called when the pygmo core module is unloaded
// (see the __init__.py file). I am not 100% sure it is needed to reset these global
// variables, but it makes me nervous to have global boost python objects around on shutdown.
// NOTE: probably it would be better to register the cleanup function directly in core.cpp,
// to be executed when the compiled module gets unloaded (now it is executed when the pygmo
// supermodule gets unloaded).
static inline void cleanup()
{
    pygmo::problem_ptr.reset();

    pygmo::algorithm_ptr.reset();

    pygmo::island_ptr.reset();
}

// Serialization support for the population class.
struct population_pickle_suite : bp::pickle_suite {
    static bp::tuple getinitargs(const population &)
    {
        return bp::make_tuple();
    }
    static bp::tuple getstate(const population &pop)
    {
        std::ostringstream oss;
        {
            cereal::PortableBinaryOutputArchive oarchive(oss);
            oarchive(pop);
        }
        auto s = oss.str();
        return bp::make_tuple(pygmo::make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())));
    }
    static void setstate(population &pop, bp::tuple state)
    {
        if (len(state) != 1) {
            pygmo_throw(PyExc_ValueError, "the state tuple must have a single element");
        }
        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize a population");
        }
        const auto size = len(state[0]);
        std::string s(ptr, ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
            cereal::PortableBinaryInputArchive iarchive(iss);
            iarchive(pop);
        }
    }
};

// Serialization support for the archi class.
struct archipelago_pickle_suite : bp::pickle_suite {
    static bp::tuple getinitargs(const archipelago &)
    {
        return bp::make_tuple();
    }
    static bp::tuple getstate(const archipelago &archi)
    {
        std::ostringstream oss;
        {
            cereal::PortableBinaryOutputArchive oarchive(oss);
            oarchive(archi);
        }
        auto s = oss.str();
        return bp::make_tuple(pygmo::make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())));
    }
    static void setstate(archipelago &archi, bp::tuple state)
    {
        if (len(state) != 1) {
            pygmo_throw(PyExc_ValueError, "the state tuple must have a single element");
        }
        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize an archipelago");
        }
        const auto size = len(state[0]);
        std::string s(ptr, ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
            cereal::PortableBinaryInputArchive iarchive(iss);
            iarchive(archi);
        }
    }
};

// Helper function to test the to_vd functionality.
static inline bool test_to_vd(const bp::object &o, unsigned n)
{
    auto res = pygmo::to_vd(o);
    if (res.size() != n) {
        return false;
    }
    for (decltype(res.size()) i = 0; i < res.size(); ++i) {
        if (res[i] != static_cast<double>(i)) {
            return false;
        }
    }
    return true;
}

// Helper function to test the to_vvd functionality.
static inline bool test_to_vvd(const bp::object &o, unsigned n, unsigned m)
{
    auto res = pygmo::to_vvd(o);
    return res.size() == n
           && std::all_of(res.begin(), res.end(), [m](const vector_double &v) { return v.size() == m; });
}

// NOTE: we need to provide a custom raii waiter in the island. The reason is the following.
// Boost.Python locks the GIL when crossing the boundary from Python into C++. So, if we call wait() from Python,
// BP will lock the GIL and then we will be waiting for evolutions in the island to finish. During this time, no
// Python code will be executed because the GIL is locked. This means that if we have a Python thread doing background
// work (e.g., managing the task queue in pythonic islands), it will have to wait before doing any progress. By
// unlocking the GIL before calling thread_island::wait(), we give the chance to other Python threads to continue
// doing some work.
// NOTE: here we have 2 RAII classes interacting with the GIL. The GIL releaser is the *second* one,
// and it is the one that is responsible for unlocking the Python interpreter while wait() is running.
// The *first* one, the GIL thread ensurer, does something else: it makes sure that we can call the Python
// interpreter from the current C++ thread. In a normal situation, in which islands are just instantiated
// from the main thread, the gte object is superfluous. However, if we are interacting with islands from a
// separate C++ thread, then we need to make sure that every time we call into the Python interpreter (e.g., by
// using the GIL releaser below) we inform Python we are about to call from a separate thread. This is what
// the GTE object does. This use case is, for instance, what happens with the PADE algorithm when, algo, prob,
// etc. are all C++ objects (when at least one object is pythonic, we will not end up using the thread island).
// NOTE: by ordering the class members in this way we ensure that gte is constructed before gr, which is essential
// (otherwise we might be calling into the interpreter with a releaser before informing Python we are calling
// from a separate thread).
struct py_wait_locks {
    pygmo::gil_thread_ensurer gte;
    pygmo::gil_releaser gr;
};

BOOST_PYTHON_MODULE(core)
{
    using pygmo::lcast;

    // This function needs to be called before doing anything with threads.
    // https://docs.python.org/3/c-api/init.html
    ::PyEval_InitThreads();

    // Init numpy.
    // NOTE: only the second import is strictly necessary. We run a first import from BP
    // because that is the easiest way to detect whether numpy is installed or not (rather
    // than trying to figure out a way to detect it from wrap_import_array()).
    try {
        bp::import("numpy.core.multiarray");
    } catch (...) {
        pygmo::builtin().attr("print")(
            u8"\033[91m====ERROR====\nThe NumPy module could not be imported. "
            u8"Please make sure that NumPy has been correctly installed.\n====ERROR====\033[0m");
        pygmo_throw(PyExc_ImportError, "");
    }
    wrap_import_array();

    // Check that dill is available.
    try {
        bp::import("dill");
    } catch (...) {
        pygmo::builtin().attr("print")(
            u8"\033[91m====ERROR====\nThe dill module could not be imported. "
            u8"Please make sure that dill has been correctly installed.\n====ERROR====\033[0m");
        pygmo_throw(PyExc_ImportError, "");
    }

    // Override the default implementation of the island factory.
    detail::island_factory<>::s_func = [](const algorithm &algo, const population &pop,
                                          std::unique_ptr<detail::isl_inner_base> &ptr) {
        if (static_cast<int>(algo.get_thread_safety()) >= static_cast<int>(thread_safety::basic)
            && static_cast<int>(pop.get_problem().get_thread_safety()) >= static_cast<int>(thread_safety::basic)) {
            // Both algo and prob have at least the basic thread safety guarantee. Use the thread island.
            ptr = detail::make_unique<detail::isl_inner<thread_island>>();
        } else {
            // NOTE: here we are re-implementing a piece of code that normally
            // is pure C++. We are calling into the Python interpreter, so, in order to handle
            // the case in which we are invoking this code from a separate C++ thread, we construct a GIL ensurer
            // in order to guard against concurrent access to the interpreter. The idea here is that this piece
            // of code normally would provide a basic thread safety guarantee, and in order to continue providing
            // it we use the ensurer.
            pygmo::gil_thread_ensurer gte;
            bp::object py_island = bp::import("pygmo")
#if defined(_WIN32) || PY_MAJOR_VERSION > 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 4)
                                       // NOTE: the mp_island is supported since Python 3.4 or on Windows.
                                       .attr("mp_island");
#else
                                       .attr("ipyparallel_island");
#endif
            ptr = detail::make_unique<detail::isl_inner<bp::object>>(py_island());
        }
    };

    // Override the default RAII waiter. We need to use shared_ptr because we don't want to move/copy/destroy
    // the locks when invoking this from island::wait(), we need to instaniate exactly 1 py_wait_lock and have it
    // destroyed at the end of island::wait().
    detail::wait_raii<>::getter = []() { return std::make_shared<py_wait_locks>(); };

    // Setup doc options
    bp::docstring_options doc_options;
    doc_options.enable_all();
    doc_options.disable_cpp_signatures();
    doc_options.disable_py_signatures();

    // The thread_safety enum.
    bp::enum_<thread_safety>("_thread_safety").value("none", thread_safety::none).value("basic", thread_safety::basic);

    // Expose utility functions for testing purposes.
    bp::def("_builtin", &pygmo::builtin);
    bp::def("_type", &pygmo::type);
    bp::def("_str", &pygmo::str);
    bp::def("_callable", &pygmo::callable);
    bp::def("_deepcopy", &pygmo::deepcopy);
    bp::def("_to_sp", &pygmo::to_sp);
    bp::def("_test_object_serialization", &test_object_serialization);
    bp::def("_test_to_vd", &test_to_vd);
    bp::def("_test_to_vvd", &test_to_vvd);

    // Expose cleanup function.
    bp::def("_cleanup", &cleanup);

    // Create the problems submodule.
    std::string problems_module_name = bp::extract<std::string>(bp::scope().attr("__name__") + ".problems");
    PyObject *problems_module_ptr = PyImport_AddModule(problems_module_name.c_str());
    if (!problems_module_ptr) {
        pygmo_throw(PyExc_RuntimeError, "error while creating the 'problems' submodule");
    }
    auto problems_module = bp::object(bp::handle<>(bp::borrowed(problems_module_ptr)));
    bp::scope().attr("problems") = problems_module;

    // Create the algorithms submodule.
    std::string algorithms_module_name = bp::extract<std::string>(bp::scope().attr("__name__") + ".algorithms");
    PyObject *algorithms_module_ptr = PyImport_AddModule(algorithms_module_name.c_str());
    if (!algorithms_module_ptr) {
        pygmo_throw(PyExc_RuntimeError, "error while creating the 'algorithms' submodule");
    }
    auto algorithms_module = bp::object(bp::handle<>(bp::borrowed(algorithms_module_ptr)));
    bp::scope().attr("algorithms") = algorithms_module;

    // Create the islands submodule.
    std::string islands_module_name = bp::extract<std::string>(bp::scope().attr("__name__") + ".islands");
    PyObject *islands_module_ptr = PyImport_AddModule(islands_module_name.c_str());
    if (!islands_module_ptr) {
        pygmo_throw(PyExc_RuntimeError, "error while creating the 'islands' submodule");
    }
    auto islands_module = bp::object(bp::handle<>(bp::borrowed(islands_module_ptr)));
    bp::scope().attr("islands") = islands_module;

    // Population class.
    bp::class_<population> pop_class("population", pygmo::population_docstring().c_str(), bp::no_init);
    // Ctors from problem.
    // NOTE: we expose only the ctors from pagmo::problem, not from C++ or Python UDPs. An __init__ wrapper
    // on the Python side will take care of cting a pagmo::problem from the input UDP, and then invoke this ctor.
    // This way we avoid having to expose a different ctor for every exposed C++ prob.
    pop_class.def(bp::init<const problem &, population::size_type>())
        .def(bp::init<const problem &, population::size_type, unsigned>())
        // Repr.
        .def(repr(bp::self))
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<population>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<population>)
        .def_pickle(population_pickle_suite())
        .def("push_back", lcast([](population &pop, const bp::object &x, const bp::object &f) {
                 if (f.is_none()) {
                     pop.push_back(pygmo::to_vd(x));
                 } else {
                     pop.push_back(pygmo::to_vd(x), pygmo::to_vd(f));
                 }
             }),
             pygmo::population_push_back_docstring().c_str(), (bp::arg("x"), bp::arg("f") = bp::object()))
        .def("random_decision_vector",
             lcast([](const population &pop) { return pygmo::v_to_a(pop.random_decision_vector()); }),
             pygmo::population_random_decision_vector_docstring().c_str())
        .def("best_idx",
             lcast([](const population &pop, const bp::object &tol) { return pop.best_idx(pygmo::to_vd(tol)); }),
             (bp::arg("tol")))
        .def("best_idx", lcast([](const population &pop, double tol) { return pop.best_idx(tol); }), (bp::arg("tol")))
        .def("best_idx", lcast([](const population &pop) { return pop.best_idx(); }),
             pygmo::population_best_idx_docstring().c_str())
        .def("worst_idx",
             lcast([](const population &pop, const bp::object &tol) { return pop.worst_idx(pygmo::to_vd(tol)); }),
             (bp::arg("tol")))
        .def("worst_idx", lcast([](const population &pop, double tol) { return pop.worst_idx(tol); }), (bp::arg("tol")))
        .def("worst_idx", lcast([](const population &pop) { return pop.worst_idx(); }),
             pygmo::population_worst_idx_docstring().c_str())
        .add_property("champion_x", lcast([](const population &pop) { return pygmo::v_to_a(pop.champion_x()); }),
                      pygmo::population_champion_x_docstring().c_str())
        .add_property("champion_f", lcast([](const population &pop) { return pygmo::v_to_a(pop.champion_f()); }),
                      pygmo::population_champion_f_docstring().c_str())
        .def("__len__", &population::size)
        .def("set_xf", lcast([](population &pop, population::size_type i, const bp::object &x, const bp::object &f) {
                 pop.set_xf(i, pygmo::to_vd(x), pygmo::to_vd(f));
             }),
             pygmo::population_set_xf_docstring().c_str())
        .def("set_x", lcast([](population &pop, population::size_type i, const bp::object &x) {
                 pop.set_x(i, pygmo::to_vd(x));
             }),
             pygmo::population_set_x_docstring().c_str())
        .add_property("problem",
                      bp::make_function(lcast([](population &pop) -> problem & { return pop.get_problem(); }),
                                        bp::return_internal_reference<>()),
                      lcast([](population &pop, const problem &p) { pop.get_problem() = p; }),
                      pygmo::population_problem_docstring().c_str())
        .def("get_f", lcast([](const population &pop) { return pygmo::vv_to_a(pop.get_f()); }),
             pygmo::population_get_f_docstring().c_str())
        .def("get_x", lcast([](const population &pop) { return pygmo::vv_to_a(pop.get_x()); }),
             pygmo::population_get_x_docstring().c_str())
        .def("get_ID", lcast([](const population &pop) { return pygmo::v_to_a(pop.get_ID()); }),
             pygmo::population_get_ID_docstring().c_str())
        .def("get_seed", &population::get_seed, pygmo::population_get_seed_docstring().c_str());

    // Problem class.
    pygmo::problem_ptr
        = detail::make_unique<bp::class_<problem>>("problem", pygmo::problem_docstring().c_str(), bp::init<>());
    auto &problem_class = *pygmo::problem_ptr;
    problem_class.def(bp::init<const bp::object &>((bp::arg("udp"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::problem_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<problem>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<problem>)
        // Problem extraction.
        .def("_py_extract", &pygmo::generic_py_extract<problem>)
        // Problem methods.
        .def("fitness", lcast([](const pagmo::problem &p, const bp::object &dv) {
                 return pygmo::v_to_a(p.fitness(pygmo::to_vd(dv)));
             }),
             pygmo::problem_fitness_docstring().c_str(), (bp::arg("dv")))
        .def("get_bounds", lcast([](const pagmo::problem &p) -> bp::tuple {
                 auto retval = p.get_bounds();
                 return bp::make_tuple(pygmo::v_to_a(retval.first), pygmo::v_to_a(retval.second));
             }),
             pygmo::problem_get_bounds_docstring().c_str())
        .def("gradient", lcast([](const pagmo::problem &p, const bp::object &dv) {
                 return pygmo::v_to_a(p.gradient(pygmo::to_vd(dv)));
             }),
             pygmo::problem_gradient_docstring().c_str(), (bp::arg("dv")))
        .def("has_gradient", &problem::has_gradient, pygmo::problem_has_gradient_docstring().c_str())
        .def("gradient_sparsity", lcast([](const pagmo::problem &p) { return pygmo::sp_to_a(p.gradient_sparsity()); }),
             pygmo::problem_gradient_sparsity_docstring().c_str())
        .def("has_gradient_sparsity", &problem::has_gradient_sparsity,
             pygmo::problem_has_gradient_sparsity_docstring().c_str())
        .def("hessians", lcast([](const pagmo::problem &p, const bp::object &dv) -> bp::list {
                 bp::list retval;
                 const auto h = p.hessians(pygmo::to_vd(dv));
                 for (const auto &v : h) {
                     retval.append(pygmo::v_to_a(v));
                 }
                 return retval;
             }),
             pygmo::problem_hessians_docstring().c_str(), (bp::arg("dv")))
        .def("has_hessians", &problem::has_hessians, pygmo::problem_has_hessians_docstring().c_str())
        .def("hessians_sparsity", lcast([](const pagmo::problem &p) -> bp::list {
                 bp::list retval;
                 const auto hs = p.hessians_sparsity();
                 for (const auto &sp : hs) {
                     retval.append(pygmo::sp_to_a(sp));
                 }
                 return retval;
             }),
             pygmo::problem_hessians_sparsity_docstring().c_str())
        .def("has_hessians_sparsity", &problem::has_hessians_sparsity,
             pygmo::problem_has_hessians_sparsity_docstring().c_str())
        .def("get_nobj", &problem::get_nobj, pygmo::problem_get_nobj_docstring().c_str())
        .def("get_nx", &problem::get_nx, pygmo::problem_get_nx_docstring().c_str())
        .def("get_nf", &problem::get_nf, pygmo::problem_get_nf_docstring().c_str())
        .def("get_nec", &problem::get_nec, pygmo::problem_get_nec_docstring().c_str())
        .def("get_nic", &problem::get_nic, pygmo::problem_get_nic_docstring().c_str())
        .def("get_nc", &problem::get_nc, pygmo::problem_get_nc_docstring().c_str())
        .add_property("c_tol", lcast([](const problem &prob) { return pygmo::v_to_a(prob.get_c_tol()); }),
                      lcast([](problem &prob, const bp::object &c_tol) { prob.set_c_tol(pygmo::to_vd(c_tol)); }),
                      pygmo::problem_c_tol_docstring().c_str())
        .def("get_fevals", &problem::get_fevals, pygmo::problem_get_fevals_docstring().c_str())
        .def("get_gevals", &problem::get_gevals, pygmo::problem_get_gevals_docstring().c_str())
        .def("get_hevals", &problem::get_hevals, pygmo::problem_get_hevals_docstring().c_str())
        .def("set_seed", &problem::set_seed, pygmo::problem_set_seed_docstring().c_str(), (bp::arg("seed")))
        .def("has_set_seed", &problem::has_set_seed, pygmo::problem_has_set_seed_docstring().c_str())
        .def("is_stochastic", &problem::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.core.problem.has_set_seed()`.\n")
        .def("feasibility_x",
             lcast([](const problem &p, const bp::object &x) { return p.feasibility_x(pygmo::to_vd(x)); }),
             pygmo::problem_feasibility_x_docstring().c_str(), (bp::arg("x")))
        .def("feasibility_f",
             lcast([](const problem &p, const bp::object &f) { return p.feasibility_f(pygmo::to_vd(f)); }),
             pygmo::problem_feasibility_f_docstring().c_str(), (bp::arg("f")))
        .def("get_name", &problem::get_name, pygmo::problem_get_name_docstring().c_str())
        .def("get_extra_info", &problem::get_extra_info, pygmo::problem_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &problem::get_thread_safety, pygmo::problem_get_thread_safety_docstring().c_str());

    // Algorithm class.
    pygmo::algorithm_ptr
        = detail::make_unique<bp::class_<algorithm>>("algorithm", pygmo::algorithm_docstring().c_str(), bp::init<>());
    auto &algorithm_class = *pygmo::algorithm_ptr;
    algorithm_class.def(bp::init<const bp::object &>((bp::arg("uda"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::algorithm_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<algorithm>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<algorithm>)
        // Algorithm extraction.
        .def("_py_extract", &pygmo::generic_py_extract<algorithm>)
        // Algorithm methods.
        .def("evolve", &algorithm::evolve, pygmo::algorithm_evolve_docstring().c_str(), (bp::arg("pop")))
        .def("set_seed", &algorithm::set_seed, pygmo::algorithm_set_seed_docstring().c_str(), (bp::arg("seed")))
        .def("has_set_seed", &algorithm::has_set_seed, pygmo::algorithm_has_set_seed_docstring().c_str())
        .def("set_verbosity", &algorithm::set_verbosity, pygmo::algorithm_set_verbosity_docstring().c_str(),
             (bp::arg("level")))
        .def("has_set_verbosity", &algorithm::has_set_verbosity, pygmo::algorithm_has_set_verbosity_docstring().c_str())
        .def("is_stochastic", &algorithm::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.core.algorithm.has_set_seed()`.\n")
        .def("get_name", &algorithm::get_name, pygmo::algorithm_get_name_docstring().c_str())
        .def("get_extra_info", &algorithm::get_extra_info, pygmo::algorithm_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &algorithm::get_thread_safety,
             pygmo::algorithm_get_thread_safety_docstring().c_str());

    // Expose problems and algorithms.
    pygmo::expose_problems();
    pygmo::expose_algorithms();

    // Exposition of various structured utilities
    // Hypervolume class
    bp::class_<hypervolume>("hypervolume", "Hypervolume Class")
        .def("__init__", bp::make_constructor(lcast([](const bp::object &points) {
                                                  auto vvd_points = pygmo::to_vvd(points);
                                                  return ::new hypervolume(vvd_points, true);
                                              }),
                                              bp::default_call_policies(), (bp::arg("points"))),
             pygmo::hv_init2_docstring().c_str())
        .def("__init__", bp::make_constructor(lcast([](const population &pop) { return ::new hypervolume(pop, true); }),
                                              bp::default_call_policies(), (bp::arg("pop"))),
             pygmo::hv_init1_docstring().c_str())
        .def("compute",
             lcast([](const hypervolume &hv, const bp::object &r_point) { return hv.compute(pygmo::to_vd(r_point)); }),
             (bp::arg("ref_point")))
        .def("compute",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.compute(pygmo::to_vd(r_point), *hv_algo);
             }),
             pygmo::hv_compute_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("exclusive", lcast([](const hypervolume &hv, unsigned p_idx, const bp::object &r_point) {
                 return hv.exclusive(p_idx, pygmo::to_vd(r_point));
             }),
             (bp::arg("idx"), bp::arg("ref_point")))
        .def("exclusive", lcast([](const hypervolume &hv, unsigned p_idx, const bp::object &r_point,
                                   boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.exclusive(p_idx, pygmo::to_vd(r_point), *hv_algo);
             }),
             pygmo::hv_exclusive_docstring().c_str(), (bp::arg("idx"), bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("least_contributor", lcast([](const hypervolume &hv, const bp::object &r_point) {
                 return hv.least_contributor(pygmo::to_vd(r_point));
             }),
             (bp::arg("ref_point")))
        .def("least_contributor",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.least_contributor(pygmo::to_vd(r_point), *hv_algo);
             }),
             pygmo::hv_least_contributor_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("greatest_contributor", lcast([](const hypervolume &hv, const bp::object &r_point) {
                 return hv.greatest_contributor(pygmo::to_vd(r_point));
             }),
             (bp::arg("ref_point")))
        .def("greatest_contributor",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.greatest_contributor(pygmo::to_vd(r_point), *hv_algo);
             }),
             pygmo::hv_greatest_contributor_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("contributions", lcast([](const hypervolume &hv, const bp::object &r_point) {
                 return pygmo::v_to_a(hv.contributions(pygmo::to_vd(r_point)));
             }),
             (bp::arg("ref_point")))
        .def("contributions",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return pygmo::v_to_a(hv.contributions(pygmo::to_vd(r_point), *hv_algo));
             }),
             pygmo::hv_contributions_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .add_property("copy_points", &hypervolume::get_copy_points, &hypervolume::set_copy_points)
        .def("get_points", lcast([](const hypervolume &hv) { return pygmo::vv_to_a(hv.get_points()); }))
        .def("refpoint", lcast([](const hypervolume &hv, double offset) { return pygmo::v_to_a(hv.refpoint(offset)); }),
             pygmo::hv_refpoint_docstring().c_str(), (bp::arg("offset") = 0));

    // Hypervolume algorithms
    bp::class_<hv_algorithm, boost::noncopyable>("_hv_algorithm", bp::no_init).def("get_name", &hv_algorithm::get_name);
    bp::class_<hvwfg, bp::bases<hv_algorithm>>("hvwfg", pygmo::hvwfg_docstring().c_str())
        .def(bp::init<unsigned>((bp::arg("stop_dimension") = 2)));
    bp::class_<bf_approx, bp::bases<hv_algorithm>>("bf_approx", pygmo::bf_approx_docstring().c_str())
        .def(bp::init<bool, unsigned, double, double, double, double, double, double>(
            (bp::arg("use_exact") = true, bp::arg("trivial_subcase_size") = 1u, bp::arg("eps") = 1e-2,
             bp::arg("delta") = 1e-6, bp::arg("delta_multiplier") = 0.775, bp::arg("alpha") = 0.2,
             bp::arg("initial_delta_coeff") = 0.1, bp::arg("gamma") = 0.25)))
        .def(bp::init<bool, unsigned, double, double, double, double, double, double, unsigned>(
            (bp::arg("use_exact") = true, bp::arg("trivial_subcase_size") = 1u, bp::arg("eps") = 1e-2,
             bp::arg("delta") = 1e-6, bp::arg("delta_multiplier") = 0.775, bp::arg("alpha") = 0.2,
             bp::arg("initial_delta_coeff") = 0.1, bp::arg("gamma") = 0.25, bp::arg("seed"))));
    bp::class_<bf_fpras, bp::bases<hv_algorithm>>("bf_fpras", pygmo::bf_fpras_docstring().c_str())
        .def(bp::init<double, double>((bp::arg("eps") = 1e-2, bp::arg("delta") = 1e-2)))
        .def(bp::init<double, double, unsigned>((bp::arg("eps") = 1e-2, bp::arg("delta") = 1e-2, bp::arg("seed"))));
    bp::class_<hv2d, bp::bases<hv_algorithm>>("hv2d", pygmo::hv2d_docstring().c_str(), bp::init<>());
    bp::class_<hv3d, bp::bases<hv_algorithm>>("hv3d", pygmo::hv3d_docstring().c_str(), bp::init<>());

    // Exposition of stand alone functions
    // Multi-objective utilities
    bp::def("fast_non_dominated_sorting", lcast([](const bp::object &x) -> bp::object {
                auto fnds = fast_non_dominated_sorting(pygmo::to_vvd(x));
                // the non-dominated fronts
                auto ndf = std::get<0>(fnds);
                bp::list ndf_py;
                for (const std::vector<vector_double::size_type> &front : ndf) {
                    ndf_py.append(pygmo::v_to_a(front));
                }
                // the domination list
                auto dl = std::get<1>(fnds);
                bp::list dl_py;
                for (const auto &item : dl) {
                    dl_py.append(pygmo::v_to_a(item));
                }
                return bp::make_tuple(ndf_py, dl_py, pygmo::v_to_a(std::get<2>(fnds)),
                                      pygmo::v_to_a(std::get<3>(fnds)));
            }),
            pygmo::fast_non_dominated_sorting_docstring().c_str(), boost::python::arg("points"));
    bp::def("nadir", lcast([](const bp::object &p) { return pygmo::v_to_a(pagmo::nadir(pygmo::to_vvd(p))); }),
            pygmo::nadir_docstring().c_str(), bp::arg("points"));
    bp::def("ideal", lcast([](const bp::object &p) { return pygmo::v_to_a(pagmo::ideal(pygmo::to_vvd(p))); }),
            pygmo::ideal_docstring().c_str(), bp::arg("points"));

    // Island.
    pygmo::island_ptr
        = detail::make_unique<bp::class_<island>>("island", pygmo::island_docstring().c_str(), bp::init<>());
    auto &island_class = *pygmo::island_ptr;
    island_class.def(bp::init<const algorithm &, const population &>())
        .def(bp::init<const bp::object &, const algorithm &, const population &>())
        .def(repr(bp::self))
        .def_pickle(pygmo::island_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<island>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<island>)
        .def("evolve", lcast([](island &isl, unsigned n) { isl.evolve(n); }), pygmo::island_evolve_docstring().c_str(),
             boost::python::arg("n") = 1u)
        .def("busy", &island::busy, pygmo::island_busy_docstring().c_str())
        .def("wait", &island::wait, pygmo::island_wait_docstring().c_str())
        .def("get", &island::get, pygmo::island_get_docstring().c_str())
        .def("get_population", &island::get_population, pygmo::island_get_population_docstring().c_str())
        .def("get_algorithm", &island::get_algorithm, pygmo::island_get_algorithm_docstring().c_str())
        .def("set_population", &island::set_population, pygmo::island_set_population_docstring().c_str(),
             bp::arg("pop"))
        .def("set_algorithm", &island::set_algorithm, pygmo::island_set_algorithm_docstring().c_str(), bp::arg("algo"))
        .def("get_thread_safety", lcast([](const island &isl) -> bp::tuple {
                 const auto ts = isl.get_thread_safety();
                 return bp::make_tuple(ts[0], ts[1]);
             }),
             pygmo::island_get_thread_safety_docstring().c_str())
        .def("get_name", &island::get_name, pygmo::island_get_name_docstring().c_str())
        .def("get_extra_info", &island::get_extra_info, pygmo::island_get_extra_info_docstring().c_str());

    // Thread island.
    auto ti = pygmo::expose_island<thread_island>("thread_island", pygmo::thread_island_docstring().c_str());

    // Archi.
    bp::class_<archipelago> archi_class("archipelago", pygmo::archipelago_docstring().c_str(),
                                        bp::init<archipelago::size_type, const island &>());
    archi_class.def(repr(bp::self))
        .def_pickle(archipelago_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<archipelago>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<archipelago>)
        // Size.
        .def("__len__", &archipelago::size)
        .def("evolve", lcast([](archipelago &archi, unsigned n) { archi.evolve(n); }),
             pygmo::archipelago_evolve_docstring().c_str(), boost::python::arg("n") = 1u)
        .def("busy", &archipelago::busy, pygmo::archipelago_busy_docstring().c_str())
        .def("wait", &archipelago::wait, pygmo::archipelago_wait_docstring().c_str())
        .def("get", &archipelago::get, pygmo::archipelago_get_docstring().c_str())
        .def("__getitem__", lcast([](archipelago &archi, archipelago::size_type n) -> island & { return archi[n]; }),
             pygmo::archipelago_getitem_docstring().c_str(), bp::return_internal_reference<>())
        // NOTE: docs for push_back() are in the Python reimplementation.
        .def("_push_back", lcast([](archipelago &archi, const island &isl) { archi.push_back(isl); }));
}
