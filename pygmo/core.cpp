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
#define PY_ARRAY_UNIQUE_SYMBOL pygmo_ARRAY_API
#include <pygmo/numpy.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

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

#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/gte_getter.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/thread_island.hpp>
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
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/gradients_and_hessians.hpp>
#include <pagmo/utils/hv_algos/hv_algorithm.hpp>
#include <pagmo/utils/hv_algos/hv_bf_approx.hpp>
#include <pagmo/utils/hv_algos/hv_bf_fpras.hpp>
#include <pagmo/utils/hv_algos/hv_hv2d.hpp>
#include <pagmo/utils/hv_algos/hv_hv3d.hpp>
#include <pagmo/utils/hv_algos/hv_hvwfg.hpp>
#include <pagmo/utils/hypervolume.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include <pygmo/algorithm.hpp>
#include <pygmo/bfe.hpp>
#include <pygmo/common_utils.hpp>
#include <pygmo/docstrings.hpp>
#include <pygmo/expose_algorithms.hpp>
#include <pygmo/expose_bfes.hpp>
#include <pygmo/expose_islands.hpp>
#include <pygmo/expose_problems.hpp>
#include <pygmo/expose_r_policies.hpp>
#include <pygmo/expose_s_policies.hpp>
#include <pygmo/expose_topologies.hpp>
#include <pygmo/island.hpp>
#include <pygmo/object_serialization.hpp>
#include <pygmo/problem.hpp>
#include <pygmo/pygmo_classes.hpp>
#include <pygmo/r_policy.hpp>
#include <pygmo/s_policy.hpp>
#include <pygmo/topology.hpp>

namespace bp = boost::python;
using namespace pagmo;

namespace pygmo
{

// Exposed pagmo::problem.
std::unique_ptr<bp::class_<pagmo::problem>> problem_ptr;

// Exposed pagmo::algorithm.
std::unique_ptr<bp::class_<pagmo::algorithm>> algorithm_ptr;

// Exposed pagmo::island.
std::unique_ptr<bp::class_<pagmo::island>> island_ptr;

// Exposed pagmo::bfe.
std::unique_ptr<bp::class_<pagmo::bfe>> bfe_ptr;

// Exposed pagmo::topology.
std::unique_ptr<bp::class_<pagmo::topology>> topology_ptr;

// Exposed pagmo::r_policy.
std::unique_ptr<bp::class_<pagmo::r_policy>> r_policy_ptr;

// Exposed pagmo::s_policy.
std::unique_ptr<bp::class_<pagmo::s_policy>> s_policy_ptr;

namespace detail
{

namespace
{

// Test that the serialization of BP objects works as expected.
// The object returned by this function should be identical to the input
// object.
bp::object test_object_serialization(const bp::object &o)
{
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << object_to_vchar(o);
    }
    const std::string tmp_str = oss.str();
    std::istringstream iss;
    iss.str(tmp_str);
    bp::object retval;
    {
        boost::archive::binary_iarchive iarchive(iss);
        std::vector<char> tmp;
        iarchive >> tmp;
        retval = vchar_to_object(tmp);
    }
    return retval;
}

// The cleanup function.
// This function will be registered to be called when the pygmo core module is unloaded
// (see the __init__.py file). I am not 100% sure it is needed to reset these global
// variables, but it makes me nervous to have global boost python objects around on shutdown.
// NOTE: probably it would be better to register the cleanup function directly in core.cpp,
// to be executed when the compiled module gets unloaded (now it is executed when the pygmo
// supermodule gets unloaded).
void cleanup()
{
    problem_ptr.reset();

    algorithm_ptr.reset();

    island_ptr.reset();

    bfe_ptr.reset();

    topology_ptr.reset();

    r_policy_ptr.reset();

    s_policy_ptr.reset();
}

// Serialization support for the population class.
struct population_pickle_suite : bp::pickle_suite {
    static bp::tuple getstate(const population &pop)
    {
        std::ostringstream oss;
        {
            boost::archive::binary_oarchive oarchive(oss);
            oarchive << pop;
        }
        auto s = oss.str();
        return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
    }
    static void setstate(population &pop, bp::tuple state)
    {
        if (len(state) != 2) {
            pygmo_throw(PyExc_ValueError, ("the state tuple passed for population deserialization "
                                           "must have 2 elements, but instead it has "
                                           + std::to_string(len(state)) + " elements")
                                              .c_str());
        }

        // Make sure we import all the aps specified in the archive.
        import_aps(bp::list(state[1]));

        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize a population");
        }
        const auto size = len(state[0]);
        std::string s(ptr, ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
            boost::archive::binary_iarchive iarchive(iss);
            iarchive >> pop;
        }
    }
};

// Serialization support for the archi class.
struct archipelago_pickle_suite : bp::pickle_suite {
    static bp::tuple getstate(const archipelago &archi)
    {
        std::ostringstream oss;
        {
            boost::archive::binary_oarchive oarchive(oss);
            oarchive << archi;
        }
        auto s = oss.str();
        return bp::make_tuple(make_bytes(s.data(), boost::numeric_cast<Py_ssize_t>(s.size())), get_ap_list());
    }
    static void setstate(archipelago &archi, bp::tuple state)
    {
        if (len(state) != 2) {
            pygmo_throw(PyExc_ValueError, ("the state tuple passed for archipelago deserialization "
                                           "must have 2 elements, but instead it has "
                                           + std::to_string(len(state)) + " elements")
                                              .c_str());
        }

        // Make sure we import all the aps specified in the archive.
        import_aps(bp::list(state[1]));

        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError, "a bytes object is needed to deserialize an archipelago");
        }
        const auto size = len(state[0]);
        std::string s(ptr, ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
            boost::archive::binary_iarchive iarchive(iss);
            iarchive >> archi;
        }
    }
};

// Helper function to test the to_vd functionality.
bool test_to_vd(const bp::object &o, unsigned n)
{
    auto res = obj_to_vector<vector_double>(o);
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
bool test_to_vvd(const bp::object &o, unsigned n, unsigned m)
{
    auto res = obj_to_vvector<std::vector<vector_double>>(o);
    return res.size() == n
           && std::all_of(res.begin(), res.end(), [m](const vector_double &v) { return v.size() == m; });
}

// NOTE: we need to provide a custom raii waiter in the island. The reason is the following.
// When we call wait() from Python, the calling thread will be holding the GIL and then we will be waiting
// for evolutions in the island to finish. During this time, no
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
    gil_thread_ensurer gte;
    gil_releaser gr;
};

// Small helper function to get the max value of unsigned.
constexpr unsigned max_unsigned()
{
    return std::numeric_limits<unsigned>::max();
}

// Small helper to return the next random unsigned
// from the global pagmo rng.
unsigned random_device_next()
{
    return pagmo::random_device::next();
}

// The set containing the list of registered APs.
std::unordered_set<std::string> ap_set;

} // namespace

} // namespace detail

} // namespace pygmo

// Detect if pygmo can use the multiprocessing module.
// NOTE: the mp machinery is supported since Python 3.4 or on Windows.
#if defined(_WIN32) || PY_MAJOR_VERSION > 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 4)

#define PYGMO_CAN_USE_MP

#endif

BOOST_PYTHON_MODULE(core)
{
    using pygmo::lcast;

    // This function needs to be called before doing anything with threads.
    // https://docs.python.org/3/c-api/init.html
    ::PyEval_InitThreads();

    // Init numpy.
    // NOTE: only the second import is strictly necessary. We run a first import from BP
    // because that is the easiest way to detect whether numpy is installed or not (rather
    // than trying to figure out a way to detect it from import_array()).
    try {
        bp::import("numpy.core.multiarray");
    } catch (...) {
        pygmo::builtin().attr("print")(
            u8"\033[91m====ERROR====\nThe NumPy module could not be imported. "
            u8"Please make sure that NumPy has been correctly installed.\n====ERROR====\033[0m");
        pygmo_throw(PyExc_ImportError, "");
    }
    pygmo::numpy_import_array();

    // Check that cloudpickle is available.
    try {
        bp::import("cloudpickle");
    } catch (...) {
        pygmo::builtin().attr("print")(
            u8"\033[91m====ERROR====\nThe cloudpickle module could not be imported. "
            u8"Please make sure that cloudpickle has been correctly installed.\n====ERROR====\033[0m");
        pygmo_throw(PyExc_ImportError, "");
    }

    // Override the default implementation of the island factory.
    detail::island_factory
        = [](const algorithm &algo, const population &pop, std::unique_ptr<detail::isl_inner_base> &ptr) {
              if (algo.get_thread_safety() >= thread_safety::basic
                  && pop.get_problem().get_thread_safety() >= thread_safety::basic) {
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
#if defined(PYGMO_CAN_USE_MP)
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
    detail::wait_raii_getter = []() { return std::make_shared<pygmo::detail::py_wait_locks>(); };

    // NOTE: set the gte getter.
    detail::gte_getter = []() { return std::make_shared<pygmo::gil_thread_ensurer>(); };

    // Setup doc options
    bp::docstring_options doc_options;
    doc_options.enable_all();
    doc_options.disable_cpp_signatures();
    doc_options.disable_py_signatures();

    // The thread_safety enum.
    bp::enum_<thread_safety>("_thread_safety")
        .value("none", thread_safety::none)
        .value("basic", thread_safety::basic)
        .value("constant", thread_safety::constant);

    // The evolve_status enum.
    bp::enum_<evolve_status>("_evolve_status")
        .value("idle", evolve_status::idle)
        .value("busy", evolve_status::busy)
        .value("idle_error", evolve_status::idle_error)
        .value("busy_error", evolve_status::busy_error);

    // Migration type enum.
    bp::enum_<migration_type>("_migration_type")
        .value("p2p", migration_type::p2p)
        .value("broadcast", migration_type::broadcast);

    // Migrant handling policy enum.
    bp::enum_<migrant_handling>("_migrant_handling")
        .value("preserve", migrant_handling::preserve)
        .value("evict", migrant_handling::evict);

    // Expose utility functions for testing purposes.
    bp::def("_builtin", &pygmo::builtin);
    bp::def("_type", &pygmo::type);
    bp::def("_str", &pygmo::str);
    bp::def("_callable", &pygmo::callable);
    bp::def("_deepcopy", &pygmo::deepcopy);
    bp::def("_to_sp", &pygmo::obj_to_sp);
    bp::def("_test_object_serialization", &pygmo::detail::test_object_serialization);
    bp::def("_test_to_vd", &pygmo::detail::test_to_vd);
    bp::def("_test_to_vvd", &pygmo::detail::test_to_vvd);

    // Expose cleanup function.
    bp::def("_cleanup", &pygmo::detail::cleanup);

    // The max_unsigned() helper.
    bp::def("_max_unsigned", &pygmo::detail::max_unsigned);

    // The random_device_next() helper.
    bp::def("_random_device_next", &pygmo::detail::random_device_next);

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

    // Create the batch_evaluators submodule.
    std::string batch_evaluators_module_name
        = bp::extract<std::string>(bp::scope().attr("__name__") + ".batch_evaluators");
    PyObject *batch_evaluators_module_ptr = PyImport_AddModule(batch_evaluators_module_name.c_str());
    if (!batch_evaluators_module_ptr) {
        pygmo_throw(PyExc_RuntimeError, "error while creating the 'batch_evaluators' submodule");
    }
    auto batch_evaluators_module = bp::object(bp::handle<>(bp::borrowed(batch_evaluators_module_ptr)));
    bp::scope().attr("batch_evaluators") = batch_evaluators_module;

    // Create the topologies submodule.
    std::string topologies_module_name = bp::extract<std::string>(bp::scope().attr("__name__") + ".topologies");
    PyObject *topologies_module_ptr = PyImport_AddModule(topologies_module_name.c_str());
    if (!topologies_module_ptr) {
        pygmo_throw(PyExc_RuntimeError, "error while creating the 'topologies' submodule");
    }
    auto topologies_module = bp::object(bp::handle<>(bp::borrowed(topologies_module_ptr)));
    bp::scope().attr("topologies") = topologies_module;

    // Create the r_policies submodule.
    std::string r_policies_module_name = bp::extract<std::string>(bp::scope().attr("__name__") + ".r_policies");
    PyObject *r_policies_module_ptr = PyImport_AddModule(r_policies_module_name.c_str());
    if (!r_policies_module_ptr) {
        pygmo_throw(PyExc_RuntimeError, "error while creating the 'r_policies' submodule");
    }
    auto r_policies_module = bp::object(bp::handle<>(bp::borrowed(r_policies_module_ptr)));
    bp::scope().attr("r_policies") = r_policies_module;

    // Create the s_policies submodule.
    std::string s_policies_module_name = bp::extract<std::string>(bp::scope().attr("__name__") + ".s_policies");
    PyObject *s_policies_module_ptr = PyImport_AddModule(s_policies_module_name.c_str());
    if (!s_policies_module_ptr) {
        pygmo_throw(PyExc_RuntimeError, "error while creating the 's_policies' submodule");
    }
    auto s_policies_module = bp::object(bp::handle<>(bp::borrowed(s_policies_module_ptr)));
    bp::scope().attr("s_policies") = s_policies_module;

    // Store the pointers to the classes that can be extended by APs.
    bp::scope().attr("_problem_address") = reinterpret_cast<std::uintptr_t>(&pygmo::problem_ptr);
    bp::scope().attr("_algorithm_address") = reinterpret_cast<std::uintptr_t>(&pygmo::algorithm_ptr);
    bp::scope().attr("_island_address") = reinterpret_cast<std::uintptr_t>(&pygmo::island_ptr);
    bp::scope().attr("_bfe_address") = reinterpret_cast<std::uintptr_t>(&pygmo::bfe_ptr);
    bp::scope().attr("_topology_address") = reinterpret_cast<std::uintptr_t>(&pygmo::topology_ptr);
    bp::scope().attr("_r_policy_address") = reinterpret_cast<std::uintptr_t>(&pygmo::r_policy_ptr);
    bp::scope().attr("_s_policy_address") = reinterpret_cast<std::uintptr_t>(&pygmo::s_policy_ptr);

    // Store the address to the list of registered APs.
    bp::scope().attr("_ap_set_address") = reinterpret_cast<std::uintptr_t>(&pygmo::detail::ap_set);

    // Population class.
    bp::class_<population> pop_class("population", pygmo::population_docstring().c_str(), bp::no_init);
    // Ctors from problem.
    // NOTE: we expose only the ctors from pagmo::problem, not from C++ or Python UDPs. An __init__ wrapper
    // on the Python side will take care of cting a pagmo::problem from the input UDP, and then invoke this ctor.
    // This way we avoid having to expose a different ctor for every exposed C++ prob. Same idea with
    // the bfe argument.
    pop_class.def(bp::init<const problem &, population::size_type, unsigned>())
        .def(bp::init<const problem &, const bfe &, population::size_type, unsigned>())
        // Repr.
        .def(repr(bp::self))
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<population>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<population>)
        .def_pickle(pygmo::detail::population_pickle_suite())
        .def("push_back", lcast([](population &pop, const bp::object &x, const bp::object &f) {
                 if (f.is_none()) {
                     pop.push_back(pygmo::obj_to_vector<vector_double>(x));
                 } else {
                     pop.push_back(pygmo::obj_to_vector<vector_double>(x), pygmo::obj_to_vector<vector_double>(f));
                 }
             }),
             pygmo::population_push_back_docstring().c_str(), (bp::arg("x"), bp::arg("f") = bp::object()))
        .def("random_decision_vector",
             lcast([](const population &pop) { return pygmo::vector_to_ndarr(pop.random_decision_vector()); }),
             pygmo::population_random_decision_vector_docstring().c_str())
        .def("best_idx", lcast([](const population &pop, const bp::object &tol) {
                 return pop.best_idx(pygmo::obj_to_vector<vector_double>(tol));
             }),
             (bp::arg("tol")))
        .def("best_idx", lcast([](const population &pop, double tol) { return pop.best_idx(tol); }), (bp::arg("tol")))
        .def("best_idx", lcast([](const population &pop) { return pop.best_idx(); }),
             pygmo::population_best_idx_docstring().c_str())
        .def("worst_idx", lcast([](const population &pop, const bp::object &tol) {
                 return pop.worst_idx(pygmo::obj_to_vector<vector_double>(tol));
             }),
             (bp::arg("tol")))
        .def("worst_idx", lcast([](const population &pop, double tol) { return pop.worst_idx(tol); }), (bp::arg("tol")))
        .def("worst_idx", lcast([](const population &pop) { return pop.worst_idx(); }),
             pygmo::population_worst_idx_docstring().c_str())
        .def("__len__", &population::size)
        .def("set_xf", lcast([](population &pop, population::size_type i, const bp::object &x, const bp::object &f) {
                 pop.set_xf(i, pygmo::obj_to_vector<vector_double>(x), pygmo::obj_to_vector<vector_double>(f));
             }),
             pygmo::population_set_xf_docstring().c_str())
        .def("set_x", lcast([](population &pop, population::size_type i, const bp::object &x) {
                 pop.set_x(i, pygmo::obj_to_vector<vector_double>(x));
             }),
             pygmo::population_set_x_docstring().c_str())
        .def("get_f", lcast([](const population &pop) { return pygmo::vvector_to_ndarr(pop.get_f()); }),
             pygmo::population_get_f_docstring().c_str())
        .def("get_x", lcast([](const population &pop) { return pygmo::vvector_to_ndarr(pop.get_x()); }),
             pygmo::population_get_x_docstring().c_str())
        .def("get_ID", lcast([](const population &pop) { return pygmo::vector_to_ndarr(pop.get_ID()); }),
             pygmo::population_get_ID_docstring().c_str())
        .def("get_seed", &population::get_seed, pygmo::population_get_seed_docstring().c_str());
    pygmo::add_property(pop_class, "champion_x",
                        lcast([](const population &pop) { return pygmo::vector_to_ndarr(pop.champion_x()); }),
                        pygmo::population_champion_x_docstring().c_str());
    pygmo::add_property(pop_class, "champion_f",
                        lcast([](const population &pop) { return pygmo::vector_to_ndarr(pop.champion_f()); }),
                        pygmo::population_champion_f_docstring().c_str());
    pygmo::add_property(pop_class, "problem",
                        bp::make_function(lcast([](population &pop) -> problem & { return pop.get_problem(); }),
                                          bp::return_internal_reference<>()),
                        pygmo::population_problem_docstring().c_str());

    // Problem class.
    pygmo::problem_ptr
        = detail::make_unique<bp::class_<problem>>("problem", pygmo::problem_docstring().c_str(), bp::init<>());
    auto &problem_class = pygmo::get_problem_class();
    problem_class.def(bp::init<const bp::object &>((bp::arg("udp"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::problem_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<problem>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<problem>)
        // UDP extraction.
        .def("_py_extract", &pygmo::generic_py_extract<problem>)
        // Problem methods.
        .def("fitness", lcast([](const pagmo::problem &p, const bp::object &dv) {
                 return pygmo::vector_to_ndarr(p.fitness(pygmo::obj_to_vector<vector_double>(dv)));
             }),
             pygmo::problem_fitness_docstring().c_str(), (bp::arg("dv")))
        .def("get_bounds", lcast([](const pagmo::problem &p) -> bp::tuple {
                 auto retval = p.get_bounds();
                 return bp::make_tuple(pygmo::vector_to_ndarr(retval.first), pygmo::vector_to_ndarr(retval.second));
             }),
             pygmo::problem_get_bounds_docstring().c_str())
        .def("get_lb", lcast([](const pagmo::problem &p) { return pygmo::vector_to_ndarr(p.get_lb()); }),
             pygmo::problem_get_lb_docstring().c_str())
        .def("get_ub", lcast([](const pagmo::problem &p) { return pygmo::vector_to_ndarr(p.get_ub()); }),
             pygmo::problem_get_ub_docstring().c_str())
        .def("batch_fitness", lcast([](const pagmo::problem &p, const bp::object &dvs) {
                 return pygmo::vector_to_ndarr(p.batch_fitness(pygmo::obj_to_vector<vector_double>(dvs)));
             }),
             pygmo::problem_batch_fitness_docstring().c_str(), (bp::arg("dvs")))
        .def("has_batch_fitness", &problem::has_batch_fitness, pygmo::problem_has_batch_fitness_docstring().c_str())
        .def("gradient", lcast([](const pagmo::problem &p, const bp::object &dv) {
                 return pygmo::vector_to_ndarr(p.gradient(pygmo::obj_to_vector<vector_double>(dv)));
             }),
             pygmo::problem_gradient_docstring().c_str(), (bp::arg("dv")))
        .def("has_gradient", &problem::has_gradient, pygmo::problem_has_gradient_docstring().c_str())
        .def("gradient_sparsity",
             lcast([](const pagmo::problem &p) { return pygmo::sp_to_ndarr(p.gradient_sparsity()); }),
             pygmo::problem_gradient_sparsity_docstring().c_str())
        .def("has_gradient_sparsity", &problem::has_gradient_sparsity,
             pygmo::problem_has_gradient_sparsity_docstring().c_str())
        .def("hessians", lcast([](const pagmo::problem &p, const bp::object &dv) -> bp::list {
                 bp::list retval;
                 const auto h = p.hessians(pygmo::obj_to_vector<vector_double>(dv));
                 for (const auto &v : h) {
                     retval.append(pygmo::vector_to_ndarr(v));
                 }
                 return retval;
             }),
             pygmo::problem_hessians_docstring().c_str(), (bp::arg("dv")))
        .def("has_hessians", &problem::has_hessians, pygmo::problem_has_hessians_docstring().c_str())
        .def("hessians_sparsity", lcast([](const pagmo::problem &p) -> bp::list {
                 bp::list retval;
                 const auto hs = p.hessians_sparsity();
                 for (const auto &sp : hs) {
                     retval.append(pygmo::sp_to_ndarr(sp));
                 }
                 return retval;
             }),
             pygmo::problem_hessians_sparsity_docstring().c_str())
        .def("has_hessians_sparsity", &problem::has_hessians_sparsity,
             pygmo::problem_has_hessians_sparsity_docstring().c_str())
        .def("get_nobj", &problem::get_nobj, pygmo::problem_get_nobj_docstring().c_str())
        .def("get_nx", &problem::get_nx, pygmo::problem_get_nx_docstring().c_str())
        .def("get_nix", &problem::get_nix, pygmo::problem_get_nix_docstring().c_str())
        .def("get_ncx", &problem::get_ncx, pygmo::problem_get_ncx_docstring().c_str())
        .def("get_nf", &problem::get_nf, pygmo::problem_get_nf_docstring().c_str())
        .def("get_nec", &problem::get_nec, pygmo::problem_get_nec_docstring().c_str())
        .def("get_nic", &problem::get_nic, pygmo::problem_get_nic_docstring().c_str())
        .def("get_nc", &problem::get_nc, pygmo::problem_get_nc_docstring().c_str())
        .def("get_fevals", &problem::get_fevals, pygmo::problem_get_fevals_docstring().c_str())
        .def("get_gevals", &problem::get_gevals, pygmo::problem_get_gevals_docstring().c_str())
        .def("get_hevals", &problem::get_hevals, pygmo::problem_get_hevals_docstring().c_str())
        .def("set_seed", &problem::set_seed, pygmo::problem_set_seed_docstring().c_str(), (bp::arg("seed")))
        .def("has_set_seed", &problem::has_set_seed, pygmo::problem_has_set_seed_docstring().c_str())
        .def("is_stochastic", &problem::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.problem.has_set_seed()`.\n")
        .def("feasibility_x", lcast([](const problem &p, const bp::object &x) {
                 return p.feasibility_x(pygmo::obj_to_vector<vector_double>(x));
             }),
             pygmo::problem_feasibility_x_docstring().c_str(), (bp::arg("x")))
        .def("feasibility_f", lcast([](const problem &p, const bp::object &f) {
                 return p.feasibility_f(pygmo::obj_to_vector<vector_double>(f));
             }),
             pygmo::problem_feasibility_f_docstring().c_str(), (bp::arg("f")))
        .def("get_name", &problem::get_name, pygmo::problem_get_name_docstring().c_str())
        .def("get_extra_info", &problem::get_extra_info, pygmo::problem_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &problem::get_thread_safety, pygmo::problem_get_thread_safety_docstring().c_str());
    pygmo::add_property(problem_class, "c_tol",
                        lcast([](const problem &prob) { return pygmo::vector_to_ndarr(prob.get_c_tol()); }),
                        lcast([](problem &prob, const bp::object &c_tol) {
                            bp::extract<double> c_tol_double(c_tol);
                            if (c_tol_double.check()) {
                                prob.set_c_tol(static_cast<double>(c_tol_double));
                            } else {
                                prob.set_c_tol(pygmo::obj_to_vector<vector_double>(c_tol));
                            }
                        }),
                        pygmo::problem_c_tol_docstring().c_str());

    // Algorithm class.
    pygmo::algorithm_ptr
        = detail::make_unique<bp::class_<algorithm>>("algorithm", pygmo::algorithm_docstring().c_str(), bp::init<>());
    auto &algorithm_class = pygmo::get_algorithm_class();
    algorithm_class.def(bp::init<const bp::object &>((bp::arg("uda"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::algorithm_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<algorithm>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<algorithm>)
        // UDA extraction.
        .def("_py_extract", &pygmo::generic_py_extract<algorithm>)
        // Algorithm methods.
        .def("evolve", &algorithm::evolve, pygmo::algorithm_evolve_docstring().c_str(), (bp::arg("pop")))
        .def("set_seed", &algorithm::set_seed, pygmo::algorithm_set_seed_docstring().c_str(), (bp::arg("seed")))
        .def("has_set_seed", &algorithm::has_set_seed, pygmo::algorithm_has_set_seed_docstring().c_str())
        .def("set_verbosity", &algorithm::set_verbosity, pygmo::algorithm_set_verbosity_docstring().c_str(),
             (bp::arg("level")))
        .def("has_set_verbosity", &algorithm::has_set_verbosity, pygmo::algorithm_has_set_verbosity_docstring().c_str())
        .def("is_stochastic", &algorithm::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.algorithm.has_set_seed()`.\n")
        .def("get_name", &algorithm::get_name, pygmo::algorithm_get_name_docstring().c_str())
        .def("get_extra_info", &algorithm::get_extra_info, pygmo::algorithm_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &algorithm::get_thread_safety,
             pygmo::algorithm_get_thread_safety_docstring().c_str());

    // Expose problems and algorithms.
    pygmo::expose_problems_0();
    pygmo::expose_problems_1();
    pygmo::expose_algorithms_0();
    pygmo::expose_algorithms_1();

    // Exposition of various structured utilities
    // Hypervolume class
    bp::class_<hypervolume> hv_class("hypervolume", "Hypervolume Class");
    hv_class
        .def("__init__",
             bp::make_constructor(lcast([](const bp::object &points) {
                                      auto vvd_points = pygmo::obj_to_vvector<std::vector<vector_double>>(points);
                                      return ::new hypervolume(vvd_points, true);
                                  }),
                                  bp::default_call_policies(), (bp::arg("points"))),
             pygmo::hv_init2_docstring().c_str())
        .def("__init__",
             bp::make_constructor(lcast([](const population &pop) { return ::new hypervolume(pop, true); }),
                                  bp::default_call_policies(), (bp::arg("pop"))),
             pygmo::hv_init1_docstring().c_str())
        .def("compute", lcast([](const hypervolume &hv, const bp::object &r_point) {
                 return hv.compute(pygmo::obj_to_vector<vector_double>(r_point));
             }),
             (bp::arg("ref_point")))
        .def("compute",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.compute(pygmo::obj_to_vector<vector_double>(r_point), *hv_algo);
             }),
             pygmo::hv_compute_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("exclusive", lcast([](const hypervolume &hv, unsigned p_idx, const bp::object &r_point) {
                 return hv.exclusive(p_idx, pygmo::obj_to_vector<vector_double>(r_point));
             }),
             (bp::arg("idx"), bp::arg("ref_point")))
        .def("exclusive",
             lcast([](const hypervolume &hv, unsigned p_idx, const bp::object &r_point,
                      boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.exclusive(p_idx, pygmo::obj_to_vector<vector_double>(r_point), *hv_algo);
             }),
             pygmo::hv_exclusive_docstring().c_str(), (bp::arg("idx"), bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("least_contributor", lcast([](const hypervolume &hv, const bp::object &r_point) {
                 return hv.least_contributor(pygmo::obj_to_vector<vector_double>(r_point));
             }),
             (bp::arg("ref_point")))
        .def("least_contributor",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.least_contributor(pygmo::obj_to_vector<vector_double>(r_point), *hv_algo);
             }),
             pygmo::hv_least_contributor_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("greatest_contributor", lcast([](const hypervolume &hv, const bp::object &r_point) {
                 return hv.greatest_contributor(pygmo::obj_to_vector<vector_double>(r_point));
             }),
             (bp::arg("ref_point")))
        .def("greatest_contributor",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return hv.greatest_contributor(pygmo::obj_to_vector<vector_double>(r_point), *hv_algo);
             }),
             pygmo::hv_greatest_contributor_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("contributions", lcast([](const hypervolume &hv, const bp::object &r_point) {
                 return pygmo::vector_to_ndarr(hv.contributions(pygmo::obj_to_vector<vector_double>(r_point)));
             }),
             (bp::arg("ref_point")))
        .def("contributions",
             lcast([](const hypervolume &hv, const bp::object &r_point, boost::shared_ptr<hv_algorithm> hv_algo) {
                 return pygmo::vector_to_ndarr(
                     hv.contributions(pygmo::obj_to_vector<vector_double>(r_point), *hv_algo));
             }),
             pygmo::hv_contributions_docstring().c_str(), (bp::arg("ref_point"), bp::arg("hv_algo")))
        .def("get_points", lcast([](const hypervolume &hv) { return pygmo::vvector_to_ndarr(hv.get_points()); }))
        .def("refpoint",
             lcast([](const hypervolume &hv, double offset) { return pygmo::vector_to_ndarr(hv.refpoint(offset)); }),
             pygmo::hv_refpoint_docstring().c_str(), (bp::arg("offset") = 0));
    pygmo::add_property(hv_class, "copy_points", &hypervolume::get_copy_points, &hypervolume::set_copy_points);

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
                auto fnds = fast_non_dominated_sorting(pygmo::obj_to_vvector<std::vector<vector_double>>(x));
                // the non-dominated fronts
                auto ndf = std::get<0>(fnds);
                bp::list ndf_py;
                for (const std::vector<vector_double::size_type> &front : ndf) {
                    ndf_py.append(pygmo::vector_to_ndarr(front));
                }
                // the domination list
                auto dl = std::get<1>(fnds);
                bp::list dl_py;
                for (const auto &item : dl) {
                    dl_py.append(pygmo::vector_to_ndarr(item));
                }
                return bp::make_tuple(ndf_py, dl_py, pygmo::vector_to_ndarr(std::get<2>(fnds)),
                                      pygmo::vector_to_ndarr(std::get<3>(fnds)));
            }),
            pygmo::fast_non_dominated_sorting_docstring().c_str(), boost::python::arg("points"));
    bp::def("pareto_dominance", lcast([](const bp::object &obj1, const bp::object &obj2) {
                return pareto_dominance(pygmo::obj_to_vector<vector_double>(obj1),
                                        pygmo::obj_to_vector<vector_double>(obj2));
            }),
            pygmo::pareto_dominance_docstring().c_str(), (bp::arg("obj1"), bp::arg("obj2")));
    bp::def("non_dominated_front_2d", lcast([](const bp::object &points) {
                return pygmo::vector_to_ndarr(
                    non_dominated_front_2d(pygmo::obj_to_vvector<std::vector<vector_double>>(points)));
            }),
            pygmo::non_dominated_front_2d_docstring().c_str(), bp::arg("points"));
    bp::def("crowding_distance", lcast([](const bp::object &points) {
                return pygmo::vector_to_ndarr(
                    crowding_distance(pygmo::obj_to_vvector<std::vector<vector_double>>(points)));
            }),
            pygmo::crowding_distance_docstring().c_str(), bp::arg("points"));
    bp::def("sort_population_mo", lcast([](const bp::object &input_f) {
                return pygmo::vector_to_ndarr(
                    sort_population_mo(pygmo::obj_to_vvector<std::vector<vector_double>>(input_f)));
            }),
            pygmo::sort_population_mo_docstring().c_str(), bp::arg("points"));
    bp::def("select_best_N_mo", lcast([](const bp::object &input_f, unsigned N) {
                return pygmo::vector_to_ndarr(
                    select_best_N_mo(pygmo::obj_to_vvector<std::vector<vector_double>>(input_f), N));
            }),
            pygmo::select_best_N_mo_docstring().c_str(), (bp::arg("points"), bp::arg("N")));
    bp::def(
        "decomposition_weights",
        lcast([](vector_double::size_type n_f, vector_double::size_type n_w, const std::string &method, unsigned seed) {
            using reng_t = pagmo::detail::random_engine_type;
            reng_t tmp_rng(static_cast<reng_t::result_type>(seed));
            return pygmo::vvector_to_ndarr(decomposition_weights(n_f, n_w, method, tmp_rng));
        }),
        pygmo::decomposition_weights_docstring().c_str(),
        (bp::arg("n_f"), bp::arg("n_w"), bp::arg("method"), bp::arg("seed")));

    bp::def("decompose_objectives",
            lcast([](const bp::object &objs, const bp::object &weights, const bp::object &ref_point,
                     const std::string &method) {
                return pygmo::vector_to_ndarr(decompose_objectives(
                    pygmo::obj_to_vector<vector_double>(objs), pygmo::obj_to_vector<vector_double>(weights),
                    pygmo::obj_to_vector<vector_double>(ref_point), method));
            }),
            pygmo::decompose_objectives_docstring().c_str(),
            (bp::arg("objs"), bp::arg("weights"), bp::arg("ref_point"), bp::arg("method")));

    bp::def("nadir", lcast([](const bp::object &p) {
                return pygmo::vector_to_ndarr(pagmo::nadir(pygmo::obj_to_vvector<std::vector<vector_double>>(p)));
            }),
            pygmo::nadir_docstring().c_str(), bp::arg("points"));
    bp::def("ideal", lcast([](const bp::object &p) {
                return pygmo::vector_to_ndarr(pagmo::ideal(pygmo::obj_to_vvector<std::vector<vector_double>>(p)));
            }),
            pygmo::ideal_docstring().c_str(), bp::arg("points"));
    // Generic utilities
    bp::def("random_decision_vector", lcast([](const pagmo::problem &p) -> bp::object {
                using reng_t = pagmo::detail::random_engine_type;
                reng_t tmp_rng(static_cast<reng_t::result_type>(pagmo::random_device::next()));
                auto retval = random_decision_vector(p, tmp_rng);
                return pygmo::vector_to_ndarr(retval);
            }),
            pygmo::random_decision_vector_docstring().c_str(), (bp::arg("prob")));
    bp::def("batch_random_decision_vector",
            lcast([](const pagmo::problem &p, pagmo::vector_double::size_type n) -> bp::object {
                using reng_t = pagmo::detail::random_engine_type;
                reng_t tmp_rng(static_cast<reng_t::result_type>(pagmo::random_device::next()));
                auto retval = batch_random_decision_vector(p, n, tmp_rng);
                return pygmo::vector_to_ndarr(retval);
            }),
            pygmo::batch_random_decision_vector_docstring().c_str(), (bp::arg("prob"), bp::arg("n")));

    // Gradient and Hessians utilities
    bp::def("estimate_sparsity", lcast([](const bp::object &func, const bp::object &x, double dx) -> bp::object {
                auto f = [&func](const vector_double &x_) {
                    return pygmo::obj_to_vector<vector_double>(func(pygmo::vector_to_ndarr(x_)));
                };
                auto retval = estimate_sparsity(f, pygmo::obj_to_vector<vector_double>(x), dx);
                return pygmo::sp_to_ndarr(retval);
            }),
            pygmo::estimate_sparsity_docstring().c_str(), (bp::arg("callable"), bp::arg("x"), bp::arg("dx") = 1e-8));
    bp::def("estimate_gradient", lcast([](const bp::object &func, const bp::object &x, double dx) -> bp::object {
                auto f = [&func](const vector_double &x_) {
                    return pygmo::obj_to_vector<vector_double>(func(pygmo::vector_to_ndarr(x_)));
                };
                auto retval = estimate_gradient(f, pygmo::obj_to_vector<vector_double>(x), dx);
                return pygmo::vector_to_ndarr(retval);
            }),
            pygmo::estimate_gradient_docstring().c_str(), (bp::arg("callable"), bp::arg("x"), bp::arg("dx") = 1e-8));
    bp::def("estimate_gradient_h", lcast([](const bp::object &func, const bp::object &x, double dx) -> bp::object {
                auto f = [&func](const vector_double &x_) {
                    return pygmo::obj_to_vector<vector_double>(func(pygmo::vector_to_ndarr(x_)));
                };
                auto retval = estimate_gradient_h(f, pygmo::obj_to_vector<vector_double>(x), dx);
                return pygmo::vector_to_ndarr(retval);
            }),
            pygmo::estimate_gradient_h_docstring().c_str(), (bp::arg("callable"), bp::arg("x"), bp::arg("dx") = 1e-2));
    // Constrained optimization utilities
    bp::def("compare_fc",
            lcast([](const bp::object &f1, const bp::object &f2, vector_double::size_type nec, const bp::object &tol) {
                return compare_fc(pygmo::obj_to_vector<vector_double>(f1), pygmo::obj_to_vector<vector_double>(f2), nec,
                                  pygmo::obj_to_vector<vector_double>(tol));
            }),
            pygmo::compare_fc_docstring().c_str(), (bp::arg("f1"), bp::arg("f2"), bp::arg("nec"), bp::arg("tol")));
    bp::def("sort_population_con",
            lcast([](const bp::object &input_f, vector_double::size_type nec, const bp::object &tol) {
                return pygmo::vector_to_ndarr(
                    sort_population_con(pygmo::obj_to_vvector<std::vector<vector_double>>(input_f), nec,
                                        pygmo::obj_to_vector<vector_double>(tol)));
            }),
            pygmo::sort_population_con_docstring().c_str(), (bp::arg("input_f"), bp::arg("nec"), bp::arg("tol")));
    // Global random number generator
    bp::def("set_global_rng_seed", lcast([](unsigned seed) { random_device::set_seed(seed); }),
            pygmo::set_global_rng_seed_docstring().c_str(), bp::arg("seed"));

    // Island.
    pygmo::island_ptr
        = detail::make_unique<bp::class_<island>>("island", pygmo::island_docstring().c_str(), bp::init<>());
    auto &island_class = pygmo::get_island_class();
    island_class.def(bp::init<const algorithm &, const population &, const r_policy &, const s_policy &>())
        .def(bp::init<const bp::object &, const algorithm &, const population &, const r_policy &, const s_policy &>())
        .def(repr(bp::self))
        .def_pickle(pygmo::island_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<island>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<island>)
        // UDI extraction.
        .def("_py_extract", &pygmo::generic_py_extract<island>)
        .def("evolve", lcast([](island &isl, unsigned n) { isl.evolve(n); }), pygmo::island_evolve_docstring().c_str(),
             boost::python::arg("n") = 1u)
        .def("wait", &island::wait, pygmo::island_wait_docstring().c_str())
        .def("wait_check", &island::wait_check, pygmo::island_wait_check_docstring().c_str())
        .def("get_population", &island::get_population, pygmo::island_get_population_docstring().c_str())
        .def("get_algorithm", &island::get_algorithm, pygmo::island_get_algorithm_docstring().c_str())
        .def("set_population", &island::set_population, pygmo::island_set_population_docstring().c_str(),
             bp::arg("pop"))
        .def("set_algorithm", &island::set_algorithm, pygmo::island_set_algorithm_docstring().c_str(), bp::arg("algo"))
        .def("get_name", &island::get_name, pygmo::island_get_name_docstring().c_str())
        .def("get_extra_info", &island::get_extra_info, pygmo::island_get_extra_info_docstring().c_str())
        .def("get_r_policy", &island::get_r_policy, pygmo::island_get_r_policy_docstring().c_str())
        .def("get_s_policy", &island::get_s_policy, pygmo::island_get_s_policy_docstring().c_str());
    pygmo::add_property(island_class, "status", &island::status, pygmo::island_status_docstring().c_str());

    // Expose islands.
    pygmo::expose_islands();

    // Archi.
    bp::class_<archipelago> archi_class("archipelago", pygmo::archipelago_docstring().c_str(), bp::init<>());
    archi_class.def(bp::init<const topology &>())
        .def(repr(bp::self))
        .def_pickle(pygmo::detail::archipelago_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<archipelago>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<archipelago>)
        // Size.
        .def("__len__", &archipelago::size)
        .def("evolve", lcast([](archipelago &archi, unsigned n) { archi.evolve(n); }),
             pygmo::archipelago_evolve_docstring().c_str(), boost::python::arg("n") = 1u)
        .def("wait", &archipelago::wait, pygmo::archipelago_wait_docstring().c_str())
        .def("wait_check", &archipelago::wait_check, pygmo::archipelago_wait_check_docstring().c_str())
        .def("__getitem__", lcast([](archipelago &archi, archipelago::size_type n) -> island & { return archi[n]; }),
             pygmo::archipelago_getitem_docstring().c_str(), bp::return_internal_reference<>())
        // NOTE: docs for push_back() are in the Python reimplementation.
        .def("_push_back", lcast([](archipelago &archi, const island &isl) { archi.push_back(isl); }))
        // Champions.
        .def("get_champions_f", lcast([](const archipelago &archi) -> bp::list {
                 bp::list retval;
                 auto fs = archi.get_champions_f();
                 for (const auto &f : fs) {
                     retval.append(pygmo::vector_to_ndarr(f));
                 }
                 return retval;
             }),
             pygmo::archipelago_get_champions_f_docstring().c_str())
        .def("get_champions_x", lcast([](const archipelago &archi) -> bp::list {
                 bp::list retval;
                 auto xs = archi.get_champions_x();
                 for (const auto &x : xs) {
                     retval.append(pygmo::vector_to_ndarr(x));
                 }
                 return retval;
             }),
             pygmo::archipelago_get_champions_x_docstring().c_str())
        .def("get_migrants_db", lcast([](const archipelago &archi) -> bp::list {
                 bp::list retval;
                 const auto tmp = archi.get_migrants_db();
                 for (const auto &ig : tmp) {
                     retval.append(pygmo::inds_to_tuple(ig));
                 }
                 return retval;
             }),
             pygmo::archipelago_get_migrants_db_docstring().c_str())
        .def("get_migration_log", lcast([](const archipelago &archi) -> bp::list {
                 bp::list retval;
                 const auto tmp = archi.get_migration_log();
                 for (const auto &le : tmp) {
                     retval.append(
                         bp::make_tuple(std::get<0>(le), std::get<1>(le), pygmo::vector_to_ndarr(std::get<2>(le)),
                                        pygmo::vector_to_ndarr(std::get<3>(le)), std::get<4>(le), std::get<5>(le)));
                 }
                 return retval;
             }),
             pygmo::archipelago_get_migration_log_docstring().c_str())
        .def("get_topology", &archipelago::get_topology, pygmo::archipelago_get_topology_docstring().c_str())
        .def("_set_topology", &archipelago::set_topology)
        .def("set_migration_type", &archipelago::set_migration_type,
             pygmo::archipelago_set_migration_type_docstring().c_str(), (bp::arg("mt")))
        .def("set_migrant_handling", &archipelago::set_migrant_handling,
             pygmo::archipelago_set_migrant_handling_docstring().c_str(), (bp::arg("mh")))
        .def("get_migration_type", &archipelago::get_migration_type,
             pygmo::archipelago_get_migration_type_docstring().c_str())
        .def("get_migrant_handling", &archipelago::get_migrant_handling,
             pygmo::archipelago_get_migrant_handling_docstring().c_str());
    pygmo::add_property(archi_class, "status", &archipelago::status, pygmo::archipelago_status_docstring().c_str());

    // Bfe class.
    pygmo::bfe_ptr = detail::make_unique<bp::class_<bfe>>("bfe", pygmo::bfe_docstring().c_str(), bp::init<>());
    auto &bfe_class = pygmo::get_bfe_class();
    bfe_class.def(bp::init<const bp::object &>((bp::arg("udbfe"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::bfe_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<bfe>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<bfe>)
        // UDBFE extraction.
        .def("_py_extract", &pygmo::generic_py_extract<bfe>)
        // Bfe methods.
        .def("__call__", lcast([](const bfe &b, const problem &prob, const bp::object &dvs) {
                 return pygmo::vector_to_ndarr(b(prob, pygmo::obj_to_vector<vector_double>(dvs)));
             }),
             pygmo::bfe_call_docstring().c_str(), (bp::arg("prob"), bp::arg("dvs")))
        .def("get_name", &bfe::get_name, pygmo::bfe_get_name_docstring().c_str())
        .def("get_extra_info", &bfe::get_extra_info, pygmo::bfe_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &bfe::get_thread_safety, pygmo::bfe_get_thread_safety_docstring().c_str());

    // Expose bfes.
    pygmo::expose_bfes();

    // Topology class.
    pygmo::topology_ptr
        = detail::make_unique<bp::class_<topology>>("topology", pygmo::topology_docstring().c_str(), bp::init<>());
    auto &topology_class = pygmo::get_topology_class();
    topology_class.def(bp::init<const bp::object &>((bp::arg("udt"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::topology_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<topology>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<topology>)
        // UDT extraction.
        .def("_py_extract", &pygmo::generic_py_extract<topology>)
        // Topology methods.
        .def("get_connections", lcast([](const topology &t, std::size_t n) -> bp::tuple {
                 auto ret = t.get_connections(n);
                 return bp::make_tuple(pygmo::vector_to_ndarr(ret.first), pygmo::vector_to_ndarr(ret.second));
             }),
             pygmo::topology_get_connections_docstring().c_str(), (bp::arg("prob"), bp::arg("dvs")))
        .def("push_back", lcast([](topology &t, unsigned n) { t.push_back(n); }),
             pygmo::topology_push_back_docstring().c_str(), (bp::arg("n") = std::size_t(1)))
        .def("get_name", &topology::get_name, pygmo::topology_get_name_docstring().c_str())
        .def("get_extra_info", &topology::get_extra_info, pygmo::topology_get_extra_info_docstring().c_str());

    // Expose topologies.
    pygmo::expose_topologies();

    // Replacement policy class.
    pygmo::r_policy_ptr
        = detail::make_unique<bp::class_<r_policy>>("r_policy", pygmo::r_policy_docstring().c_str(), bp::init<>());
    auto &r_policy_class = pygmo::get_r_policy_class();
    r_policy_class.def(bp::init<const bp::object &>((bp::arg("udrp"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::r_policy_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<r_policy>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<r_policy>)
        // UDRP extraction.
        .def("_py_extract", &pygmo::generic_py_extract<r_policy>)
        // r_policy methods.
        .def("replace",
             lcast([](const r_policy &r, const bp::object &inds, const vector_double::size_type &nx,
                      const vector_double::size_type &nix, const vector_double::size_type &nobj,
                      const vector_double::size_type &nec, const vector_double::size_type &nic, const bp::object &tol,
                      const bp::object &mig) -> bp::tuple {
                 auto ret = r.replace(pygmo::obj_to_inds(inds), nx, nix, nobj, nec, nic,
                                      pygmo::obj_to_vector<vector_double>(tol), pygmo::obj_to_inds(mig));
                 return pygmo::inds_to_tuple(ret);
             }),
             pygmo::r_policy_replace_docstring().c_str(),
             (bp::arg("inds"), bp::arg("nx"), bp::arg("nix"), bp::arg("nobj"), bp::arg("nec"), bp::arg("nic"),
              bp::arg("tol"), bp::arg("mig")))
        .def("get_name", &r_policy::get_name, pygmo::r_policy_get_name_docstring().c_str())
        .def("get_extra_info", &r_policy::get_extra_info, pygmo::r_policy_get_extra_info_docstring().c_str());

    // Expose r_policies.
    pygmo::expose_r_policies();

    // Selection policy class.
    pygmo::s_policy_ptr
        = detail::make_unique<bp::class_<s_policy>>("s_policy", pygmo::s_policy_docstring().c_str(), bp::init<>());
    auto &s_policy_class = pygmo::get_s_policy_class();
    s_policy_class.def(bp::init<const bp::object &>((bp::arg("udsp"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::s_policy_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<s_policy>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<s_policy>)
        // UDSP extraction.
        .def("_py_extract", &pygmo::generic_py_extract<s_policy>)
        // s_policy methods.
        .def("select",
             lcast([](const s_policy &s, const bp::object &inds, const vector_double::size_type &nx,
                      const vector_double::size_type &nix, const vector_double::size_type &nobj,
                      const vector_double::size_type &nec, const vector_double::size_type &nic,
                      const bp::object &tol) -> bp::tuple {
                 auto ret = s.select(pygmo::obj_to_inds(inds), nx, nix, nobj, nec, nic,
                                     pygmo::obj_to_vector<vector_double>(tol));
                 return pygmo::inds_to_tuple(ret);
             }),
             pygmo::s_policy_select_docstring().c_str(),
             (bp::arg("inds"), bp::arg("nx"), bp::arg("nix"), bp::arg("nobj"), bp::arg("nec"), bp::arg("nic"),
              bp::arg("tol")))
        .def("get_name", &s_policy::get_name, pygmo::s_policy_get_name_docstring().c_str())
        .def("get_extra_info", &s_policy::get_extra_info, pygmo::s_policy_get_extra_info_docstring().c_str());

    // Expose s_policies.
    pygmo::expose_s_policies();
}
