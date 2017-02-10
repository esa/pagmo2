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
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/def.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/docstring_options.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/self.hpp>
#include <boost/python/tuple.hpp>
#include <memory>
#include <sstream>
#include <string>

#include <pagmo/algorithm.hpp>
#ifdef PAGMO_WITH_EIGEN3
#include <pagmo/algorithms/cmaes.hpp>
#endif

#include <pagmo/algorithms/compass_search.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/algorithms/moead.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/ackley.hpp>
#if !defined(_MSC_VER)
#include <pagmo/problems/cec2013.hpp>
#endif
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/griewank.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>

#include "algorithm.hpp"
#include "algorithm_exposition_suite.hpp"
#include "common_utils.hpp"
#include "docstrings.hpp"
#include "numpy.hpp"
#include "object_serialization.hpp"
#include "problem.hpp"
#include "problem_exposition_suite.hpp"
#include "pygmo_classes.hpp"

#if defined(_MSC_VER)

#pragma warning(pop)

#endif

// Implementation of std::make_unique:
// http://stackoverflow.com/questions/17902405/how-to-implement-make-unique-function-in-c11
template <typename T, typename... Args>
static inline std::unique_ptr<T> make_unique(Args &&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

namespace bp = boost::python;
using namespace pagmo;

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

// A pickle suite for pagmo::null_problem. The problem pickle suite
// uses null_problem for the initialization of a problem instance,
// and the initialization argument returned by getinitargs
// must be serializable itself.
struct null_problem_pickle_suite : bp::pickle_suite {
    static bp::tuple getinitargs(const null_problem &)
    {
        return bp::make_tuple();
    }
};

// Same as above for the null algo.
struct null_algorithm_pickle_suite : bp::pickle_suite {
    static bp::tuple getinitargs(const null_algorithm &)
    {
        return bp::make_tuple();
    }
};

// Instances of the classes in pygmo_classes.hpp.
namespace pygmo
{

// Problem and meta-problem classes.
std::unique_ptr<bp::class_<problem>> problem_ptr(nullptr);
std::unique_ptr<bp::class_<translate>> translate_ptr(nullptr);
std::unique_ptr<bp::class_<decompose>> decompose_ptr(nullptr);

// Algorithm and meta-algorithm classes.
std::unique_ptr<bp::class_<algorithm>> algorithm_ptr(nullptr);
}

// The cleanup function.
// This function will be registered to be called when the pygmo core module is unloaded
// (see the __init__.py file). I am not 100% sure it is needed to reset these global
// variables, but it makes me nervous to have global boost python objects around on shutdown.
static inline void cleanup()
{
    pygmo::problem_ptr.reset();
    pygmo::translate_ptr.reset();
    pygmo::decompose_ptr.reset();

    pygmo::algorithm_ptr.reset();
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

// Various wrappers for the population exposition.

// Expose a population constructor from problem.
template <typename Prob>
static inline void population_prob_init(bp::class_<population> &pop_class)
{
    pop_class.def(bp::init<const Prob &, population::size_type>())
        .def(bp::init<const Prob &, population::size_type, unsigned>());
}

// Various best_idx() overloads.
static inline vector_double::size_type pop_best_idx_wrapper_0(const population &pop, const bp::object &tol)
{
    return pop.best_idx(pygmo::to_vd(tol));
}

static inline vector_double::size_type pop_best_idx_wrapper_1(const population &pop, double tol)
{
    return pop.best_idx(tol);
}

static inline vector_double::size_type pop_best_idx_wrapper_2(const population &pop)
{
    return pop.best_idx();
}

// Various worst_idx() overloads.
static inline vector_double::size_type pop_worst_idx_wrapper_0(const population &pop, const bp::object &tol)
{
    return pop.worst_idx(pygmo::to_vd(tol));
}

static inline vector_double::size_type pop_worst_idx_wrapper_1(const population &pop, double tol)
{
    return pop.worst_idx(tol);
}

static inline vector_double::size_type pop_worst_idx_wrapper_2(const population &pop)
{
    return pop.worst_idx();
}

// set_xf().
static inline void pop_set_xf_wrapper(population &pop, population::size_type i, const bp::object &x,
                                      const bp::object &f)
{
    pop.set_xf(i, pygmo::to_vd(x), pygmo::to_vd(f));
}

// set_x().
static inline void pop_set_x_wrapper(population &pop, population::size_type i, const bp::object &x)
{
    pop.set_x(i, pygmo::to_vd(x));
}

// get_f().
static inline bp::object pop_get_f_wrapper(const population &pop)
{
    return pygmo::vv_to_a(pop.get_f());
}

// get_x().
static inline bp::object pop_get_x_wrapper(const population &pop)
{
    return pygmo::vv_to_a(pop.get_x());
}

// get_ID().
static inline bp::object pop_get_ID_wrapper(const population &pop)
{
    return pygmo::v_to_a(pop.get_ID());
}

// Decompose methods wrappers
static inline bp::object decompose_decompose_fitness_wrapper(const pagmo::decompose &p, const bp::object &f,
                                                             const bp::object &weights, const bp::object &z_ref)
{
    return pygmo::v_to_a(p.decompose_fitness(pygmo::to_vd(f), pygmo::to_vd(weights), pygmo::to_vd(z_ref)));
}

// ZDT wrappers.
static inline double zdt_p_distance_wrapper(const zdt &z, const bp::object &x)
{
    return z.p_distance(pygmo::to_vd(x));
}

// DE1220 ctors.
static inline de1220 *de1220_init_0(unsigned gen, const bp::object &allowed_variants, unsigned variant_adptv,
                                    double ftol, double xtol, bool memory)
{
    return ::new de1220(gen, pygmo::to_vu(allowed_variants), variant_adptv, ftol, xtol, memory);
}

static inline de1220 *de1220_init_1(unsigned gen, const bp::object &allowed_variants, unsigned variant_adptv,
                                    double ftol, double xtol, bool memory, unsigned seed)
{
    return ::new de1220(gen, pygmo::to_vu(allowed_variants), variant_adptv, ftol, xtol, memory, seed);
}

static inline bp::list de1220_allowed_variants()
{
    bp::list retval;
    for (const auto &n : de1220_statics<void>::allowed_variants) {
        retval.append(n);
    }
    return retval;
}

// Wrappers for utils/multi_objective stuff
// fast_non_dominated_sorting
static inline bp::object fast_non_dominated_sorting_wrapper(const bp::object &x)
{
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
    return bp::make_tuple(ndf_py, dl_py, pygmo::v_to_a(std::get<2>(fnds)), pygmo::v_to_a(std::get<3>(fnds)));
}

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

// A test problem.
struct test_problem {
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
    int m_n = 1;
};

BOOST_PYTHON_MODULE(core)
{
    // Setup doc options
    bp::docstring_options doc_options;
    doc_options.enable_all();
    doc_options.disable_cpp_signatures();
    doc_options.disable_py_signatures();

    // Init numpy.
    // NOTE: only the second import is strictly necessary. We run a first import from BP
    // because that is the easiest way to detect whether numpy is installed or not (rather
    // than trying to figure out a way to detect it from wrap_import_array()).
    // NOTE: if we split the module in multiple C++ files, we need to take care of importing numpy
    // from every extension file and also defining PY_ARRAY_UNIQUE_SYMBOL as explained here:
    // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    try {
        bp::import("numpy.core.multiarray");
    } catch (...) {
        pygmo::builtin().attr("print")(
            u8"\033[91m====ERROR====\nThe NumPy module could not be imported. "
            u8"Please make sure that NumPy has been correctly installed.\n====ERROR====\033[0m");
        pygmo_throw(PyExc_ImportError, "");
    }
    wrap_import_array();

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

    // Population class.
    bp::class_<population> pop_class("population", pygmo::population_docstring().c_str(), bp::no_init);
    // Ctor from problem.
    population_prob_init<problem>(pop_class);
    pop_class
        .def(repr(bp::self))
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<population>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<population>)
        .def_pickle(population_pickle_suite())
        .def("push_back", +[](population &pop, const bp::object &x) { pop.push_back(pygmo::to_vd(x)); },
             pygmo::population_push_back_docstring().c_str(), (bp::arg("x")))
        .def("decision_vector", +[](const population &pop) { return pygmo::v_to_a(pop.random_decision_vector()); },
             pygmo::population_decision_vector_docstring().c_str())
        .add_property("champion_x", +[](const population &pop) { return pygmo::v_to_a(pop.champion_x()); })
        .add_property("champion_f", +[](const population &pop) { return pygmo::v_to_a(pop.champion_f()); })
        .def("best_idx", &pop_best_idx_wrapper_0)
        .def("best_idx", &pop_best_idx_wrapper_1)
        .def("best_idx", &pop_best_idx_wrapper_2, pygmo::population_best_idx_docstring().c_str())
        .def("worst_idx", &pop_worst_idx_wrapper_0)
        .def("worst_idx", &pop_worst_idx_wrapper_1)
        .def("worst_idx", &pop_worst_idx_wrapper_2, pygmo::population_worst_idx_docstring().c_str())
        .def("size", &population::size, pygmo::population_size_docstring().c_str())
        .def("__len__", &population::size)
        .def("set_xf", &pop_set_xf_wrapper, pygmo::population_set_xf_docstring().c_str())
        .def("set_x", &pop_set_x_wrapper, pygmo::population_set_x_docstring().c_str())
        .def("set_problem_seed", &population::set_problem_seed, pygmo::population_set_problem_seed_docstring().c_str(),
             (bp::arg("seed")))
        .def("get_problem", &population::get_problem, pygmo::population_get_problem_docstring().c_str(),
             bp::return_value_policy<bp::copy_const_reference>())
        .def("get_f", &pop_get_f_wrapper, pygmo::population_get_f_docstring().c_str())
        .def("get_x", &pop_get_x_wrapper, pygmo::population_get_x_docstring().c_str())
        .def("get_ID", &pop_get_ID_wrapper, pygmo::population_get_ID_docstring().c_str())
        .def("get_seed", &population::get_seed, pygmo::population_get_seed_docstring().c_str());

    // Problem class.
    pygmo::problem_ptr = make_unique<bp::class_<problem>>("problem", pygmo::problem_docstring().c_str(), bp::no_init);
    auto &problem_class = *pygmo::problem_ptr;
    problem_class.def(bp::init<const bp::object &>((bp::arg("prob"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::problem_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<problem>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<problem>)
        // Problem extraction.
        .def("_py_extract", &pygmo::generic_py_extract<problem>)
        // Problem methods.
        .def("fitness",
             +[](const pagmo::problem &p, const bp::object &dv) { return pygmo::v_to_a(p.fitness(pygmo::to_vd(dv))); },
             pygmo::problem_fitness_docstring().c_str(), (bp::arg("dv")))
        .def("get_bounds",
             +[](const pagmo::problem &p) -> bp::tuple {
                 auto retval = p.get_bounds();
                 return bp::make_tuple(pygmo::v_to_a(retval.first), pygmo::v_to_a(retval.second));
             },
             pygmo::problem_get_bounds_docstring().c_str())
        .def("gradient",
             +[](const pagmo::problem &p, const bp::object &dv) { return pygmo::v_to_a(p.gradient(pygmo::to_vd(dv))); },
             pygmo::problem_gradient_docstring().c_str(), (bp::arg("dv")))
        .def("has_gradient", &problem::has_gradient, pygmo::problem_has_gradient_docstring().c_str())
        .def("gradient_sparsity", +[](const pagmo::problem &p) { return pygmo::sp_to_a(p.gradient_sparsity()); },
             pygmo::problem_gradient_sparsity_docstring().c_str())
        .def("has_gradient_sparsity", &problem::has_gradient_sparsity,
             pygmo::problem_has_gradient_sparsity_docstring().c_str())
        .def("hessians",
             +[](const pagmo::problem &p, const bp::object &dv) -> bp::list {
                 bp::list retval;
                 const auto h = p.hessians(pygmo::to_vd(dv));
                 for (const auto &v : h) {
                     retval.append(pygmo::v_to_a(v));
                 }
                 return retval;
             },
             pygmo::problem_hessians_docstring().c_str(), (bp::arg("dv")))
        .def("has_hessians", &problem::has_hessians, pygmo::problem_has_hessians_docstring().c_str())
        .def("hessians_sparsity",
             +[](const pagmo::problem &p) -> bp::list {
                 bp::list retval;
                 const auto hs = p.hessians_sparsity();
                 for (const auto &sp : hs) {
                     retval.append(pygmo::sp_to_a(sp));
                 }
                 return retval;
             },
             pygmo::problem_hessians_sparsity_docstring().c_str())
        .def("has_hessians_sparsity", &problem::has_hessians_sparsity,
             pygmo::problem_has_hessians_sparsity_docstring().c_str())
        .def("get_nobj", &problem::get_nobj, pygmo::problem_get_nobj_docstring().c_str())
        .def("get_nx", &problem::get_nx, pygmo::problem_get_nx_docstring().c_str())
        .def("get_nf", &problem::get_nf, pygmo::problem_get_nf_docstring().c_str())
        .def("get_nec", &problem::get_nec, pygmo::problem_get_nec_docstring().c_str())
        .def("get_nic", &problem::get_nic, pygmo::problem_get_nic_docstring().c_str())
        .def("get_nc", &problem::get_nc, pygmo::problem_get_nc_docstring().c_str())
        .add_property("c_tol", +[](const problem &prob) { return pygmo::v_to_a(prob.get_c_tol()); },
                      +[](problem &prob, const bp::object &c_tol) { prob.set_c_tol(pygmo::to_vd(c_tol)); },
                      pygmo::problem_c_tol_docstring().c_str())
        .def("get_fevals", &problem::get_fevals, pygmo::problem_get_fevals_docstring().c_str())
        .def("get_gevals", &problem::get_gevals, pygmo::problem_get_gevals_docstring().c_str())
        .def("get_hevals", &problem::get_hevals, pygmo::problem_get_hevals_docstring().c_str())
        .def("set_seed", &problem::set_seed, pygmo::problem_set_seed_docstring().c_str(), (bp::arg("seed")))
        .def("has_set_seed", &problem::has_set_seed, pygmo::problem_has_set_seed_docstring().c_str())
        .def("is_stochastic", &problem::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.core.problem.has_set_seed()`.")
        .def("feasibility_x", +[](const problem &p, const bp::object &x) { return p.feasibility_x(pygmo::to_vd(x)); },
             pygmo::problem_feasibility_x_docstring().c_str())
        .def("feasibility_f", +[](const problem &p, const bp::object &f) { return p.feasibility_f(pygmo::to_vd(f)); },
             pygmo::problem_feasibility_f_docstring().c_str())
        .def("get_name", &problem::get_name, "Get problem's name.")
        .def("get_extra_info", &problem::get_extra_info, "Get problem's extra info.");

    // Algorithm class.
    pygmo::algorithm_ptr
        = make_unique<bp::class_<algorithm>>("algorithm", pygmo::algorithm_docstring().c_str(), bp::no_init);
    auto &algorithm_class = *pygmo::algorithm_ptr;
    algorithm_class.def(bp::init<const bp::object &>((bp::arg("a"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::algorithm_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<algorithm>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<algorithm>)
        // Algorithm extraction.
        .def("_py_extract", &pygmo::generic_py_extract<algorithm>)
        // Algorithm methods.
        .def("evolve", &algorithm::evolve, "evolve(pop)\n\nEvolve population.\n\n:param pop: the population to evolve\n"
                                           ":type pop: :class:`pygmo.core.population`\n"
                                           ":returns: the evolved population\n"
                                           ":rtype: :class:`pygmo.core.population`\n\n",
             (bp::arg("pop")))
        .def("set_seed", &algorithm::set_seed,
             "set_seed(seed)\n\nSet algorithm seed.\n\n:param seed: the desired seed\n:type seed: ``int``\n"
             ":raises: :exc:`RuntimeError` if the user-defined algorithm does not support seed setting\n"
             ":raises: :exc:`OverflowError` if *seed* is negative or too large\n\n",
             (bp::arg("seed")))
        .def("has_set_seed", &algorithm::has_set_seed,
             "has_set_seed()\n\nDetect the presence of the ``set_seed()`` method in the user-defined algorithm.\n\n"
             ":returns: ``True`` if the user-defined algorithm has the ability of setting a random seed, ``False`` "
             "otherwise\n"
             ":rtype: ``bool``\n\n")
        .def("set_verbosity", &algorithm::set_verbosity,
             "set_verbosity(level)\n\nSet algorithm verbosity.\n\n:param level: the desired verbosity level\n:type "
             "level: ``int``\n"
             ":raises: :exc:`RuntimeError` if the user-defined algorithm does not support verbosity setting\n"
             ":raises: :exc:`OverflowError` if *level* is negative or too large\n\n",
             (bp::arg("level")))
        .def("has_set_verbosity", &algorithm::has_set_verbosity,
             "has_set_verbosity()\n\nDetect the presence of the ``set_verbosity()`` method in the user-defined "
             "algorithm.\n\n"
             ":returns: ``True`` if the user-defined algorithm has the ability of setting a verbosity level, ``False`` "
             "otherwise\n"
             ":rtype: ``bool``\n\n")
        .def("is_stochastic", &algorithm::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.core.algorithm.has_set_seed`.")
        .def("get_name", &algorithm::get_name, "Get algorithm's name.")
        .def("get_extra_info", &algorithm::get_extra_info, "Get algorithm's extra info.");

    // Translate meta-problem.
    pygmo::translate_ptr = make_unique<bp::class_<translate>>(
        "translate", "The translate meta-problem.\n\nBlah blah blah blah.\n\nAdditional constructors:", bp::init<>());
    auto &tp = *pygmo::translate_ptr;
    // Constructor from Python user-defined problem and translation vector (allows to translate Python problems).
    tp.def("__init__", pygmo::make_translate_init<bp::object>())
        // Constructor of translate from translate and translation vector. This allows to apply the
        // translation multiple times.
        .def("__init__", pygmo::make_translate_init<translate>())
        // Problem extraction.
        .def("_py_extract", &pygmo::generic_py_extract<translate>)
        .def("_cpp_extract", &pygmo::generic_cpp_extract<translate, translate>, bp::return_internal_reference<>());
    // Mark it as a cpp problem.
    tp.attr("_pygmo_cpp_problem") = true;
    // Ctor of problem from translate.
    pygmo::problem_prob_init<translate>();
    // Extract a translated problem from the problem class.
    problem_class.def("_cpp_extract", &pygmo::generic_cpp_extract<problem, translate>,
                      bp::return_internal_reference<>());
    // Add it to the the problems submodule.
    bp::scope().attr("problems").attr("translate") = tp;

    // Decompose meta-problem.
    pygmo::decompose_ptr
        = make_unique<bp::class_<decompose>>("decompose", "The decompose meta-problem.\n\n", bp::init<>());
    auto &dp = *pygmo::decompose_ptr;
    // Constructor from Python user-defined problem.
    dp.def("__init__", pygmo::make_decompose_init<bp::object>())
        // Problem extraction.
        .def("_py_extract", &pygmo::generic_py_extract<decompose>)
        // Returns the decomposed fitness with an arbitrary weight and reference point
        .def("decompose_fitness", &decompose_decompose_fitness_wrapper,
             pygmo::decompose_decompose_fitness_docstring().c_str(),
             (bp::arg("f"), bp::arg("weights"), bp::arg("ref_point")));
    // Mark it as a cpp problem.
    dp.attr("_pygmo_cpp_problem") = true;
    // Ctor of problem from decompose.
    pygmo::problem_prob_init<decompose>();
    // Extract a decomposed problem from the problem class.
    problem_class.def("_cpp_extract", &pygmo::generic_cpp_extract<problem, decompose>,
                      bp::return_internal_reference<>());
    // Add it to the problems submodule.
    bp::scope().attr("problems").attr("decompose") = dp;

    // Before moving to the user-defined C++ problems, we need to expose the interoperability between
    // meta-problems.
    // Construct translate from decompose.
    tp.def("__init__", pygmo::make_translate_init<decompose>());
    // Extract decompose from translate.
    tp.def("_cpp_extract", &pygmo::generic_cpp_extract<translate, decompose>, bp::return_internal_reference<>());
    // Construct decompose from translate.
    dp.def("__init__", pygmo::make_decompose_init<translate>());
    // Extract translate from decompose.
    dp.def("_cpp_extract", &pygmo::generic_cpp_extract<decompose, translate>, bp::return_internal_reference<>());

    // Exposition of C++ problems.
    // Test problem.
    auto test_p = pygmo::expose_problem<test_problem>("_test_problem", "A test problem.");
    test_p.def("get_n", &test_problem::get_n);
    test_p.def("set_n", &test_problem::set_n);
    // Null problem.
    auto np = pygmo::expose_problem<null_problem>(
        "null_problem",
        "__init__()\n\nThe null problem.\n\nA problem used only in the initialization of meta-problems.\n\n");
    // NOTE: this is needed only for the null_problem, as it is used in the implementation of the
    // serialization of the problem. Not necessary for any other problem type.
    // NOTE: this is needed because problem does not have a def ctor.
    np.def_pickle(null_problem_pickle_suite());
    // Rosenbrock.
    auto rb = pygmo::expose_problem<rosenbrock>("rosenbrock", pygmo::rosenbrock_docstring().c_str());
    rb.def(bp::init<unsigned>((bp::arg("dim"))));
    rb.def("best_known", &pygmo::best_known_wrapper<rosenbrock>,
           pygmo::problem_get_best_docstring("Rosenbrock").c_str());
    // Hock-Schittkowsky 71
    auto hs71 = pygmo::expose_problem<hock_schittkowsky_71>("hock_schittkowsky_71",
                                                            "__init__()\n\nThe Hock-Schittkowsky 71 problem.\n\n"
                                                            "See :cpp:class:`pagmo::hock_schittkowsky_71`.\n\n");
    hs71.def("best_known", &pygmo::best_known_wrapper<hock_schittkowsky_71>,
             pygmo::problem_get_best_docstring("Hock-Schittkowsky 71").c_str());
    // Rastrigin.
    auto rastr = pygmo::expose_problem<rastrigin>("rastrigin", "__init__(dim = 1)\n\nThe Rastrigin problem.\n\n"
                                                               "See :cpp:class:`pagmo::rastrigin`.\n\n");
    rastr.def(bp::init<unsigned>((bp::arg("dim"))));
    rastr.def("best_known", &pygmo::best_known_wrapper<rastrigin>,
              pygmo::problem_get_best_docstring("Rastrigin").c_str());
    // Schwefel.
    auto sch = pygmo::expose_problem<schwefel>("schwefel", "__init__(dim = 1)\n\nThe Schwefel problem.\n\n"
                                                           "See :cpp:class:`pagmo::schwefel`.\n\n");
    sch.def(bp::init<unsigned>((bp::arg("dim"))));
    sch.def("best_known", &pygmo::best_known_wrapper<schwefel>, pygmo::problem_get_best_docstring("Schwefel").c_str());
    // Ackley.
    auto ack = pygmo::expose_problem<ackley>("ackley", "__init__(dim = 1)\n\nThe Ackley problem.\n\n"
                                                       "See :cpp:class:`pagmo::ackley`.\n\n");
    ack.def(bp::init<unsigned>((bp::arg("dim"))));
    ack.def("best_known", &pygmo::best_known_wrapper<ackley>, pygmo::problem_get_best_docstring("Ackley").c_str());
    // Griewank.
    auto griew = pygmo::expose_problem<griewank>("griewank", "__init__(dim = 1)\n\nThe Griewank problem.\n\n"
                                                             "See :cpp:class:`pagmo::griewank`.\n\n");
    griew.def(bp::init<unsigned>((bp::arg("dim"))));
    griew.def("best_known", &pygmo::best_known_wrapper<griewank>,
              pygmo::problem_get_best_docstring("Griewank").c_str());
    // ZDT.
    auto zdt_p = pygmo::expose_problem<zdt>("zdt", "__init__(id = 1, param = 30)\n\nThe ZDT problem.\n\n"
                                                   "See :cpp:class:`pagmo::zdt`.\n\n");
    zdt_p.def(bp::init<unsigned, unsigned>((bp::arg("id") = 1u, bp::arg("param") = 30u)));
    zdt_p.def("p_distance", &zdt_p_distance_wrapper);
    // Inventory.
    auto inv = pygmo::expose_problem<inventory>(
        "inventory", "__init__(weeks = 4,sample_size = 10,seed = random)\n\nThe inventory problem.\n\n"
                     "See :cpp:class:`pagmo::inventory`.\n\n");
    inv.def(bp::init<unsigned, unsigned>((bp::arg("weeks") = 4u, bp::arg("sample_size") = 10u)));
    inv.def(
        bp::init<unsigned, unsigned, unsigned>((bp::arg("weeks") = 4u, bp::arg("sample_size") = 10u, bp::arg("seed"))));
// excluded in MSVC
#if !defined(_MSC_VER)
    // CEC 2013.
    auto cec2013_ = pygmo::expose_problem<cec2013>("cec2013", pygmo::cec2013_docstring().c_str());
    cec2013_.def(bp::init<unsigned, unsigned>((bp::arg("prob_id") = 1, bp::arg("dim") = 2)));
#endif
    // Exposition of C++ algorithms.
    // Null algo.
    auto na = pygmo::expose_algorithm<null_algorithm>("null_algorithm",
                                                      "__init__()\n\nThe null algorithm.\n\nA test algorithm.\n\n");
    // NOTE: this is needed only for the null_algorithm, as it is used in the implementation of the
    // serialization of the algorithm. Not necessary for any other algorithm type.
    na.def_pickle(null_algorithm_pickle_suite());
    // DE
    auto de_ = pygmo::expose_algorithm<de>("de", pygmo::de_docstring().c_str());
    de_.def(bp::init<unsigned int, double, double, unsigned int, double, double>(
        (bp::arg("gen") = 1u, bp::arg("F") = .8, bp::arg("CR") = .9, bp::arg("variant") = 2u, bp::arg("ftol") = 1e-6,
         bp::arg("tol") = 1E-6)));
    de_.def(bp::init<unsigned int, double, double, unsigned int, double, double, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("F") = .8, bp::arg("CR") = .9, bp::arg("variant") = 2u, bp::arg("ftol") = 1e-6,
         bp::arg("tol") = 1E-6, bp::arg("seed"))));
    pygmo::expose_algo_log(de_, pygmo::de_get_log_docstring().c_str());
    de_.def("get_seed", &de::get_seed);
    // COMPASS SEARCH
    auto compass_search_
        = pygmo::expose_algorithm<compass_search>("compass_search", pygmo::compass_search_docstring().c_str());
    compass_search_.def(
        bp::init<unsigned int, double, double, double>((bp::arg("max_fevals") = 1u, bp::arg("start_range") = .1,
                                                        bp::arg("stop_range") = .01, bp::arg("reduction_coeff") = .5)));
    pygmo::expose_algo_log(compass_search_, pygmo::compass_search_get_log_docstring().c_str());
    compass_search_.def("get_max_fevals", &compass_search::get_max_fevals);
    compass_search_.def("get_start_range", &compass_search::get_start_range);
    compass_search_.def("get_stop_range", &compass_search::get_stop_range);
    compass_search_.def("get_reduction_coeff", &compass_search::get_reduction_coeff);
    compass_search_.def("get_verbosity", &compass_search::get_verbosity);
    compass_search_.def("set_verbosity", &compass_search::set_verbosity);
    // PSO
    auto pso_ = pygmo::expose_algorithm<pso>("pso", pygmo::pso_docstring().c_str());
    pso_.def(bp::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool>(
        (bp::arg("gen") = 1u, bp::arg("omega") = 0.7298, bp::arg("eta1") = 2.05, bp::arg("eta2") = 2.05,
         bp::arg("max_vel") = 0.5, bp::arg("variant") = 5u, bp::arg("neighb_type") = 2u, bp::arg("neighb_param") = 4u,
         bp::arg("memory") = false)));
    pso_.def(bp::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("omega") = 0.7298, bp::arg("eta1") = 2.05, bp::arg("eta2") = 2.05,
         bp::arg("max_vel") = 0.5, bp::arg("variant") = 5u, bp::arg("neighb_type") = 2u, bp::arg("neighb_param") = 4u,
         bp::arg("memory") = false, bp::arg("seed"))));
    pygmo::expose_algo_log(pso_, pygmo::pso_get_log_docstring().c_str());
    pso_.def("get_seed", &pso::get_seed);
    // SEA
    auto sea_ = pygmo::expose_algorithm<sea>("sea", "__init__(gen = 1, seed = random)\n\n"
                                                    "(N+1)-ES simple evolutionary algorithm.\n\n");
    sea_.def(bp::init<unsigned>((bp::arg("gen") = 1u)));
    sea_.def(bp::init<unsigned, unsigned>((bp::arg("gen") = 1u, bp::arg("seed"))));
    pygmo::expose_algo_log(sea_, "");
    sea_.def("get_seed", &sea::get_seed);
    // SIMULATED ANNEALING
    auto simulated_annealing_ = pygmo::expose_algorithm<simulated_annealing>(
        "simulated_annealing", pygmo::simulated_annealing_docstring().c_str());
    simulated_annealing_.def(bp::init<double, double, unsigned, unsigned, unsigned, double>(
        (bp::arg("Ts") = 10., bp::arg("Tf") = 0.1, bp::arg("n_T_adj") = 10u, bp::arg("n_range_adj") = 1u,
         bp::arg("bin_size") = 20u, bp::arg("start_range") = 1.)));
    simulated_annealing_.def(bp::init<double, double, unsigned, unsigned, unsigned, double, unsigned>(
        (bp::arg("Ts") = 10., bp::arg("Tf") = 0.1, bp::arg("n_T_adj") = 10u, bp::arg("n_range_adj") = 10u,
         bp::arg("bin_size") = 10u, bp::arg("start_range") = 1., bp::arg("seed"))));
    pygmo::expose_algo_log(simulated_annealing_, pygmo::simulated_annealing_get_log_docstring().c_str());
    simulated_annealing_.def("get_seed", &simulated_annealing::get_seed);
    // SADE
    auto sade_ = pygmo::expose_algorithm<sade>("sade", pygmo::sade_docstring().c_str());
    sade_.def(bp::init<unsigned, unsigned, unsigned, double, double, bool>(
        (bp::arg("gen") = 1u, bp::arg("variant") = 2u, bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6,
         bp::arg("xtol") = 1e-6, bp::arg("memory") = false)));
    sade_.def(bp::init<unsigned, unsigned, unsigned, double, double, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("variant") = 2u, bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6,
         bp::arg("xtol") = 1e-6, bp::arg("memory") = false, bp::arg("seed"))));
    pygmo::expose_algo_log(sade_, pygmo::sade_get_log_docstring().c_str());
    sade_.def("get_seed", &sade::get_seed);
    // DE-1220
    auto de1220_ = pygmo::expose_algorithm<de1220>("de1220", pygmo::de1220_docstring().c_str());
    de1220_.def("__init__",
                bp::make_constructor(&de1220_init_0, bp::default_call_policies(),
                                     (bp::arg("gen") = 1u, bp::arg("allowed_variants") = de1220_allowed_variants(),
                                      bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6,
                                      bp::arg("memory") = false)));
    de1220_.def("__init__",
                bp::make_constructor(&de1220_init_1, bp::default_call_policies(),
                                     (bp::arg("gen") = 1u, bp::arg("allowed_variants") = de1220_allowed_variants(),
                                      bp::arg("variant_adptv") = 1u, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6,
                                      bp::arg("memory") = false, bp::arg("seed"))));
    pygmo::expose_algo_log(de1220_, pygmo::de1220_get_log_docstring().c_str());
    de1220_.def("get_seed", &de1220::get_seed);
// CMA-ES
#ifdef PAGMO_WITH_EIGEN3
    auto cmaes_ = pygmo::expose_algorithm<cmaes>("cmaes", pygmo::cmaes_docstring().c_str());
    cmaes_.def(bp::init<unsigned, double, double, double, double, double, double, double, bool>(
        (bp::arg("gen") = 1u, bp::arg("cc") = -1., bp::arg("cs") = -1., bp::arg("c1") = -1., bp::arg("cmu") = -1.,
         bp::arg("sigma0") = 0.5, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false)));
    cmaes_.def(bp::init<unsigned, double, double, double, double, double, double, double, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("cc") = -1., bp::arg("cs") = -1., bp::arg("c1") = -1., bp::arg("cmu") = -1.,
         bp::arg("sigma0") = 0.5, bp::arg("ftol") = 1e-6, bp::arg("xtol") = 1e-6, bp::arg("memory") = false,
         bp::arg("seed"))));
    pygmo::expose_algo_log(cmaes_, pygmo::cmaes_get_log_docstring().c_str());
    cmaes_.def("get_seed", &cmaes::get_seed);
#endif
    // MOEA/D - DE
    auto moead_ = pygmo::expose_algorithm<moead>("moead", pygmo::moead_docstring().c_str());
    moead_.def(bp::init<unsigned, std::string, unsigned, double, double, double, double, unsigned, bool>(
        (bp::arg("gen") = 1u, bp::arg("weight_generation") = "grid", bp::arg("neighbours") = 20u, bp::arg("CR") = 1.,
         bp::arg("F") = 0.5, bp::arg("eta_m") = 20, bp::arg("realb") = 0.9, bp::arg("limit") = 2u,
         bp::arg("preserve_diversity") = true)));
    moead_.def(bp::init<unsigned, std::string, unsigned, double, double, double, double, unsigned, bool, unsigned>(
        (bp::arg("gen") = 1u, bp::arg("weight_generation") = "grid", bp::arg("neighbours") = 20u, bp::arg("CR") = 1.,
         bp::arg("F") = 0.5, bp::arg("eta_m") = 20, bp::arg("realb") = 0.9, bp::arg("limit") = 2u,
         bp::arg("preserve_diversity") = true, bp::arg("seed"))));
    // moead needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    moead_.def("get_log",
               +[](const moead &a) -> bp::list {
                   bp::list retval;
                   for (const auto &t : a.get_log()) {
                       retval.append(bp::make_tuple(std::get<0>(t), std::get<1>(t), std::get<2>(t),
                                                    pygmo::v_to_a(std::get<3>(t))));
                   }
                   return retval;
               },
               pygmo::moead_get_log_docstring().c_str());

    moead_.def("get_seed", &moead::get_seed);

    // Exposition of stand alone functions
    // Multi-objective utilities
    bp::def("fast_non_dominated_sorting", fast_non_dominated_sorting_wrapper,
            pygmo::fast_non_dominated_sorting_docstring().c_str(), boost::python::arg("points"));
}
