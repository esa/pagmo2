#include "python_includes.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/def.hpp>
#include <boost/python/docstring_options.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/import.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/self.hpp>
#include <boost/python/tuple.hpp>
#include <memory>
#include <sstream>
#include <string>

#include "../include/population.hpp"
#include "../include/problem.hpp"
#include "../include/problems/decompose.hpp"
#include "../include/problems/hock_schittkowsky_71.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/translate.hpp"
#include "../include/serialization.hpp"
#include "common_utils.hpp"
#include "docstrings.hpp"
#include "numpy.hpp"
#include "object_serialization.hpp"
#include "problem.hpp"
#include "problem_exposition_suite.hpp"
#include "pygmo_classes.hpp"

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
struct null_problem_pickle_suite : bp::pickle_suite
{
    static bp::tuple getinitargs(const null_problem &)
    {
        return bp::make_tuple();
    }
};

// Instances of the classes in pygmo_classes.hpp.
namespace pygmo
{

std::unique_ptr<bp::class_<problem>> problem_ptr(nullptr);
std::unique_ptr<bp::class_<translate>> translate_ptr(nullptr);
std::unique_ptr<bp::class_<decompose>> decompose_ptr(nullptr);

std::unique_ptr<bp::class_<population>> population_ptr(nullptr);

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

    pygmo::population_ptr.reset();
}

// Serialization support for the population class.
struct population_pickle_suite : bp::pickle_suite
{
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
        return bp::make_tuple(pygmo::make_bytes(s.data(),boost::numeric_cast<Py_ssize_t>(s.size())));
    }
    static void setstate(population &pop, bp::tuple state)
    {
        if (len(state) != 1) {
            pygmo_throw(PyExc_ValueError,"the state tuple must have a single element");
        }
        auto ptr = PyBytes_AsString(bp::object(state[0]).ptr());
        if (!ptr) {
            pygmo_throw(PyExc_TypeError,"a bytes object is needed to deserialize a population");
        }
        const auto size = len(state[0]);
        std::string s(ptr,ptr + size);
        std::istringstream iss;
        iss.str(s);
        {
        cereal::PortableBinaryInputArchive iarchive(iss);
        iarchive(pop);
        }
    }
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
        pygmo::builtin().attr("print")(u8"\033[91m====ERROR====\nThe NumPy module could not be imported. "
            u8"Please make sure that NumPy has been correctly installed.\n====ERROR====\033[0m");
        pygmo_throw(PyExc_ImportError,"");
    }
    wrap_import_array();

    // Expose utility functions for testing purposes.
    bp::def("_builtin",&pygmo::builtin);
    bp::def("_type",&pygmo::type);
    bp::def("_str",&pygmo::str);
    bp::def("_callable",&pygmo::callable);
    bp::def("_deepcopy",&pygmo::deepcopy);
    bp::def("_to_sp",&pygmo::to_sp);
    bp::def("_test_object_serialization",&test_object_serialization);

    // Expose cleanup function.
    bp::def("_cleanup",&cleanup);

    // Create the problems submodule.
	std::string problems_module_name = bp::extract<std::string>(bp::scope().attr("__name__") + ".problems");
	PyObject *problems_module_ptr = PyImport_AddModule(problems_module_name.c_str());
	if (!problems_module_ptr) {
		pygmo_throw(PyExc_RuntimeError,"error while creating the 'problems' submodule");
	}
	auto problems_module = bp::object(bp::handle<>(bp::borrowed(problems_module_ptr)));
	bp::scope().attr("problems") = problems_module;

    // Population class.
    pygmo::population_ptr = std::make_unique<bp::class_<population>>("population",pygmo::population_docstring().c_str(),bp::init<>());
    auto &pop_class = *pygmo::population_ptr;
    // Ctor from Python problem.
    pygmo::population_prob_init<bp::object>();
    pop_class.def(repr(bp::self))
        // Copy and deepcopy.
        .def("__copy__",&pygmo::generic_copy_wrapper<population>)
        .def("__deepcopy__",&pygmo::generic_deepcopy_wrapper<population>)
        .def_pickle(population_pickle_suite())
        ;

    // Problem class.
    pygmo::problem_ptr = std::make_unique<bp::class_<problem>>("problem",pygmo::problem_docstring().c_str(),bp::no_init);
    auto &problem_class = *pygmo::problem_ptr;
    problem_class.def(bp::init<const bp::object &>((bp::arg("p"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::problem_pickle_suite())
        // Copy and deepcopy.
        .def("__copy__",&pygmo::generic_copy_wrapper<problem>)
        .def("__deepcopy__",&pygmo::generic_deepcopy_wrapper<problem>)
        // Problem extraction.
        .def("_py_extract",&pygmo::generic_py_extract<problem>)
        .def("_cpp_extract",&pygmo::generic_cpp_extract<problem,translate>)
        // Problem methods.
        .def("fitness",&pygmo::fitness_wrapper,"Fitness.\n\nThis method will calculate the fitness of the input "
            "decision vector *dv*. The fitness is returned as a an array of doubles.",(bp::arg("dv")))
        .def("gradient",&pygmo::gradient_wrapper,"Gradient.\n\nThis method will calculate the gradient of the input "
            "decision vector *dv*. The gradient is returned as a an array of doubles.",(bp::arg("dv")))
        .def("has_gradient",&problem::has_gradient,"Gradient availability.")
        .def("gradient_sparsity",&pygmo::gradient_sparsity_wrapper,"Gradient sparsity.")
        .def("hessians",&pygmo::hessians_wrapper,"Hessians.\n\nThis method will calculate the Hessians of the input "
            "decision vector *dv*. The Hessians are returned as a list of arrays of doubles.",(bp::arg("dv")))
        .def("has_hessians",&problem::has_hessians,"Hessians availability.")
        .def("hessians_sparsity",&pygmo::hessians_sparsity_wrapper,"Hessians sparsity.")
        .def("get_nobj",&problem::get_nobj,"Get number of objectives.")
        .def("get_nx",&problem::get_nx,"Get problem dimension.")
        .def("get_nf",&problem::get_nf,"Get fitness dimension.")
        .def("get_bounds",&pygmo::get_bounds_wrapper,"Get bounds.\n\nThis method will return the problem bounds as a pair "
            "of arrays of doubles of equal length.")
        .def("get_nec",&problem::get_nec,"Get number of equality constraints.")
        .def("get_nic",&problem::get_nic,"Get number of inequality constraints.")
        .def("get_nc",&problem::get_nc,"Get total number of constraints.")
        .def("get_fevals",&problem::get_fevals,"Get total number of objective function evaluations.")
        .def("get_gevals",&problem::get_gevals,"Get total number of gradient evaluations.")
        .def("get_hevals",&problem::get_hevals,"Get total number of Hessians evaluations.")
        .def("set_seed",&problem::set_seed,"set_seed(seed)\n\nSet problem seed.\n\n:param seed: the desired seed\n:type seed: ``int``\n"
            ":raises: :exc:`RuntimeError` if the user-defined problem does not support seed setting\n"
            ":raises: :exc:`OverflowError` if *seed* is negative or too large\n\n",(bp::arg("seed")))
        .def("has_set_seed",&problem::has_set_seed,"has_set_seed()\n\nDetect the presence of the ``set_seed()`` method in the user-defined problem.\n\n"
            ":returns: ``True`` if the user-defined problem has the ability of setting a random seed, ``False`` otherwise\n"
            ":rtype: ``bool``\n\n")
        .def("is_stochastic",&problem::is_stochastic,"is_stochastic()\n\nAlias for :func:`~pygmo.core.problem.has_set_seed`.")
        .def("get_name",&problem::get_name,"Get problem's name.")
        .def("get_extra_info",&problem::get_extra_info,"Get problem's extra info.");

    // Translate meta-problem.
    pygmo::translate_ptr = std::make_unique<bp::class_<translate>>("translate",
        "The translate meta-problem.\n\nBlah blah blah blah.\n\nAdditional constructors:",bp::init<>());
    auto &tp = *pygmo::translate_ptr;
    // Constructor from Python concrete problem and translation vector (allows to translate Python problems).
    tp.def("__init__",bp::make_constructor(&pygmo::translate_init<bp::object>,boost::python::default_call_policies(),
        (bp::arg("p"),bp::arg("t"))))
        // Constructor of translate from translate and translation vector. This allows to apply the
        // translation multiple times.
        .def("__init__",bp::make_constructor(&pygmo::translate_init<translate>,boost::python::default_call_policies(),
            (bp::arg("p"),bp::arg("t"))))
        // Problem extraction.
        .def("_py_extract",&pygmo::generic_py_extract<translate>)
        .def("_cpp_extract",&pygmo::generic_cpp_extract<translate,translate>);
    // Mark it as a cpp problem.
    tp.attr("_pygmo_cpp_problem") = true;
    // Ctor of pop from translate.
    pygmo::population_prob_init<translate>();
    // Ctor of problem from translate.
    pygmo::problem_prob_init<translate>();
    // Add it the the problems submodule.
    bp::scope().attr("problems").attr("translate") = tp;

    // Exposition of C++ problems.
    // Null problem.
    auto np = pygmo::expose_problem<null_problem>("null_problem","__init__()\n\nThe null problem.\n\nA test problem.\n\n");
    // NOTE: this is needed only for the null_problem, as it is used in the implementation of the
    // serialization of the problem. Not necessary for any other problem type.
    np.def_pickle(null_problem_pickle_suite());
    // Rosenbrock.
    auto rb = pygmo::expose_problem<rosenbrock>("rosenbrock",pygmo::rosenbrock_docstring().c_str());
    rb.def(bp::init<unsigned>((bp::arg("dim"))));
    rb.def("best_known",&pygmo::best_known_wrapper<rosenbrock>,pygmo::get_best_docstring("Rosenbrock").c_str());
    // Hock-Schittkowsky 71
    auto hs71 = pygmo::expose_problem<hock_schittkowsky_71>("hock_schittkowsky_71","__init__()\n\nThe Hock-Schittkowsky 71 problem.\n\n"
        "See :cpp:class:`pagmo::hock_schittkowsky_71`.\n\n");
    hs71.def("best_known",&pygmo::best_known_wrapper<hock_schittkowsky_71>,
        pygmo::get_best_docstring("Hock-Schittkowsky 71").c_str());
}
