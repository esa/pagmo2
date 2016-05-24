#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/def.hpp>
#include <boost/python/docstring_options.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/import.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/self.hpp>
#include <boost/python/tuple.hpp>
#include <sstream>

#include "../include/problem.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/problems/translate.hpp"
#include "../include/serialization.hpp"
#include "common_utils.hpp"
#include "numpy.hpp"
#include "object_serialization.hpp"
#include "problem.hpp"
#include "problem_docstring.hpp"
#include "problem_exposition_suite.hpp"

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

// Wrapper for the fitness function.
static inline bp::object fitness_wrapper(const problem &p, const bp::object &dv)
{
    return pygmo::vd_to_a(p.fitness(pygmo::to_vd(dv)));
}

BOOST_PYTHON_MODULE(core)
{
    // Setup doc options
    bp::docstring_options doc_options;
    doc_options.enable_py_signatures();
    doc_options.disable_cpp_signatures();

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

    // Problem class.
    bp::class_<problem> problem_class("problem",pygmo::problem_docstring().c_str(),bp::no_init);
    problem_class.def(bp::init<const bp::object &>((bp::arg("p"))))
        .def(bp::init<const problem &>("Deep copy constructor from a :class:`pygmo.core.problem` *p*.",(bp::arg("p"))))
        .def(bp::init<const translate &>("Constructor from a :class:`pygmo.core.translate` problem *p*.",(bp::arg("p"))))
        .def(repr(bp::self))
        .def_pickle(pygmo::problem_pickle_suite())
        .def("fitness",&fitness_wrapper,"Fitness.",(bp::arg("dv")));

    // Translate meta-problem.
    bp::class_<translate> tp("translate","The translate meta-problem.\n\nBlah blah blah blah.\n\nAdditional constructors:",bp::init<>());
    // Constructor from Python concrete problem and translation vector (allows to translate Python problems).
    tp.def("__init__",bp::make_constructor(&pygmo::translate_init<bp::object>,boost::python::default_call_policies(),
        (bp::arg("p"),bp::arg("t"))),"Constructor from a concrete Python problem *p* and a translation vector *t*.");
    // Constructor of translate from translate and translation vector. This allows to apply the
    // translation multiple times.
    tp.def("__init__",bp::make_constructor(&pygmo::translate_init<translate>,boost::python::default_call_policies(),
        (bp::arg("p"),bp::arg("t"))),"Constructor from a :class:`pygmo.core.translate` problem *p* and a translation vector *t*.\n\n"
        "This constructor allows to chain multiple problem translations.");

    auto np = pygmo::expose_problem<null_problem>("null_problem","The null problem.",problem_class,tp);
    // NOTE: this is needed only for the null_problem, as it is used in the implementation of the
    // serialization of the problem. Not necessary for any other problem type.
    np.def_pickle(null_problem_pickle_suite());
    auto rb = pygmo::expose_problem<rosenbrock>("rosenbrock","The Rosenbrock problem.",problem_class,tp);
    rb.def(bp::init<unsigned>("Constructor from dimension *dim*.",(bp::arg("dim"))));
}
