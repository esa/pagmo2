#include "python_includes.hpp"

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/import.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/self.hpp>
#include <sstream>

#include "../include/problem.hpp"
#include "../include/problems/null_problem.hpp"
#include "../include/serialization.hpp"
#include "common_utils.hpp"
#include "numpy.hpp"
#include "object_serialization.hpp"
#include "prob_inner_python.hpp"

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

// TODO move out
struct null_problem_pickle_suite : bp::pickle_suite
{
    static bp::tuple getinitargs(const null_problem &)
    {
        return bp::make_tuple();
    }
};

static inline bp::object fitness_wrapper(const problem &p, const bp::object &dv)
{
    return pygmo::vd_to_a(p.fitness(pygmo::to_vd(dv)));
}

BOOST_PYTHON_MODULE(_core)
{
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
    bp::class_<problem> problem_class("problem",bp::init<const problem &>());
    problem_class.def(bp::init<bp::object>())
        .def(bp::init<const null_problem &>())
        .def(repr(bp::self))
        .def("fitness",&fitness_wrapper)
        .def_pickle(pygmo::problem_pickle_suite());

    bp::class_<null_problem> np_class("null_problem",bp::init<>());
    np_class.def_pickle(null_problem_pickle_suite());
}
