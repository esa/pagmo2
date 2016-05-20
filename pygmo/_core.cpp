#include "python_includes.hpp"

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/import.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <iostream>

#include "../include/problem.hpp"
#include "common_utils.hpp"
#include "numpy.hpp"
//#include "prob_inner_python.hpp"

namespace bp = boost::python;
using namespace pagmo;

#if PY_MAJOR_VERSION < 3

static inline void wrap_import_array()
{
    import_array();
}

#else

static void *wrap_import_array()
{
    import_array();
    return nullptr;
}

#endif

// Simple vector-to-array conversion.
static inline auto test_vd_to_a()
{
    return pygmo::vd_to_a({1,2,3,4,5,6,7,8,9,10});
}

static inline bool test_to_vd(bp::object o)
{
    return pygmo::to_vd(o) == vector_double{1,2,3,4,5,6,7,8,9,10};
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
        std::cout << "The NumPy module could not be imported. Make sure that NumPy has been correctly installed.";
        throw;
    }
    wrap_import_array();

    // Expose utility functions for testing purposes.
    bp::def("_builtin",&pygmo::builtin);
    bp::def("_type",&pygmo::type);
    bp::def("_str",&pygmo::str);
    bp::def("_callable",&pygmo::callable);
    bp::def("_deepcopy",&pygmo::deepcopy);
    bp::def("_test_vd_to_a",&test_vd_to_a);
    bp::def("_test_to_vd",&test_to_vd);
    bp::def("_to_vd",&pygmo::to_vd);

    // Problem class.
    bp::class_<problem> problem_class("problem",bp::init<const problem &>());

    //problem_class.def(bp::init<bp::object>());
}
