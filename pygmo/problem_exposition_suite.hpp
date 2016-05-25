#ifndef PYGMO_PROBLEM_EXPOSITION_SUITE_HPP
#define PYGMO_PROBLEM_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/make_constructor.hpp>
#include <string>

#include "../include/problem.hpp"
#include "../include/problems/translate.hpp"
#include "common_utils.hpp"

namespace pygmo
{

namespace bp = boost::python;

// NOTE: it seems like returning a raw pointer is fine. See the examples here:
// http://www.boost.org/doc/libs/1_61_0/libs/python/test/injected.cpp
template <typename Prob>
inline pagmo::translate *translate_init(const Prob &p, const bp::object &o)
{
    return ::new pagmo::translate(p,to_vd(o));
}

template <typename Prob>
inline bp::class_<Prob> expose_problem(const char *name, const char *descr, bp::class_<pagmo::problem> &problem_class,
    bp::class_<pagmo::translate> &tp_class)
{
    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name,descr,bp::init<>());
    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Expose the problem constructor from Prob.
    problem_class.def(bp::init<const Prob &>(("Constructor from a :class:`pygmo.core." + std::string(name) + "` problem *p*.").c_str(),
        (bp::arg("p"))));
    // Extract Prob.
    problem_class.def("_cpp_extract",&problem_cpp_extract<Prob>);

    // Expose translate's constructor from Prob and translation vector.
    tp_class.def("__init__",bp::make_constructor(&translate_init<Prob>,boost::python::default_call_policies(),
        (bp::arg("p"),bp::arg("t"))),
        ("Constructor from a :class:`pygmo.core." + std::string(name) + "` problem *p* and a translation vector *t*.").c_str());

    return c;
}

}

#endif
