#ifndef PYGMO_PROBLEM_EXPOSITION_SUITE_HPP
#define PYGMO_PROBLEM_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/shared_ptr.hpp>
#include <string>

#include "../include/problem.hpp"
#include "../include/problems/translate.hpp"
#include "common_utils.hpp"

namespace pygmo
{

namespace bp = boost::python;

template <typename Prob>
inline boost::shared_ptr<pagmo::translate> translate_init(const Prob &p, const bp::object &o)
{
    return boost::shared_ptr<pagmo::translate>(::new pagmo::translate(p,to_vd(o)));
}

template <typename Prob>
inline bp::class_<Prob> expose_problem(const char *name, const char *descr, bp::class_<pagmo::problem> &problem_class,
    bp::class_<pagmo::translate,boost::shared_ptr<pagmo::translate>> &tp_class)
{
    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name,descr,bp::init<>("Default constructor."));

    // Expose the problem constructor from Prob.
    problem_class.def(bp::init<const Prob &>(("Constructor from the C++ problem '" + std::string(name) + "'.").c_str()));

    // Expose translate's constructor from Prob and translation vector.
    tp_class.def("__init__",bp::make_constructor(&translate_init<Prob>,boost::python::default_call_policies(),
        (bp::arg("problem"),bp::arg("translation"))),
        ("Constructor from the C++ problem '" + std::string(name) + "' and translation vector.").c_str());

    return c;
}

}

#endif
