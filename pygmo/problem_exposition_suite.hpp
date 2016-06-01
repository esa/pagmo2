#ifndef PYGMO_PROBLEM_EXPOSITION_SUITE_HPP
#define PYGMO_PROBLEM_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/object.hpp>
#include <boost/python/scope.hpp>
#include <cassert>
#include <string>

#include "../include/population.hpp"
#include "../include/problem.hpp"
#include "../include/problems/translate.hpp"
#include "common_utils.hpp"
#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

// Wrapper for the best known method.
template <typename Prob>
inline bp::object best_known_wrapper(const Prob &p)
{
    return vd_to_a(p.best_known());
}

// NOTE: it seems like returning a raw pointer is fine. See the examples here:
// http://www.boost.org/doc/libs/1_61_0/libs/python/test/injected.cpp
template <typename Prob>
inline pagmo::translate *translate_init(const Prob &p, const bp::object &o)
{
    return ::new pagmo::translate(p,to_vd(o));
}

// Expose a population constructor from problem.
template <typename Prob>
inline void population_prob_init()
{
    assert(population_ptr.get() != nullptr);
    auto &pop_class = *population_ptr;
    pop_class.def(bp::init<const Prob &,pagmo::population::size_type>((bp::arg("p"),bp::arg("size") = 0u)))
        .def(bp::init<const Prob &,pagmo::population::size_type,unsigned>((bp::arg("p"),bp::arg("size") = 0u,bp::arg("seed"))));
}

// Expose a problem ctor from a user-defined problem.
template <typename Prob>
inline void problem_prob_init()
{
    assert(problem_ptr.get() != nullptr);
    auto &prob_class = *problem_ptr;
    prob_class.def(bp::init<const Prob &>((bp::arg("p"))));
}

// Main problem exposition function.
template <typename Prob>
inline bp::class_<Prob> expose_problem(const char *name, const char *descr)
{
    assert(problem_ptr.get() != nullptr);
    assert(translate_ptr.get() != nullptr);
    auto &problem_class = *problem_ptr;
    auto &tp_class = *translate_ptr;
    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name,descr,bp::init<>());
    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Expose the ctor of population from problem.
    population_prob_init<Prob>();

    // Expose the problem constructor from Prob.
    problem_prob_init<Prob>();
    // Expose extract.
    problem_class.def("_cpp_extract",&generic_cpp_extract<pagmo::problem,Prob>);

    // Expose translate's constructor from Prob and translation vector.
    tp_class.def("__init__",bp::make_constructor(&translate_init<Prob>,boost::python::default_call_policies(),
        (bp::arg("p"),bp::arg("t"))))
        // Extract.
        .def("_cpp_extract",&generic_cpp_extract<pagmo::translate,Prob>);

    // Add the problem to the problems submodule.
    bp::scope().attr("problems").attr(name) = c;

    return c;
}

}

#endif
