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

#ifndef PYGMO_PROBLEM_EXPOSITION_SUITE_HPP
#define PYGMO_PROBLEM_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/object.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/scope.hpp>
#include <cassert>
#include <memory>
#include <string>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

// Wrapper for the best known method.
// NOTE: abstracted here because it is used in multiple places.
template <typename Prob>
inline bp::object best_known_wrapper(const Prob &p)
{
    return v_to_a(p.best_known());
}

// This is a class whose call operator is invoked to expose the constructors of
// the meta-problem Meta from the UDP T. It needs to be specialised for the various
// meta-problems, as the default implementation does not define a call operator.
template <typename Meta, typename T>
struct make_meta_problem_init {
};

// Constructor of translate from problem and translation vector.
// NOTE: it seems like returning a raw pointer is fine. See the examples here:
// http://www.boost.org/doc/libs/1_61_0/libs/python/test/injected.cpp
template <typename Prob>
inline pagmo::translate *translate_init(const Prob &p, const bp::object &o)
{
    auto vd = to_vd(o);
    return ::new pagmo::translate(p, vd);
}

// NOTE: we specialise this as we need to avoid that we end up using a pagmo::problem
// wrapped in a bp::object as a UDP. This is needed in order to make consistent the behaviour
// between C++ (where translate cannot be cted from pagmo::problem) and Python.
template <>
inline pagmo::translate *translate_init<bp::object>(const bp::object &p, const bp::object &o)
{
    if (type(p) == *problem_ptr) {
        pygmo_throw(PyExc_TypeError, "a pygmo problem is not a user-defined problem, and it cannot be used "
                                     "as a construction argument for the translate meta-problem");
    }
    auto vd = to_vd(o);
    return ::new pagmo::translate(p, vd);
}

// Implement the structure to define constructors for the translate meta-problem.
template <typename T>
struct make_meta_problem_init<pagmo::translate, T> {
    void operator()(bp::class_<pagmo::translate> &tp) const
    {
        tp.def("__init__",
               bp::make_constructor(
                   +[](const T &p, const bp::object &translation) { return translate_init(p, translation); },
                   bp::default_call_policies(), (bp::arg("udp"), bp::arg("translation"))));
    }
};

// Constructor of translate from problem and translation vector.
// NOTE: it seems like returning a raw pointer is fine. See the examples here:
// http://www.boost.org/doc/libs/1_61_0/libs/python/test/injected.cpp
template <typename Prob>
inline pagmo::unconstrain *unconstrain_init(const Prob &p, const std::string &method, const bp::object &o)
{
    auto vd = to_vd(o);
    return ::new pagmo::unconstrain(p, method, vd);
}

// NOTE: we specialise this as we need to avoid that we end up using a pagmo::problem
// wrapped in a bp::object as a UDP. This is needed in order to make consistent the behaviour
// between C++ (where translate cannot be cted from pagmo::problem) and Python.
template <>
inline pagmo::unconstrain *unconstrain_init<bp::object>(const bp::object &p, const std::string &method,
                                                        const bp::object &o)
{
    if (type(p) == *problem_ptr) {
        pygmo_throw(PyExc_TypeError, "a pygmo problem is not a user-defined problem, and it cannot be used "
                                     "as a construction argument for the translate meta-problem");
    }
    auto vd = to_vd(o);
    return ::new pagmo::unconstrain(p, method, vd);
}

// Implement the structure to define constructors for the unconstrain meta-problem.
template <typename T>
struct make_meta_problem_init<pagmo::unconstrain, T> {
    void operator()(bp::class_<pagmo::unconstrain> &tp) const
    {
        tp.def("__init__",
               bp::make_constructor(+[](const T &p, const std::string &method,
                                        const bp::object &weights) { return unconstrain_init(p, method, weights); },
                                    bp::default_call_policies(),
                                    (bp::arg("udp"), bp::arg("method"), bp::arg("weights") = pagmo::vector_double{})));
    }
};

// Constructor of decompose from problem and weight, z, method and bool flag.
template <typename Prob>
inline pagmo::decompose *decompose_init(const Prob &p, const bp::object &weight, const bp::object &z,
                                        const std::string &method, bool adapt_ideal)
{
    auto vd_w = to_vd(weight);
    auto vd_z = to_vd(z);
    return ::new pagmo::decompose(p, vd_w, vd_z, method, adapt_ideal);
}

template <>
inline pagmo::decompose *decompose_init<bp::object>(const bp::object &p, const bp::object &weight, const bp::object &z,
                                                    const std::string &method, bool adapt_ideal)
{
    if (type(p) == *problem_ptr) {
        pygmo_throw(PyExc_TypeError, "a pygmo problem is not a user-defined problem, and it cannot be used "
                                     "as a construction argument for the decompose meta-problem");
    }
    auto vd_w = to_vd(weight);
    auto vd_z = to_vd(z);
    return ::new pagmo::decompose(p, vd_w, vd_z, method, adapt_ideal);
}

// Implement the structure to define constructors for the decompose meta-problem.
template <typename T>
struct make_meta_problem_init<pagmo::decompose, T> {
    void operator()(bp::class_<pagmo::decompose> &dp) const
    {
        dp.def("__init__", bp::make_constructor(
                               +[](const T &p, const bp::object &weight, const bp::object &z, const std::string &method,
                                   bool adapt_ideal) { return decompose_init(p, weight, z, method, adapt_ideal); },
                               bp::default_call_policies(),
                               (bp::arg("udp"), bp::arg("weight"), bp::arg("z"),
                                bp::arg("method") = std::string("weighted"), bp::arg("adapt_ideal") = false)));
    }
};

// Expose a problem ctor from a C++ UDP.
// NOTE: abstracted in a separate wrapper because it is re-used in core.cpp.
template <typename Prob>
inline void problem_expose_init_cpp_udp()
{
    assert(problem_ptr.get() != nullptr);
    auto &prob_class = *problem_ptr;
    prob_class.def(bp::init<const Prob &>((bp::arg("udp"))));
}

// This is a helper struct used to connect C++ meta-problems and C++ UDPs. It will expose
// the constructor of the meta prob T from the C++ UDP Prob, and the extraction from T of Prob.
template <typename Prob>
struct problem_connect_metas_cpp_udp {
    template <typename T>
    void operator()(std::unique_ptr<bp::class_<T>> &ptr) const
    {
        // Expose the meta's constructor from Prob.
        make_meta_problem_init<T, Prob>{}(*ptr);
        // Extract Prob from the meta.
        ptr->def("_cpp_extract", &generic_cpp_extract<T, Prob>, bp::return_internal_reference<>());
    }
};

// Main C++ UDP exposition function.
template <typename Prob>
inline bp::class_<Prob> expose_problem(const char *name, const char *descr)
{
    assert(problem_ptr.get() != nullptr);
    auto &problem_class = *problem_ptr;
    // We require all problems to be def-ctible at the bare minimum.
    bp::class_<Prob> c(name, descr, bp::init<>());
    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Expose the problem constructor from Prob.
    problem_expose_init_cpp_udp<Prob>();
    // Expose extract.
    problem_class.def("_cpp_extract", &generic_cpp_extract<pagmo::problem, Prob>, bp::return_internal_reference<>());

    // Add the problem to the problems submodule.
    bp::scope().attr("problems").attr(name) = c;

    // Expose ctor/extract functionality of the metas wrt Prob.
    pagmo::detail::tuple_for_each(meta_probs_ptrs, problem_connect_metas_cpp_udp<Prob>{});

    return c;
}

// Main C++ meta-problem exposition function.
template <typename Meta>
inline void expose_meta_problem(std::unique_ptr<bp::class_<Meta>> &ptr, const char *name, const char *descr)
{
    assert(ptr.get() == nullptr);
    assert(problem_ptr.get() != nullptr);
    auto &problem_class = *problem_ptr;
    // Create the class and expose def ctor.
    ptr = make_unique<bp::class_<Meta>>(name, descr, bp::init<>());
    // Make meta constructor from Python user-defined problem (allows to init a meta from Python UDPs).
    // This needs to be the first exposed ctor as BP tries the constructors in reverse order, so this needs
    // to be the last constructor tried during overload resolution.
    make_meta_problem_init<Meta, bp::object>{}(*ptr);
    // Python udp extraction.
    ptr->def("_py_extract", &generic_py_extract<Meta>);
    // Mark it as a cpp problem.
    ptr->attr("_pygmo_cpp_problem") = true;
    // Ctor of problem from Meta.
    problem_expose_init_cpp_udp<Meta>();
    // Extract a Meta problem from pagmo::problem.
    problem_class.def("_cpp_extract", &generic_cpp_extract<pagmo::problem, Meta>, bp::return_internal_reference<>());
    // Add it to the problems submodule.
    bp::scope().attr("problems").attr(name) = *ptr;
}
}

#endif
