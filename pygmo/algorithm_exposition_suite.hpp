#ifndef PYGMO_ALGORITHM_EXPOSITION_SUITE_HPP
#define PYGMO_ALGORITHM_EXPOSITION_SUITE_HPP

#include "python_includes.hpp"

#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/scope.hpp>
#include <cassert>

#include "common_utils.hpp"
#include "pygmo_classes.hpp"

namespace pygmo
{

namespace bp = boost::python;

template <typename Algo>
inline void algorithm_algo_init()
{
    assert(algorithm_ptr.get() != nullptr);
    auto &algo_class = *algorithm_ptr;
    algo_class.def(bp::init<const Algo &>((bp::arg("a"))));
}

template <typename Algo>
inline bp::class_<Algo> expose_algorithm(const char *name, const char *descr)
{
    assert(algorithm_ptr.get() != nullptr);
    auto &algorithm_class = *algorithm_ptr;
    // We require all algorithms to be def-ctible at the bare minimum.
    bp::class_<Algo> c(name,descr,bp::init<>());
    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Expose the algorithm constructor from Algo.
    algorithm_algo_init<Algo>();
    // Expose extract.
    algorithm_class.def("_cpp_extract",&generic_cpp_extract<pagmo::algorithm,Algo>);

    // Add the algorithm to the algorithms submodule.
    bp::scope().attr("algorithms").attr(name) = c;

    return c;
}

}

#endif
