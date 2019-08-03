/* Copyright 2017-2018 PaGMO development team

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

#ifndef PYGMO_PYGMO_CLASSES_HPP
#define PYGMO_PYGMO_CLASSES_HPP

#include <pygmo/python_includes.hpp>

#include <boost/python/class.hpp>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/island.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/topology.hpp>

namespace pygmo
{

namespace bp = boost::python;

// pagmo::problem.
extern std::unique_ptr<bp::class_<pagmo::problem>> problem_ptr;

// pagmo::algorithm.
extern std::unique_ptr<bp::class_<pagmo::algorithm>> algorithm_ptr;

// pagmo::island.
extern std::unique_ptr<bp::class_<pagmo::island>> island_ptr;

// pagmo::bfe.
extern std::unique_ptr<bp::class_<pagmo::bfe>> bfe_ptr;

// pagmo::topology.
extern std::unique_ptr<bp::class_<pagmo::topology>> topology_ptr;

// pagmo::r_policy.
extern std::unique_ptr<bp::class_<pagmo::r_policy>> r_policy_ptr;

// pagmo::s_policy.
extern std::unique_ptr<bp::class_<pagmo::s_policy>> s_policy_ptr;

// Getters for the objects above.
inline bp::class_<pagmo::problem> &get_problem_class()
{
    if (!problem_ptr) {
        std::cerr << "Null problem class pointer." << std::endl;
        std::abort();
    }
    return *problem_ptr;
}

inline bp::class_<pagmo::algorithm> &get_algorithm_class()
{
    if (!algorithm_ptr) {
        std::cerr << "Null algorithm class pointer." << std::endl;
        std::abort();
    }
    return *algorithm_ptr;
}

inline bp::class_<pagmo::island> &get_island_class()
{
    if (!island_ptr) {
        std::cerr << "Null island class pointer." << std::endl;
        std::abort();
    }
    return *island_ptr;
}

inline bp::class_<pagmo::bfe> &get_bfe_class()
{
    if (!bfe_ptr) {
        std::cerr << "Null bfe class pointer." << std::endl;
        std::abort();
    }
    return *bfe_ptr;
}

inline bp::class_<pagmo::topology> &get_topology_class()
{
    if (!topology_ptr) {
        std::cerr << "Null topology class pointer." << std::endl;
        std::abort();
    }
    return *topology_ptr;
}

inline bp::class_<pagmo::r_policy> &get_r_policy_class()
{
    if (!r_policy_ptr) {
        std::cerr << "Null r_policy class pointer." << std::endl;
        std::abort();
    }
    return *r_policy_ptr;
}

inline bp::class_<pagmo::s_policy> &get_s_policy_class()
{
    if (!s_policy_ptr) {
        std::cerr << "Null s_policy class pointer." << std::endl;
        std::abort();
    }
    return *s_policy_ptr;
}

} // namespace pygmo

#endif
