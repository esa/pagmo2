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
#include <pagmo/island.hpp>
#include <pagmo/problem.hpp>

namespace pygmo
{

namespace bp = boost::python;

// pagmo::problem.
extern std::unique_ptr<bp::class_<pagmo::problem>> problem_ptr;

// pagmo::algorithm.
extern std::unique_ptr<bp::class_<pagmo::algorithm>> algorithm_ptr;

// pagmo::island.
extern std::unique_ptr<bp::class_<pagmo::island>> island_ptr;

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
} // namespace pygmo

#endif
