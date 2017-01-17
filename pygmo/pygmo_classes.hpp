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

#ifndef PYGMO_PYGMO_CLASSES_HPP
#define PYGMO_PYGMO_CLASSES_HPP

#include "python_includes.hpp"

#include <boost/python/class.hpp>
#include <memory>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/translate.hpp>

namespace pygmo
{

// pagmo::problem and meta-problems.
extern std::unique_ptr<bp::class_<pagmo::problem>> problem_ptr;
extern std::unique_ptr<bp::class_<pagmo::translate>> translate_ptr;
extern std::unique_ptr<bp::class_<pagmo::decompose>> decompose_ptr;

// pagmo::algorithm and meta-algorithms.
extern std::unique_ptr<bp::class_<pagmo::algorithm>> algorithm_ptr;
}

#endif
