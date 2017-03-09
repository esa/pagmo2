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
#include <tuple>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/mbh.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/problems/unconstrain.hpp>

namespace pygmo
{

// pagmo::problem.
extern std::unique_ptr<bp::class_<pagmo::problem>> problem_ptr;

// meta-problems.
extern std::tuple<std::unique_ptr<bp::class_<pagmo::translate>>, std::unique_ptr<bp::class_<pagmo::decompose>>,
                  std::unique_ptr<bp::class_<pagmo::unconstrain>>>
    meta_probs_ptrs;

// pagmo::algorithm.
extern std::unique_ptr<bp::class_<pagmo::algorithm>> algorithm_ptr;

// meta-algorithms.
extern std::tuple<std::unique_ptr<bp::class_<pagmo::mbh>>> meta_algos_ptrs;
}

#endif
