#ifndef PYGMO_PYGMO_CLASSES_HPP
#define PYGMO_PYGMO_CLASSES_HPP

#include "python_includes.hpp"

#include <boost/python/class.hpp>
#include <memory>

#include "../include/algorithm.hpp"
#include "../include/population.hpp"
#include "../include/problem.hpp"
#include "../include/problems/decompose.hpp"
#include "../include/problems/translate.hpp"

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
