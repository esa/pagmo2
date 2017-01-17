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
