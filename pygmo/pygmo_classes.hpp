#ifndef PYGMO_PYGMO_CLASSES_HPP
#define PYGMO_PYGMO_CLASSES_HPP

#include <boost/python/class.hpp>
#include <memory>

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

// Population.
extern std::unique_ptr<bp::class_<pagmo::population>> population_ptr;

}

#endif
