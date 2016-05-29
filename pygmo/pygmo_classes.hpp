#ifndef PYGMO_PYGMO_CLASSES_HPP
#define PYGMO_PYGMO_CLASSES_HPP

#include <boost/python/class.hpp>
#include <memory>

#include "../include/problem.hpp"
#include "../include/problems/decompose.hpp"
#include "../include/problems/translate.hpp"

namespace pygmo
{

extern std::unique_ptr<bp::class_<pagmo::problem>> problem_ptr;
extern std::unique_ptr<bp::class_<pagmo::translate>> translate_ptr;
extern std::unique_ptr<bp::class_<pagmo::decompose>> decompose_ptr;

}

#endif
