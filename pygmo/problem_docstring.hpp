#ifndef PYGMO_PROBLEM_DOCSTRING_HPP
#define PYGMO_PROBLEM_DOCSTRING_HPP

#include <string>

namespace pygmo
{

inline std::string problem_docstring()
{
    return R"(The main problem class.

>>> from pygmo import problem, rosenbrock
>>> p = problem(rosenbrock(dim=5))
>>> p.fitness([1,2,3,4,5])
array([ 14814.])

Additional constructors:

)";
}

}

#endif
