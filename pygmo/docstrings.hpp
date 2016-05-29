#ifndef PYGMO_DOCSTRINGS_HPP
#define PYGMO_DOCSTRINGS_HPP

#include <string>

namespace pygmo
{

inline std::string population_docstring()
{
    return R"(__init__(p, size = 0, seed = random)

The population class.

See also :cpp:class:`pagmo::population`.

)";
}

inline std::string problem_docstring()
{
    return R"(The main problem class.

>>> from pygmo import problem, rosenbrock
>>> p = problem(rosenbrock(dim=5))
>>> p.fitness([1,2,3,4,5])
array([ 14814.])

See also :cpp:class:`pagmo::problem`.

)";
}

inline std::string get_best_docstring(const std::string &name)
{
    return "best_known()\n\nThe best known solution for the " + name + " problem.\n\n"
        ":returns: the best known solution for the " + name + " problem\n"
        ":rtype: an array of doubles\n\n";
}

inline std::string rosenbrock_docstring()
{
    return R"(__init__(dim = 2)

The Rosenbrock problem.

:param dim: number of dimensions
:type dim: ``int``
:raises: :exc:`OverflowError` if *dim* is negative or greater than an implementation-defined value
:raises: :exc:`ValueError` if *dim* is less than 2

See :cpp:class:`pagmo::rosenbrock`.

)";
}

}

#endif
