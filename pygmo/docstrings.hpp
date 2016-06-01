#ifndef PYGMO_DOCSTRINGS_HPP
#define PYGMO_DOCSTRINGS_HPP

#include <string>

namespace pygmo
{

inline std::string population_docstring()
{
    return R"(__init__(prob, size = 0, seed = random)

The population class.

Args:
    prob: a user-defined problem
    size (int): the number of individuals
    seed (int): the random seed (if not specified, randomly-generated)

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

inline std::string algorithm_docstring()
{
    return R"(The main algorithm class.

See also :cpp:class:`pagmo::algorithm`.

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

Args:
    dim (int): number of dimensions

Raises:
    OverflowError: if `dim` is negative or greater than an implementation-defined value
    ValueError: if `dim` is less than 2

See :cpp:class:`pagmo::rosenbrock`.

)";
}

}

#endif
