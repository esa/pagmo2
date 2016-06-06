#include <string>

#include "docstrings.hpp"

namespace pygmo
{

std::string population_docstring()
{
    return R"(The population class.

This class represents a population of individuals, i.e., potential candidate solutions to a given problem. In PaGMO an
individual is determined:

* by a unique ID used to track him across generations and migrations,
* by a chromosome (a decision vector),
* by the fitness of the chromosome as evaluated by a :class:`~pygmo.core.problem` and thus including objectives,
  equality constraints and inequality constraints if present.

See also :cpp:class:`pagmo::population`.

)";
}

std::string population_push_back_docstring()
{
    return R"(push_back(x)

Adds one decision vector (chromosome) to the population.

Appends a new chromosome x to the population, evaluating its fitness and creating a new unique identifier for the newly
born individual. In case of exceptions, the population will not be altered.

Args:
    x (array or list of doubles): decision vector to be added to the population

Raises:
    ValueError: if the dimension of *x* is inconsistent with the problem dimension or with the dimension of existing
        decision vectors in the population, the calculated fitness vector has a dimension which is inconsistent with the
        fitness dimension of the problem or with the dimension of existing fitness vectors in the population
    TypeError: if *x* cannot be converted to a vector of doubles

Examples:

>>> from numpy import array
>>> pop = population()
>>> pop.push_back([1])
>>> pop.push_back(array([2]))
>>> pop # doctest: +SKIP
[...]
List of individuals:
#0:
        ID:                     7905479722040097576
        Decision vector:        [1]
        Fitness vector:         [0, 0, 0]
#1:
        ID:                     11652046723302057419
        Decision vector:        [2]
        Fitness vector:         [0, 0, 0]
[...]
>>> pop.push_back(3) # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
   ...
TypeError: cannot convert the type '<class 'int'>' to a vector of doubles

)";
}

std::string population_decision_vector_docstring()
{
    return R"(decision_vector()

Create random decision_vector.

Returns:
    NumPy array of doubles: a random decision vector within the problem’s bounds

)";
}

std::string population_best_idx_docstring()
{
    return R"(best_idx(tol = 0.)

Get best idx. See also :cpp:func:`pagmo::population::best_idx()`.

Args:
    tol (a ``float`` or a array or list of doubles): tolerance

)";
}

std::string problem_docstring()
{
    return R"(The main problem class.

>>> from pygmo import problem, rosenbrock
>>> p = problem(rosenbrock(dim=5))
>>> p.fitness([1,2,3,4,5])
array([ 14814.])

See also :cpp:class:`pagmo::problem`.

)";
}

std::string algorithm_docstring()
{
    return R"(The main algorithm class.

See also :cpp:class:`pagmo::algorithm`.

)";
}

std::string get_best_docstring(const std::string &name)
{
    return R"(best_known()

The best known solution for the )" + name + R"( problem.

Returns:
    NumPy array of doubles: the best known solution for the )" + name + R"( problem

)";
}

std::string rosenbrock_docstring()
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
