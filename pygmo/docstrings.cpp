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
    ValueError: if the dimension of *x* is inconsistent with the problem dimension or the calculated fitness vector has
        a dimension which is inconsistent with the fitness dimension of the problem
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

Examples:

>>> pop = population()
>>> pop.decision_vector() # doctest: +SKIP
array([ 0.5393175])

)";
}

std::string population_best_idx_docstring()
{
    return R"(best_idx(tol = 0.)

Index of best individual. See :cpp:func:`pagmo::population::best_idx()`.

Args:
    tol (a double, or an array or list of doubles): a scalar tolerance or a vector of tolerances to be applied to
      each constraints

Returns:
    ``int``: the index of the best individual

Examples:

>>> pop = population(size = 5)
>>> pop.best_idx()
0

)";
}

std::string population_worst_idx_docstring()
{
    return R"(worst_idx(tol = 0.)

Index of worst individual. See :cpp:func:`pagmo::population::worst_idx()`.

Args:
    tol (a double, or an array or list of doubles): a scalar tolerance or a vector of tolerances to be applied to
      each constraints

Returns:
    ``int``: the index of the worst individual

Examples:

>>> pop = population(size = 5)
>>> pop.worst_idx()
0

)";
}

std::string population_size_docstring()
{
    return R"(size()

Size of the population.

The population size can also be queried using the builtin ``len()`` method.

Returns:
    ``int``: the number of individuals

Examples:

>>> pop = population(size = 5)
>>> pop.size()
5
>>> len(pop)
5

)";
}

std::string population_set_xf_docstring()
{
    return R"(set_xf(i,x,f)

Sets the i-th individual's decision vector and fitness.

Sets simultaneously the i-th individual decision vector and fitness, thus avoiding to trigger a fitness
function evaluation.

Args:
    i (an ``int``): individual’s index in the population
    x (an array or list of doubles): a decision vector (chromosome)
    f (an array or list of doubles): a fitness vector

Raises:
    ValueError: if *i* is invalid, or if *x* or *f* have the wrong dimensions (i.e., their dimensions are
        inconsistent with the problem properties)
    TypeError: if the argument types are invalid

Examples:

>>> pop = population(size = 1)
>>> pop.set_xf(0,[1],[1,2,3])
>>> pop # doctest: +SKIP
[...]
List of individuals:
#0:
        ID:                     12917122990260990364
        Decision vector:        [1]
        Fitness vector:         [1, 2, 3]
>>> pop.set_xf(1,[1],[1,2,3]) # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  ...
ValueError: Trying to access individual at position: 1, while population has size: 1

)";
}

std::string population_set_x_docstring()
{
    return R"(set_x(i,x)

Sets the i-th individual's decision vector.

The fitness of the individual will be computed from *x*.

Args:
    i (an ``int``): individual’s index in the population
    x (an array or list of doubles): a decision vector (chromosome)

Raises:
    ValueError: if *i* is invalid, or if *x* has the wrong dimensions (i.e., the dimension is
        inconsistent with the problem properties)
    TypeError: if the argument types are invalid

Examples:

>>> pop = population(size = 1)
>>> pop.set_x(0,[1])
>>> pop # doctest: +SKIP
[...]
List of individuals:
#0:
        ID:                     5051278815751827100
        Decision vector:        [1]
        Fitness vector:         [0, 0, 0]
>>> pop.set_x(1,[1,2]) # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  ...
ValueError: Length of decision vector is 2, should be 1

)";
}

std::string population_set_problem_seed_docstring()
{
    return R"(set_problem_seed(seed)

Sets the problem seed.

Args:
    seed (an ``int``): the desired seed (must be non-negative)

Raises:
    RuntimeError: if the problem is not stochastic
    OverflowError: if *seed* is too large or negative
    TypeError: if the argument types are invalid

Examples:

>>> pop = population(inventory())
>>> pop.set_problem_seed(42)
>>> pop # doctest: +SKIP
[...]
Extra info:
        Weeks: 4
        Sample size: 10
        Seed: 42
[...]
>>> pop = population()
>>> pop.set_problem_seed(seed = 42) # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  ...
RuntimeError: the user-defined problem does not support seed setting

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
