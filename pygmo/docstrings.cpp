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
    x (array or list of floats): decision vector to be added to the population

Raises:
    ValueError: if the dimension of *x* is inconsistent with the problem dimension or the calculated fitness vector has
        a dimension which is inconsistent with the fitness dimension of the problem
    TypeError: if the type of *x* is invalid

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
    NumPy array of floats: a random decision vector within the problem’s bounds

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
    tol (float, or array or list of floats): a scalar tolerance or a vector of tolerances to be applied to
      each constraints

Returns:
    int: the index of the best individual

Raises:
    ValueError: if the population is empty

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
    tol (float, or array or list of floats): a scalar tolerance or a vector of tolerances to be applied to
      each constraints

Returns:
    int: the index of the worst individual

Raises:
    ValueError: if the population is empty

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
    int: the number of individuals

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
    i (int): individual’s index in the population
    x (array or list of floats): a decision vector (chromosome)
    f (array or list of floats): a fitness vector

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
    i (int): individual’s index in the population
    x (array or list of floats): a decision vector (chromosome)

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
    seed (int): the desired seed (must be non-negative)

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

std::string population_get_problem_docstring()
{
    return R"(get_problem()

This method will return a deep copy of the internal problem instance.

Returns:
    :class:`~pygmo.core.problem`: a deep copy of the internal problem instance

Examples:

>>> pop = population()
>>> pop.get_problem() # doctest: +SKIP
Problem name: Null problem
        Global dimension:                       1
        Fitness dimension:                      3
[...]

)";
}

std::string population_get_f_docstring()
{
    return R"(get_f()

This method will return the fitness vectors of the individuals as a 2D NumPy array.

Each row of the returned array represents the fitness vector of the individual at the corresponding position in the
population.

Returns:
    NumPy array of floats: a deep copy of the fitness vectors of the individuals

Examples:

>>> pop = population(size = 1)
>>> pop.get_f() # doctest: +SKIP
array([[ 0.13275027, 0.41543223, 0.28420476]])

)";
}

std::string population_get_x_docstring()
{
    return R"(get_x()

This method will return the chromosomes of the individuals as a 2D NumPy array.

Each row of the returned array represents the chromosome of the individual at the corresponding position in the
population.

Returns:
    NumPy array of floats: a deep copy of the chromosomes of the individuals

Examples:

>>> pop = population(size = 5)
>>> pop.get_x() # doctest: +SKIP
array([[ 0.13275027],
       [ 0.26826544],
       [ 0.30058279],
       [ 0.41543223],
       [ 0.13370117]])

)";
}

std::string population_get_ID_docstring()
{
    return R"(get_ID()

This method will return the IDs of the individuals as a 2D NumPy array.

Each row of the returned array represents the ID of the individual at the corresponding position in the
population.

Returns:
    NumPy array of int: a deep copy of the IDs of the individuals

Examples:

>>> pop = population(size = 5)
>>> pop.get_ID() # doctest: +SKIP
array([12098820240406021962,  2435494813514879429, 16758705632650014019,
       13060277951708126199,  1018350750245690412], dtype=uint64)

)";
}

std::string population_get_seed_docstring()
{
    return R"(get_seed()

This method will return the random seed of the population.

Returns:
    int: the random seed of the population

Examples:

>>> pop = population(seed = 12)
>>> pop.get_seed()
12

)";
}

std::string problem_docstring()
{
    return R"(__init__(prob)

The problem class.

This class represents a generic mathematical programming or evolutionary optimization problem. Optimization problems
can be created in PyGMO by defining a class with an appropriate set of methods (i.e., a user-defined problem, or UDP)
and then by using an instance of the UDP to construct a :class:`~pygmo.core.problem`. The exposed C++ problems can
also be used as UDPs.

Note that the UDP provided on construction will be deep-copied and stored inside the problem.

See also :cpp:class:`pagmo::problem`.

Args:
    prob: a user-defined problem (either C++ or Python)

Raises:
    TypeError,ValueError,RuntimeError: if *prob* is not a user-defined problem
    unspecified: any exception thrown by the constructor of the underlying C++ class

)";
}

std::string problem_fitness_docstring()
{
    return R"(fitness(dv)

Fitness.

This method will calculate the fitness of the input decision vector *dv*. See :cpp:func:`pagmo::problem::fitness()`.

Args:
    dv (array or list of floats): the decision vector (chromosome) to be evaluated

Returns:
    NumPy array of floats: the fitness of *dv*

Raises:
    ValueError: if the length of *dv* is not equal to the dimension of the problem, or if the size of the returned
        fitness is inconsistent with the fitness dimension of the UDP
    TypeError: if the type of *dv* is invalid

Examples:

>>> p = problem(rosenbrock())
>>> p.fitness([1,2])
array([ 100.])
>>> p.fitness('abc') # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  ...
TypeError: cannot convert the type '<class 'str'>' to a vector of doubles

)";
}

std::string problem_gradient_docstring()
{
    return R"(gradient(dv)

Gradient.

This method will calculate the gradient of the input decision vector *dv*. See :cpp:func:`pagmo::problem::gradient()`.

Args:
    dv (array or list of floats): the decision vector (chromosome) to be evaluated

Returns:
    NumPy array of floats: the gradient of *dv*

Raises:
    ValueError: if the length of *dv* is not equal to the dimension of the problem, or if the size of the returned
        value does not match the gradient sparsity pattern dimension
    TypeError: if the type of *dv* is invalid
    RuntimeError: if the user-defined problem does not implement the gradient method

Examples:

>>> p = problem(hock_schittkowsky_71())
>>> p.gradient([1,2,3,4])
array([ 28.,   4.,   5.,   6.,   2.,   4.,   6.,   8., -24., -12.,  -8.,
        -6.])
>>> p = problem(rosenbrock())
>>> p.gradient([1,2]) # doctest: +IGNORE_EXCEPTION_DETAIL
  ...
RuntimeError: gradients have been requested but not implemented.

)";
}

std::string problem_get_best_docstring(const std::string &name)
{
    return R"(best_known()

The best known solution for the )" + name + R"( problem.

Returns:
    NumPy array of floats: the best known solution for the )" + name + R"( problem

)";
}

std::string algorithm_docstring()
{
    return R"(The main algorithm class.

See also :cpp:class:`pagmo::algorithm`.

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

std::string cmaes_docstring()
{
    return R"(__init__(gen = 1, cc = -1, cs = -1, c1 = -1, cmu = -1, sigma0 = -1, ftol = 1e-6, xtol = 1e-6, memory = false, seed = random)

Covariance Matrix Evolutionary Strategy (CMA-ES).

Args:
    gen (int): number of generations
    cc (float): backward time horizon for the evolution path (by default is automatically assigned)
    cs (float): makes partly up for the small variance loss in case the indicator is zero (by default is automatically assigned)
    c1 (float): learning rate for the rank-one update of the covariance matrix (by default is automatically assigned)
    cmu (float): learning rate for the rank-mu  update of the covariance matrix (by default is automatically assigned)
    sigma0 (float): initial step-size
    ftol (float): stopping criteria on the x tolerance
    xtol (float): stopping criteria on the f tolerance
    memory (bool): when true the adapted parameters are not reset between successive calls to the evolve method
    seed (int): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if `gen` is negative or greater than an implementation-defined value
    ValueError: if `cc`, `cs`, `c1`, `cmu` are not in [0,1] or -1

See :cpp:class:`pagmo::cmaes`.

)";
}

} // namespace
