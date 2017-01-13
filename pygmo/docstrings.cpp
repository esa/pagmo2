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

See also the docs of the C++ class :cpp:class:`pagmo::population`.

)";
}

std::string population_push_back_docstring()
{
    return R"(push_back(x)

Adds one decision vector (chromosome) to the population.

Appends a new chromosome x to the population, evaluating its fitness and creating a new unique identifier for the newly
born individual. In case of exceptions, the population will not be altered.

Args:
    x (``array``, or ``list`` of ``floats``): decision vector to be added to the population

Raises:
    ValueError: if the dimension of *x* is inconsistent with the problem dimension or the calculated fitness vector has
        a dimension which is inconsistent with the fitness dimension of the problem
    TypeError: if *x* cannot be converted to a C++ ``vector`` of ``floats``

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
TypeError: cannot convert the type '<class 'int'>' to a vector of floats

)";
}

std::string population_decision_vector_docstring()
{
    return R"(decision_vector()

Create random decision_vector.

Returns:
    ``array`` of ``floats``: a random decision vector within the problem’s bounds

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
    tol (``float``, or ``array``, or ``list`` of ``floats``): a scalar tolerance or a vector of tolerances to be applied to
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
    tol (``float``, or an ``array``, or ``list`` of ``floats``): a scalar tolerance or a vector of tolerances to be applied to
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
    i (``int``): individual’s index in the population
    x (``array`` or ``list`` of ``floats``): a decision vector (chromosome)
    f (``array`` or ``list`` of ``floats``): a fitness vector

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
    i (``int``): individual’s index in the population
    x (``array`` or ``list`` of ``floats``): a decision vector (chromosome)

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
    seed (``int``): the desired seed (must be non-negative)

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
    ``array`` of ``floats``: a deep copy of the fitness vectors of the individuals

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
    ``array`` of ``floats``: a deep copy of the chromosomes of the individuals

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
    ``array`` of ``int``: a deep copy of the IDs of the individuals

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

See also the docs of the C++ class :cpp:class:`pagmo::problem`.

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

The best known solution for the )"
           + name + R"( problem.

Returns:
    ``array`` of ``floats``: the best known solution for the )"
           + name + R"( problem

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
    dim (``int``): number of dimensions

Raises:
    OverflowError: if *dim* is negative or greater than an implementation-defined value
    ValueError: if *dim* is less than 2

See also the docs of the C++ class :cpp:class:`pagmo::rosenbrock`.

)";
}

std::string de_docstring()
{
    return R"(__init__(gen = 1, F = 0.8, CR = 0.9, variant = 2, ftol = 1e-6, xtol = 1e-6, seed = random)

Differential Evolution

Args:
    gen (``int``): number of generations
    F (``float``): weight coefficient (dafault value is 0.8)
    CR (``float``): crossover probability (dafault value is 0.9)
    variant (``int``): mutation variant (dafault variant is 2: /rand/1/exp)
    ftol (``float``): stopping criteria on the x tolerance (default is 1e-6)
    xtol (``float``): stopping criteria on the f tolerance (default is 1e-6)
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen*, *variant* or *seed* is negative or greater than an implementation-defined value
    ValueError: if *F*, *CR* are not in [0,1] or *variant* is not in [1, 10]

The following variants (mutation variants) are available to create a new candidate individual:

+-------------------------+-------------------------+
| 1 - best/1/exp          | 2 - rand/1/exp          |
+-------------------------+-------------------------+
| 3 - rand-to-best/1/exp  | 4 - best/2/exp          |
+-------------------------+-------------------------+
| 5 - rand/2/exp          | 6 - best/1/bin          |
+-------------------------+-------------------------+
| 7 - rand/1/bin          | 8 - rand-to-best/1/bin  |
+-------------------------+-------------------------+
| 9 - best/2/bin          | 10 - rand/2/bin         |
+-------------------------+-------------------------+

See also the docs of the C++ class :cpp:class:`pagmo::de`.

)";
}

std::string de_get_log_docstring()
{
    return R"(de.get_log()

Returns a log containing relevant parameters recorded during the last call to evolve and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method set_verbosity on an object :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.de`. A verbosity of N implies a log line each N generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values Gen, Fevals, Best, dx, df

Where:
    * Gen (``int``), generation number
    * Fevals (``int``), number of functions evaluation made.
    * Best (``float``), the best fitness function currently in the population
    * dx (``float``), the norm of the distance to the population mean of the mutant vectors
    * df (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(de(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop)
    Gen:        Fevals:          Best:            dx:            df:
      1             20         162446        65.2891    1.78686e+06
    101           2020        198.402         8.4454        572.161
    201           4020        21.1155        2.60629        24.5152
    301           6020        6.67069        0.51811        1.99744
    401           8020        3.60022       0.583444       0.554511
    Exit condition -- generations = 500
    >>> al = algo.extract(de)
    >>> al.get_log()
    [(1, 20, 162446.0185265718, 65.28911664703388, 1786857.8926660626), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::de::get_log`.

)";
}

std::string compass_search_docstring()
{
    return R"(__init__(max_fevals = 1, start_range = .1, stop_range = .01, reduction_coeff = .5)

Compass Search

Args:
    max_fevals (``int``): maximum number of function evaluation
    start_range (``float``): start range (dafault value is .1)
    stop_range (``float``): stop range (dafault value is .01)
    reduction_coeff (``float``): range reduction coefficient (dafault value is .5)

Raises:
    OverflowError: if *max_fevals* is negative or greater than an implementation-defined value
    ValueError: if *start_range* is not in (0, 1]
    ValueError: if *stop_range* is not in (*start_range*, 1]
    ValueError: if *reduction_coeff* is not in (0,1)

See also the docs of the C++ class :cpp:class:`pagmo::compass_search`.

)";
}

std::string compass_search_get_log_docstring()
{
    return R"(compass_search.get_log()

Returns a log containing relevant parameters recorded during the last call to evolve and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method set_verbosity on an object :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.compass_search`. A verbosity larger than 0 implies one log line at each improvment of the fitness or
change in the search range.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values Fevals, Best, Range

Where:
    * Fevals (``int``), number of functions evaluation made.
    * Best (``float``), the best fitness function currently in the population
    * Range (``float``), the range used to vary the chromosome (relative to the box bounds width)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(compass_search(max_fevals = 500))
    >>> algo.set_verbosity(1)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 1)
    >>> pop = algo.evolve(pop)
    Fevals:          Best:         Range:
          2         352531            0.1
          4         256100            0.1
        ...        .......            ...
        159        1006.28            0.1
        178        815.293            0.1
        198        815.293           0.05
        204        654.538           0.05
        207        481.065           0.05
        217        372.571           0.05
        229        116.117           0.05
        249        116.117          0.025
        257         90.486          0.025
        ...        .......         ......
        382        14.1021         0.0125
        402        14.1021        0.00625
    Exit condition -- range: 0.00625 <= 0.01

    >>> al = algo.extract(compass_search)
    >>> al.get_log()
    [(2, 352531.0009957846, 0.1), (4, 256100.15408875767, 0.1), ...


See also the docs of the relevant C++ method :cpp:func:`pagmo::compass_search::get_log`.

)";
}

std::string sade_docstring()
{
    return R"(__init__(gen = 1, variant = 2, variant_adptv = 1, ftol = 1e-6, xtol = 1e-6, memory = False, seed = random)

Self-adaptive Differential Evolution

Args:
    gen (``int``): number of generations
    variant (``int``): mutation variant (dafault variant is 2: /rand/1/exp)
    variant_adptv (``int``): F and CR parameter adaptation scheme to be used (one of 1..2)
    ftol (``float``): stopping criteria on the x tolerance (default is 1e-6)
    xtol (``float``): stopping criteria on the f tolerance (default is 1e-6)
    memory (``bool``): when true the adapted parameters CR anf F are not reset between successive calls to the evolve method
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen*, *variant*, *variant_adptv* or *seed* is negative or greater than an implementation-defined value
    ValueError: if *variant* is not in [1,18] or *variant_adptv* is not in [0,1]

The following variants (mutation variants) are available to create a new candidate individual:

+--------------------------------------+--------------------------------------+
| 1 - best/1/exp                       | 2 - rand/1/exp                       |
+--------------------------------------+--------------------------------------+
| 3 - rand-to-best/1/exp               | 4 - best/2/exp                       |
+--------------------------------------+--------------------------------------+
| 5 - rand/2/exp                       | 6 - best/1/bin                       |
+--------------------------------------+--------------------------------------+
| 7 - rand/1/bin                       | 8 - rand-to-best/1/bin               |
+--------------------------------------+--------------------------------------+
| 9 - best/2/bin                       | 10 - rand/2/bin                      |
+--------------------------------------+--------------------------------------+
| 11 - rand/3/exp                      | 12 - rand/3/bin                      |
+--------------------------------------+--------------------------------------+
| 13 - best/3/exp                      | 14 - best/3/bin                      |
+--------------------------------------+--------------------------------------+
| 15 - rand-to-current/2/exp           | 16 - rand-to-current/2/bin           |
+--------------------------------------+--------------------------------------+
| 17 - rand-to-best-and-current/2/exp  | 18 - rand-to-best-and-current/2/bin  |
+--------------------------------------+--------------------------------------+

The following adaptation schemes are available:

+--------------------------------------+--------------------------------------+
| 1 - jDE                              | 2 - iDE                              |
+--------------------------------------+--------------------------------------+

See also the docs of the C++ class :cpp:class:`pagmo::sade`.

)";
}

std::string sade_get_log_docstring()
{
    return R"(sade.get_log()

Returns a log containing relevant parameters recorded during the last call to evolve and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method set_verbosity on an object :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.sade`. A verbosity of N implies a log line each N generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values Gen, Fevals, Best, F, CR, dx, df

Where:
    * Gen (``int``), generation number
    * Fevals (``int``), number of functions evaluation made.
    * Best (``float``), the best fitness function currently in the population
    * F (``float``), the value of the adapted paramter F used to create the best so far
    * CR (``float``), the value of the adapted paramter CR used to create the best so far
    * dx (``float``), the norm of the distance to the population mean of the mutant vectors
    * df (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(sade(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problems.rosenbrock(10)
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop)
    Gen:        Fevals:          Best:             F:            CR:            dx:            df:
      1             20         297060       0.690031       0.294769        44.1494    2.30584e+06
    101           2020        97.4258        0.58354       0.591527        13.3115        441.545
    201           4020        8.79247         0.6678        0.53148        17.8822        121.676
    301           6020        6.84774       0.494549        0.98105        12.2781        40.9626
    401           8020         4.7861       0.428741       0.743813        12.2938        39.7791
    Exit condition -- generations = 500
    >>> al = algo.extract(sade)
    >>> al.get_log()
    [(1, 20, 297059.6296130389, 0.690031071850855, 0.29476914701127666, 44.14940516578547, 2305836.7422693395), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::sade::get_log`.

)";
}

std::string moead_docstring()
{
    return R"(__init__(gen = 1, weight_generation = "grid", neighbours = 20, CR = 1, F = 0.5, eta_m = 20, realb = 0.9, limit = 2, preserve_diversity = true, seed = random)

Multi Objective Evolutionary Algorithms by Decomposition (the DE variant)

Args:
    gen (``int``): number of generations
    weight_generation (``str``): method used to generate the weights, one of "grid", "low discrepancy" or "random"
    neighbours (``int``): size of the weight's neighborhood
    CR (``float``): crossover parameter in the Differential Evolution operator
    F (``float``): parameter for the Differential Evolution operator
    eta_m (``float``): distribution index used by the polynomial mutation
    realb (``float``): chance that the neighbourhood is considered at each generation, rather than the whole population (only if preserve_diversity is true)
    limit (``int``):  maximum number of copies reinserted in the population  (only if m_preserve_diversity is true)
    preserve_diversity (``bool``): when true activates diversity preservation mechanisms
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen*, *neighbours*, *seed* or *limit* are negative or greater than an implementation-defined value
    ValueError: if *weight_generation* is not one of 'random', 'low discrepancy', 'grid'
    ValueError: if *CR* or *F* or *realb* are not in [0.,1.]
    ValueError: if *eta_m* is negative

See also the docs of the C++ class :cpp:class:`pagmo::moead`.

)";
}

std::string moead_get_log_docstring()
{
    return R"(moead.get_log()

Returns a log containing relevant parameters recorded during the last call to evolve and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method set_verbosity on an object :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.moead`. A verbosity of N implies a log line each N generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values Gen, Fevals, ADR, ideal_point.

Where:
    * Gen (``int``), generation number
    * Fevals (``int``), number of functions evaluation made.
    * ADF (``float``), Average Decomposed Fitness, that is the average across all decomposed problem of the single objective decomposed fitness along the corresponding direction.
    * ideal_point (``array``), The ideal point of the current population (cropped to max 5 dimensions only in the screen output)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(moead(gen=500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(zdt())
    >>> pop = population(prob, 40)
    >>> pop = algo.evolve(pop)
    Gen:        Fevals:           ADF:        ideal1:        ideal2:
      1              0        32.5747     0.00190532        2.65685
    101           4000        5.67751    2.56736e-09       0.468789
    201           8000        5.38297    2.56736e-09      0.0855025
    301          12000        5.05509    9.76581e-10      0.0574796
    401          16000        5.13126    9.76581e-10      0.0242256
    >>> al = algo.extract(moead)
    >>> al.get_log()
    [(1, 0, 32.574745630075874, array([  1.90532430e-03,   2.65684834e+00])), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::moead::get_log`.

)";
}

std::string cmaes_docstring()
{
    return R"(__init__(gen = 1, cc = -1, cs = -1, c1 = -1, cmu = -1, sigma0 = 0.5, ftol = 1e-6, xtol = 1e-6, memory = false, seed = random)

Covariance Matrix Evolutionary Strategy (CMA-ES).

Args:
    gen (``int``): number of generations
    cc (``float``): backward time horizon for the evolution path (by default is automatically assigned)
    cs (``float``): makes partly up for the small variance loss in case the indicator is zero (by default is automatically assigned)
    c1 (``float``): learning rate for the rank-one update of the covariance matrix (by default is automatically assigned)
    cmu (``float``): learning rate for the rank-mu  update of the covariance matrix (by default is automatically assigned)
    sigma0 (``float``): initial step-size
    ftol (``float``): stopping criteria on the x tolerance
    xtol (``float``): stopping criteria on the f tolerance
    memory (``bool``): when true the adapted parameters are not reset between successive calls to the evolve method
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen* is negative or greater than an implementation-defined value
    ValueError: if *cc*, *cs*, *c1*, *cmu* are not in [0,1] or -1

See also the docs of the C++ class :cpp:class:`pagmo::cmaes`.

)";
}

std::string cmaes_get_log_docstring()
{
    return R"(cmaes.get_log()

Returns a log containing relevant parameters recorded during the last call to evolve and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method set_verbosity on an object :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.cmaes`. A verbosity of N implies a log line each N generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values Gen, Fevals, Best, dx, df, sigma

Where:
    * Gen (``int``), generation number
    * Fevals (``int``), number of functions evaluation made.
    * Best (``float``), the best fitness function currently in the population
    * dx (``float``), the norm of the distance to the population mean of the mutant vectors
    * df (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual
    * sigma (``float``), the current step-size

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(cmaes(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop)
    Gen:        Fevals:          Best:            dx:            df:         sigma:
      1              0         173924        33.6872    3.06519e+06            0.5
    101           2000        92.9612       0.583942        156.921      0.0382078
    201           4000        8.79819       0.117574          5.101      0.0228353
    301           6000        4.81377      0.0698366        1.34637      0.0297664
    401           8000        1.04445      0.0568541       0.514459      0.0649836
    Exit condition -- generations = 500
    >>> al = algo.extract(cmaes)
    >>> al.get_log()
    [(1, 0, 173924.2840042722, 33.68717961390855, 3065192.3843070837, 0.5), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::cmaes::get_log`.

)";
}

std::string de1220_docstring()
{
    return R"(__init__(gen = 1, allowed_variants = [2,3,7,10,13,14,15,16], variant_adptv = 1, ftol = 1e-6, xtol = 1e-6, memory = False, seed = random)

Self-adaptive Differential Evolution, PaGMO flavour (pDE).
The adaptation of the mutation variant is added to :class:`~pygmo.core.sade`

Args:
    gen (``int``): number of generations
    allowed_variants (``NumPy array or list of floats``): allowed mutation variants, each one being a number in [1, 18]
    variant_adptv (``int``): F and CR parameter adaptation scheme to be used (one of 1..2)
    ftol (``float``): stopping criteria on the x tolerance (default is 1e-6)
    xtol (``float``): stopping criteria on the f tolerance (default is 1e-6)
    memory (``bool``): when true the adapted parameters CR anf F are not reset between successive calls to the evolve method
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen*, *variant*, *variant_adptv* or *seed* is negative or greater than an implementation-defined value
    ValueError: if each id in *variant_adptv* is not in [1,18] or *variant_adptv* is not in [0,1]

The following variants (mutation variants) can be put into *allowed_variants*:

+--------------------------------------+--------------------------------------+
| 1 - best/1/exp                       | 2 - rand/1/exp                       |
+--------------------------------------+--------------------------------------+
| 3 - rand-to-best/1/exp               | 4 - best/2/exp                       |
+--------------------------------------+--------------------------------------+
| 5 - rand/2/exp                       | 6 - best/1/bin                       |
+--------------------------------------+--------------------------------------+
| 7 - rand/1/bin                       | 8 - rand-to-best/1/bin               |
+--------------------------------------+--------------------------------------+
| 9 - best/2/bin                       | 10 - rand/2/bin                      |
+--------------------------------------+--------------------------------------+
| 11 - rand/3/exp                      | 12 - rand/3/bin                      |
+--------------------------------------+--------------------------------------+
| 13 - best/3/exp                      | 14 - best/3/bin                      |
+--------------------------------------+--------------------------------------+
| 15 - rand-to-current/2/exp           | 16 - rand-to-current/2/bin           |
+--------------------------------------+--------------------------------------+
| 17 - rand-to-best-and-current/2/exp  | 18 - rand-to-best-and-current/2/bin  |
+--------------------------------------+--------------------------------------+

The following adaptation schemes for the parameters F and CR are available:

+--------------------------------------+--------------------------------------+
| 1 - jDE                              | 2 - iDE                              |
+--------------------------------------+--------------------------------------+

See also the docs of the C++ class :cpp:class:`pagmo::de1220`.

)";
}

std::string de1220_get_log_docstring()
{
    return R"(de1220.get_log()

Returns a log containing relevant parameters recorded during the last call to evolve and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method set_verbosity on an object :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.de1220`. A verbosity of N implies a log line each N generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values Gen, Fevals, Best, F, CR, Variant, dx, df

Where:
    * Gen (``int``), generation number
    * Fevals (``int``), number of functions evaluation made.
    * Best (``float``), the best fitness function currently in the population
    * F (``float``), the value of the adapted paramter F used to create the best so far
    * CR (``float``), the value of the adapted paramter CR used to create the best so far
    * Variant (``int``), the mutation variant used to create the best so far
    * dx (``float``), the norm of the distance to the population mean of the mutant vectors
    * df (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(de1220(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop)
       Gen:        Fevals:          Best:             F:            CR:       Variant:            dx:            df:
          1             20         285653        0.55135       0.441551             16        43.9719    2.02379e+06
        101           2020        12.2721       0.127285      0.0792493             14        3.22986        106.764
        201           4020        5.72927       0.148337       0.777806             14        2.72177        4.10793
        301           6020        4.85084        0.12193       0.996191              3        2.95555        3.85027
        401           8020        4.20638       0.235997       0.996259              3        3.60338        4.49432
    Exit condition -- generations = 500
    >>> al = algo.extract(de1220)
    >>> al.get_log()
    [(1, 20, 285652.7928977573, 0.551350234239449, 0.4415510963067054, 16, 43.97185788345982, 2023791.5123259544), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::de1220::get_log`.

)";
}

std::string decompose_decompose_fitness_docstring()
{
    return R"(decompose_fitness(f, weights, ref_point)

Returns the original fitness of the multi-objective problem

Args:
    f (``array`` or ``list`` of ``floats``): fitness vector to be decomposed
    weights (``array`` or ``list`` of ``floats``): weights of the decomposition
    ref_point (``array`` or ``list`` of ``floats``): reference point for the decomposition (only for tchebycheff and bi)

Returns:
    ``array`` of ``floats``: one single value representing the decomposed fitness

Raises:
    ValueError: if the dimensions of *f*, *weights* or *ref_point* are inconsistent
    TypeError: if *f*, *weights* or *ref_point* cannot be converted to vectors of floats

Examples:
>>> from pygmo import *
>>> prob = problem(zdt(id=1, param=30))
>>> prob_d = problem(decompose(prob, [0.5,0.5], [0,0], "weighted", False))
>>> fit = prob.fitness([0.5]*30)
>>> fit_d = prob_d.fitness([0.5]*30)
>>> print(fit)
[ 0.5        3.8416876]
>>> print(fit_d)
[ 2.1708438]
>>> prob_d.extract(decompose).decompose_fitness(fit, [0.5,0.5],[0,0])
array([ 2.1708438])
>>> prob_d.extract(decompose).decompose_fitness(fit, [0.4,0.6],[0,0])
array([ 2.50501256])

)";
}

std::string fast_non_dominated_sorting_docstring()
{
    return R"(fast_non_dominated_sorting(points)

Runs the fast non dominated sorting algorithm on the input *points*

Args:
    points (``array`` [or ``list``] of ``arrays`` [or ``lists``] of ``floats``): the input points

Raises:
    ValueError: if *points* is malformed
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    (``tuple``): (*ndf*, *dl*, *dc*, *ndr*)

Where:
    * *ndf* (``list`` of ``arrays``): the non dominated fronts
    * *dl* (``list`` of ``arrays``): the domination list
    * *dc* (``array``): the domination count
    * *ndr* (``array``): the non domination ranks

Examples:
    >>> from pygmo import *
    >>> ndf, dl, dc, ndr = fast_non_dominated_sorting([[2,3],[-1,2],[-3,2],[0,5],[1,1]])
    >>> print(ndf)
    [array([2, 4], dtype=uint64), array([1], dtype=uint64), array([0, 3], dtype=uint64)]
    >>> print(dl)
    [array([], dtype=uint64), array([0, 3], dtype=uint64), array([0, 1, 3], dtype=uint64), array([], dtype=uint64), array([0], dtype=uint64)]
    >>> print(dc)
    [3 1 0 2 0]
    >>> print(ndr)
    [2 1 0 2 0]

)";
}

} // namespace
