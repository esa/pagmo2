/* Copyright 2017-2018 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#include <string>

#include <pygmo/docstrings.hpp>

namespace pygmo
{

std::string population_docstring()
{
    return R"(The population class.

This class represents a population of individuals, i.e., potential candidate solutions to a given problem. In pygmo an
individual is determined:

* by a unique ID used to track him across generations and migrations,
* by a chromosome (a decision vector),
* by the fitness of the chromosome as evaluated by a :class:`~pygmo.problem` and thus including objectives,
  equality constraints and inequality constraints if present.

A special mechanism is implemented to track the best individual that has ever been part of the population. Such an individual
is called *champion* and its decision vector and fitness vector are automatically kept updated. The *champion* is not necessarily
an individual currently in the population. The *champion* is only defined and accessible via the population interface if the
:class:`~pygmo.problem` currently contained in the :class:`~pygmo.population` is single objective.

See also the docs of the C++ class :cpp:class:`pagmo::population`.

)";
}

std::string population_push_back_docstring()
{
    return R"(push_back(x, f = None)

Adds one decision vector (chromosome) to the population.

This method will append a new chromosome *x* to the population, creating a new unique identifier for the newly born individual
and, if *f* is not provided, evaluating its fitness. If *f* is provided, the fitness of the new individual will be set to *f*.
It is the user's responsibility to ensure that *f* actually corresponds to the fitness of *x*.

In case of exceptions, the population will not be altered.

Args:
    x (array-like object): decision vector to be added to the population

Raises:
    ValueError: if the dimensions of *x* or *f* (if provided) are incompatible with the population's problem
    unspecified: any exception thrown by :func:`pygmo.problem.fitness()` or by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_random_decision_vector_docstring()
{
    return R"(random_decision_vector()

This method will create a random decision vector within the problem's bounds.

Returns:
    1D NumPy float array: a random decision vector within the problem’s bounds

Raises:
    unspecified: any exception thrown by :func:`pygmo.problem.fitness()` or by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_best_idx_docstring()
{
    return R"(best_idx(tol = self.problem.c_tol)

Index of the best individual.

If the problem is single-objective and unconstrained, the best is simply the individual with the smallest fitness. If the problem
is, instead, single objective, but with constraints, the best will be defined using the criteria specified in :func:`pygmo.sort_population_con()`.
If the problem is multi-objective one single best is not well defined. In this case the user can still obtain a strict ordering of the population
individuals by calling the :func:`pygmo.sort_population_ mo()` function.

Args:
    tol (``float`` or array-like object): scalar tolerance or vector of tolerances to be applied to each constraints. By default, the c_tol attribute 
    from the population problem is used.

Returns:
    ``int``: the index of the best individual

Raises:
     ValueError: if the problem is multiobjective and thus a best individual is not well defined, or if the population is empty
     unspecified: any exception thrown by :cpp:func:`pagmo::sort_population_con()`

)";
}

std::string population_worst_idx_docstring()
{
    return R"(worst_idx(tol = 0.)

Index of the worst individual.

If the problem is single-objective and unconstrained, the worst is simply the individual with the largest fitness. If the problem
is, instead, single objective, but with constraints, the worst will be defined using the criteria specified in :func:`pygmo.sort_population_con()`.
If the problem is multi-objective one single worst is not well defined. In this case the user can still obtain a strict ordering of the population
individuals by calling the :func:`pygmo.sort_population_ mo()` function.

Args:
    tol (``float`` or array-like object): scalar tolerance or vector of tolerances to be applied to each constraints

Returns:
    ``int``: the index of the worst individual

Raises:
     ValueError: if the problem is multiobjective and thus a worst individual is not well defined, or if the population is empty
     unspecified: any exception thrown by :func:`pygmo.sort_population_con()`

)";
}

std::string population_champion_x_docstring()
{
    return R"(Champion's decision vector.

This read-only property contains an array of ``float`` representing the decision vector of the population's champion.

.. note::

   If the problem is stochastic the champion is the individual that had the lowest fitness for
   some lucky seed, not on average across seeds. Re-evaluating its decision vector may then result in a different
   fitness.

Returns:
    1D NumPy float array: the champion's decision vector

Raises:
    ValueError: if the current problem is not single objective
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_champion_f_docstring()
{
    return R"(Champion's fitness vector.

This read-only property contains an array of ``float`` representing the fitness vector of the population's champion.

.. note::

   If the problem is stochastic, the champion is the individual that had the lowest fitness for
   some lucky seed, not on average across seeds. Re-evaluating its decision vector may then result in a different
   fitness.

Returns:
    1D NumPy float array: the champion's fitness vector

Raises:
    ValueError: if the current problem is not single objective
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_set_xf_docstring()
{
    return R"(set_xf(i,x,f)

Sets the :math:`i`-th individual decision vector, and fitness.

Sets simultaneously the :math:`i`-th individual decision vector and fitness thus avoiding to trigger a fitness function evaluation.

.. note::

   The user must make sure that the input fitness *f* makes sense as pygmo will only check its dimension.

Args:
    i (``int``): individual’s index in the population
    x (array-like object): a decision vector (chromosome)
    f (array-like object): a fitness vector

Raises:
    ValueError: if *i* is invalid, or if *x* or *f* have the wrong dimensions (i.e., their dimensions are
        inconsistent with the problem's properties)
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_set_x_docstring()
{
    return R"(set_x(i,x)

Sets the :math:`i`-th individual decision vector.

Sets the chromosome of the :math:`i`-th individual to the value *x* and changes its fitness accordingly. The
individual's ID remains the same.

.. note::

   A call to this method triggers one fitness function evaluation.

Args:
    i (``int``): individual’s index in the population
    x (array-like object): a decision vector (chromosome)

Raises:
    ValueError: if *i* is invalid, or if *x* has the wrong dimensions (i.e., the dimension is
        inconsistent with the problem's properties)
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_problem_docstring()
{
    return R"(Population's problem.

This read-only property gives direct access to the :class:`~pygmo.problem` stored within the population.

Returns:
    :class:`~pygmo.problem`: a reference to the internal problem

)";
}

std::string population_get_f_docstring()
{
    return R"(get_f()

This method will return the fitness vectors of the individuals as a 2D NumPy array.

Each row of the returned array represents the fitness vector of the individual at the corresponding position in the
population.

Returns:
    2D NumPy float array: a deep copy of the fitness vectors of the individuals

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_get_x_docstring()
{
    return R"(get_x()

This method will return the chromosomes of the individuals as a 2D NumPy array.

Each row of the returned array represents the chromosome of the individual at the corresponding position in the
population.

Returns:
    2D NumPy float array: a deep copy of the chromosomes of the individuals

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_get_ID_docstring()
{
    return R"(get_ID()

This method will return the IDs of the individuals as a 1D NumPy array.

Each element of the returned array represents the ID of the individual at the corresponding position in the
population.

Returns:
    1D NumPy int array: a deep copy of the IDs of the individuals

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_get_seed_docstring()
{
    return R"(get_seed()

This method will return the random seed of the population.

Returns:
    ``int``: the random seed of the population

)";
}

std::string problem_docstring()
{
    return R"(__init__(udp = null_problem())

Problem class.

This class represents a generic *mathematical programming* or *evolutionary optimization* problem in the form:

.. math::
   \begin{array}{rl}
   \mbox{find:}      & \mathbf {lb} \le \mathbf x \le \mathbf{ub}\\
   \mbox{to minimize: } & \mathbf f(\mathbf x, s) \in \mathbb R^{n_{obj}}\\
   \mbox{subject to:} & \mathbf {c}_e(\mathbf x, s) = 0 \\
                     & \mathbf {c}_i(\mathbf x, s) \le 0
   \end{array}

where :math:`\mathbf x \in \mathbb R^{n_{cx}} \times  \mathbb Z^{n_{ix}}` is called *decision vector* or
*chromosome*, and is made of :math:`n_{cx}` real numbers and :math:`n_{ix}` integers (all represented as doubles). The
total problem dimension is then indicated with :math:`n_x = n_{cx} + n_{ix}`. :math:`\mathbf{lb}, \mathbf{ub} \in
\mathbb R^{n_{cx}} \times  \mathbb Z^{n_{ix}}` are the *box-bounds*, :math:`\mathbf f: \mathbb R^{n_{cx}} \times
\mathbb Z^{n_{ix}} \rightarrow \mathbb R^{n_{obj}}` define the *objectives*, :math:`\mathbf c_e:  \mathbb R^{n_{cx}}
\times  \mathbb Z^{n_{ix}} \rightarrow \mathbb R^{n_{ec}}` are non linear *equality constraints*, and :math:`\mathbf
c_i:  \mathbb R^{n_{cx}} \times  \mathbb Z^{n_{ix}} \rightarrow \mathbb R^{n_{ic}}` are non linear *inequality
constraints*. Note that the objectives and constraints may also depend from an added value :math:`s` seeding the
values of any number of stochastic variables. This allows also for stochastic programming tasks to be represented by
this class. A tolerance is also considered for all constraints and set, by default, to zero. It can be modified
via the :attr:`~pygmo.problem.c_tol` attribute.

In order to define an optimizaztion problem in pygmo, the user must first define a class
whose methods describe the properties of the problem and allow to compute
the objective function, the gradient, the constraints, etc. In pygmo, we refer to such
a class as a **user-defined problem**, or UDP for short. Once defined and instantiated,
a UDP can then be used to construct an instance of this class, :class:`~pygmo.problem`, which
provides a generic interface to optimization problems.

Every UDP must implement at least the following two methods:

.. code-block:: python

   def fitness(self, dv):
     ...
   def get_bounds(self):
     ...

The ``fitness()`` method is expected to return the fitness of the input decision vector (
* concatenating the objectives, the equality and the inequality constraints), while
``get_bounds()`` is expected to return the box bounds of the problem,
:math:`(\mathbf{lb}, \mathbf{ub})`, which also implicitly define the dimension of the problem.
The ``fitness()`` and ``get_bounds()`` methods of the UDP are accessible from the corresponding
:func:`pygmo.problem.fitness()` and :func:`pygmo.problem.get_bounds()`
methods (see their documentation for information on how the two methods should be implemented
in the UDP and other details).

The two mandatory methods above allow to define a single objective, deterministic, derivative-free, unconstrained
optimization problem. In order to consider more complex cases, the UDP may implement one or more of the following
methods:

.. code-block:: python

   def get_nobj(self):
     ...
   def get_nec(self):
     ...
   def get_nic(self):
     ...
   def get_nix(self):
     ...
   def has_gradient(self):
     ...
   def gradient(self, dv):
     ...
   def has_gradient_sparsity(self):
     ...
   def gradient_sparsity(self):
     ...
   def has_hessians(self):
     ...
   def hessians(self, dv):
     ...
   def has_hessians_sparsity(self):
     ...
   def hessians_sparsity(self):
     ...
   def has_set_seed(self):
     ...
   def set_seed(self, s):
     ...
   def get_name(self):
     ...
   def get_extra_info(self):
     ...

See the documentation of the corresponding methods in this class for details on how the optional
methods in the UDP should be implemented and on how they are used by :class:`~pygmo.problem`.
Note that the exposed C++ problems can also be used as UDPs, even if they do not expose any of the
mandatory or optional methods listed above (see :ref:`here <py_problems>` for the
full list of UDPs already coded in pygmo).

This class is the Python counterpart of the C++ class :cpp:class:`pagmo::problem`.

Args:
    udp: a user-defined problem, either C++ or Python

Raises:
    NotImplementedError: if *udp* does not implement the mandatory methods detailed above
    ValueError: in the following cases:

      * the number of objectives of the UDP is zero,
      * the number of objectives, equality or inequality constraints is larger than an implementation-defined value,
      * the problem bounds are invalid (e.g., they contain NaNs, the dimensionality of the lower bounds is
        different from the dimensionality of the upper bounds, etc. - note that infinite bounds are allowed),
      * the ``gradient_sparsity()`` and ``hessians_sparsity()`` methods of the UDP fail basic sanity checks
        (e.g., they return vectors with repeated indices, they contain indices exceeding the problem's dimensions, etc.)
    unspecified: any exception thrown by:

      * methods of the UDP invoked during construction,
      * the deep copy of the UDP,
      * the constructor of the underlying C++ class,
      * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
        signatures, etc.)

)";
}

std::string problem_fitness_docstring()
{
    return R"(fitness(dv)

Fitness.

This method will invoke the ``fitness()`` method of the UDP to compute the fitness of the
input decision vector *dv*. The return value of the ``fitness()`` method of the UDP is expected to have a
dimension of :math:`n_{f} = n_{obj} + n_{ec} + n_{ic}` and to contain the concatenated values of
:math:`\mathbf f, \mathbf c_e` and :math:`\mathbf c_i` (in this order).
Equality constraints are all assumed in the form :math:`c_{e_i}(\mathbf x) = 0` while inequalities are assumed in
the form :math:`c_{i_i}(\mathbf x) <= 0` so that negative values are associated to satisfied inequalities.

In addition to invoking the ``fitness()`` method of the UDP, this method will perform sanity checks on
*dv* and on the returned fitness vector. A successful call of this method will increase the internal fitness
evaluation counter (see :func:`~pygmo.problem.get_fevals()`).

The ``fitness()`` method of the UDP must be able to take as input the decision vector as a 1D NumPy array, and it must
return the fitness vector as an iterable Python object (e.g., 1D NumPy array, list, tuple, etc.).

Args:
    dv (array-like object): the decision vector (chromosome) to be evaluated

Returns:
    1D NumPy float array: the fitness of *dv*

Raises:
    ValueError: if either the length of *dv* differs from the value returned by :func:`~pygmo.problem.get_nx()`, or
      the length of the returned fitness vector differs from the value returned by :func:`~pygmo.problem.get_nf()`
    unspecified: any exception thrown by the ``fitness()`` method of the UDP, or by failures at the intersection
      between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string problem_get_bounds_docstring()
{
    return R"(get_bounds()

Box-bounds.

This method will invoke the ``get_bounds()`` method of the UDP to return the box-bounds
:math:`(\mathbf{lb}, \mathbf{ub})` of the problem. Infinities in the bounds are allowed.

The ``get_bounds()`` method of the UDP must return the box-bounds as a tuple of 2 elements,
the lower bounds vector and the upper bounds vector, which must be represented as iterable Python objects (e.g.,
1D NumPy arrays, lists, tuples, etc.). The box-bounds returned by the UDP are checked upon the construction
of a :class:`~pygmo.problem`.

Returns:
    ``tuple``: a tuple of two 1D NumPy float arrays representing the lower and upper box-bounds of the problem

Raises:
    unspecified: any exception thrown by the invoked method of the underlying C++ class, or failures at the
      intersection between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string problem_get_nobj_docstring()
{
    return R"(get_nobj()

Number of objectives.

This method will return :math:`n_{obj}`, the number of objectives of the problem.

The optional ``get_nobj()`` method of the UDP must return the number of objectives as an ``int``.
If the UDP does not implement the ``get_nobj()`` method, a single-objective optimizaztion problem
will be assumed. The number of objectives returned by the UDP is checked upon the construction
of a :class:`~pygmo.problem`.

Returns:
    ``int``: the number of objectives of the problem

)";
}

std::string problem_get_nx_docstring()
{
    return R"(get_nx()

Dimension of the problem.

This method will return :math:`n_{x}`, the dimension of the problem as established by the length of
the bounds returned by :func:`~pygmo.problem.get_bounds()`.

Returns:
    ``int``: the dimension of the problem

)";
}

std::string problem_get_nix_docstring()
{
    return R"(get_nix()

Integer dimension of the problem.

This method will return :math:`n_{ix}`, the integer dimension of the problem.

The optional ``get_nix()`` method of the UDP must return the problem's integer dimension as an ``int``.
If the UDP does not implement the ``get_nix()`` method, a zero integer dimension will be assumed.
The integer dimension returned by the UDP is checked upon the construction
of a :class:`~pygmo.problem`.

Returns:
    ``int``: the integer dimension of the problem

)";
}

std::string problem_get_ncx_docstring()
{
    return R"(get_ncx()

Continuous dimension of the problem.

This method will return :math:`n_{cx}`, the continuous dimension of the problem.

Returns:
    ``int``: the continuous dimension of the problem

)";
}

std::string problem_get_nf_docstring()
{
    return R"(get_nf()

Dimension of the fitness.

This method will return :math:`n_{f}`, the dimension of the fitness, which is the sum of
:math:`n_{obj}`, :math:`n_{ec}` and :math:`n_{ic}`.

Returns:
    ``int``: the dimension of the fitness

)";
}

std::string problem_get_nec_docstring()
{
    return R"(get_nec()

Number of equality constraints.

This method will return :math:`n_{ec}`, the number of equality constraints of the problem.

The optional ``get_nec()`` method of the UDP must return the number of equality constraints as an ``int``.
If the UDP does not implement the ``get_nec()`` method, zero equality constraints will be assumed.
The number of equality constraints returned by the UDP is checked upon the construction
of a :class:`~pygmo.problem`.

Returns:
    ``int``: the number of equality constraints of the problem

)";
}

std::string problem_get_nic_docstring()
{
    return R"(get_nic()

Number of inequality constraints.

This method will return :math:`n_{ic}`, the number of inequality constraints of the problem.

The optional ``get_nic()`` method of the UDP must return the number of inequality constraints as an ``int``.
If the UDP does not implement the ``get_nic()`` method, zero inequality constraints will be assumed.
The number of inequality constraints returned by the UDP is checked upon the construction
of a :class:`~pygmo.problem`.

Returns:
    ``int``: the number of inequality constraints of the problem

)";
}

std::string problem_get_nc_docstring()
{
    return R"(get_nc()

Total number of constraints.

This method will return the sum of the output of :func:`~pygmo.problem.get_nic()` and
:func:`~pygmo.problem.get_nec()` (i.e., the total number of constraints).

Returns:
    ``int``: the total number of constraints of the problem

)";
}

std::string problem_c_tol_docstring()
{
    return R"(Constraints tolerance.

This property contains an array of ``float`` that are used when checking for constraint feasibility.
The dimension of the array is :math:`n_{ec} + n_{ic}` (i.e., the total number of constraints), and
the array is zero-filled on problem construction.

This property can also be set via a scalar, instead of an array. In such case, all the tolerances
will be set to the provided scalar value.

Returns:
    1D NumPy float array: the constraints' tolerances

Raises:
    ValueError: if, when setting this property, the size of the input array differs from the number
      of constraints of the problem or if any element of the array is negative or NaN
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import problem, hock_schittkowsky_71 as hs71
    >>> prob = problem(hs71())
    >>> prob.c_tol
    array([0., 0.])
    >>> prob.c_tol = [1, 2]
    >>> prob.c_tol
    array([1., 2.])
    >>> prob.c_tol = .5
    >>> prob.c_tol
    array([0.5, 0.5])

)";
}

std::string problem_get_fevals_docstring()
{
    return R"(get_fevals()

Number of fitness evaluations.

Each time a call to :func:`~pygmo.problem.fitness()` successfully completes, an internal counter
is increased by one. The counter is initialised to zero upon problem construction and it is never
reset. Copy operations copy the counter as well.

Returns:
    ``int`` : the number of times :func:`~pygmo.problem.fitness()` was successfully called

)";
}

std::string problem_get_gevals_docstring()
{
    return R"(get_gevals()

Number of gradient evaluations.

Each time a call to :func:`~pygmo.problem.gradient()` successfully completes, an internal counter
is increased by one. The counter is initialised to zero upon problem construction and it is never
reset. Copy operations copy the counter as well.

Returns:
    ``int`` : the number of times :func:`~pygmo.problem.gradient()` was successfully called

)";
}

std::string problem_get_hevals_docstring()
{
    return R"(get_hevals()

Number of hessians evaluations.

Each time a call to :func:`~pygmo.problem.hessians()` successfully completes, an internal counter
is increased by one. The counter is initialised to zero upon problem construction and it is never
reset. Copy operations copy the counter as well.

Returns:
    ``int`` : the number of times :func:`~pygmo.problem.hessians()` was successfully called

)";
}

std::string problem_has_gradient_docstring()
{
    return R"(has_gradient()

Check if the gradient is available in the UDP.

This method will return ``True`` if the gradient is available in the UDP, ``False`` otherwise.

The availability of the gradient is determined as follows:

* if the UDP does not provide a ``gradient()`` method, then this method will always return ``False``;
* if the UDP provides a ``gradient()`` method but it does not provide a ``has_gradient()`` method,
  then this method will always return ``True``;
* if the UDP provides both a ``gradient()`` and a ``has_gradient()`` method, then this method will return
  the output of the ``has_gradient()`` method of the UDP.

The optional ``has_gradient()`` method of the UDP must return a ``bool``. For information on how to
implement the ``gradient()`` method of the UDP, see :func:`~pygmo.problem.gradient()`.

Returns:
    ``bool``: a flag signalling the availability of the gradient in the UDP

)";
}

std::string problem_gradient_docstring()
{
    return R"(gradient(dv)

Gradient.

This method will compute the gradient of the input decision vector *dv* by invoking
the ``gradient()`` method of the UDP. The ``gradient()`` method of the UDP must return
a sparse representation of the gradient: the :math:`k`-th term of the gradient vector
is expected to contain :math:`\frac{\partial f_i}{\partial x_j}`, where the pair :math:`(i,j)`
is the :math:`k`-th element of the sparsity pattern (collection of index pairs), as returned by
:func:`~pygmo.problem.gradient_sparsity()`.

If the UDP provides a ``gradient()`` method, this method will forward *dv* to the ``gradient()``
method of the UDP after sanity checks. The output of the ``gradient()`` method of the UDP will
also be checked before being returned. If the UDP does not provide a ``gradient()`` method, an
error will be raised. A successful call of this method will increase the internal gradient
evaluation counter (see :func:`~pygmo.problem.get_gevals()`).

The ``gradient()`` method of the UDP must be able to take as input the decision vector as a 1D NumPy
array, and it must return the gradient vector as an iterable Python object (e.g., 1D NumPy array,
list, tuple, etc.).

Args:
    dv (array-like object): the decision vector whose gradient will be computed

Returns:
    1D NumPy float array: the gradient of *dv*

Raises:
    ValueError: if either the length of *dv* differs from the value returned by :func:`~pygmo.problem.get_nx()`, or
      the returned gradient vector does not have the same size as the vector returned by
      :func:`~pygmo.problem.gradient_sparsity()`
    NotImplementedError: if the UDP does not provide a ``gradient()`` method
    unspecified: any exception thrown by the ``gradient()`` method of the UDP, or by failures at the intersection
      between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string problem_has_gradient_sparsity_docstring()
{
    return R"(has_gradient_sparsity()

Check if the gradient sparsity is available in the UDP.

This method will return ``True`` if the gradient sparsity is available in the UDP, ``False`` otherwise.

The availability of the gradient sparsity is determined as follows:

* if the UDP does not provide a ``gradient_sparsity()`` method, then this method will always return ``False``;
* if the UDP provides a ``gradient_sparsity()`` method but it does not provide a ``has_gradient_sparsity()``
  method, then this method will always return ``True``;
* if the UDP provides both a ``gradient_sparsity()`` method and a ``has_gradient_sparsity()`` method,
  then this method will return the output of the ``has_gradient_sparsity()`` method of the UDP.

The optional ``has_gradient_sparsity()`` method of the UDP must return a ``bool``. For information on how to
implement the ``gradient_sparsity()`` method of the UDP, see :func:`~pygmo.problem.gradient_sparsity()`.

.. note::

   Regardless of what this method returns, the :func:`~pygmo.problem.gradient_sparsity()` method will always
   return a sparsity pattern: if the UDP does not provide the gradient sparsity, pygmo will assume that the sparsity
   pattern of the gradient is dense. See :func:`~pygmo.problem.gradient_sparsity()` for more details.

Returns:
    ``bool``: a flag signalling the availability of the gradient sparsity in the UDP

)";
}

std::string problem_gradient_sparsity_docstring()
{
    return R"(gradient_sparsity()

Gradient sparsity pattern.

This method will return the gradient sparsity pattern of the problem. The gradient sparsity pattern is a lexicographically sorted
collection of the indices :math:`(i,j)` of the non-zero elements of :math:`g_{ij} = \frac{\partial f_i}{\partial x_j}`.

If :func:`~pygmo.problem.has_gradient_sparsity()` returns ``True``, then the ``gradient_sparsity()`` method of the
UDP will be invoked, and its result returned (after sanity checks). Otherwise, a a dense pattern is assumed and the
returned vector will be :math:`((0,0),(0,1), ... (0,n_x-1), ...(n_f-1,n_x-1))`.

The ``gradient_sparsity()`` method of the UDP must return either a 2D NumPy array of integers, or an iterable
Python object of any kind. Specifically:

* if the returned value is a NumPy array, its shape must be :math:`(n,2)` (with :math:`n \geq 0`),
* if the returned value is an iterable Python object, then its elements must in turn be iterable Python objects
  containing each exactly 2 elements representing the indices :math:`(i,j)`.

Returns:
    2D Numpy int array: the gradient sparsity pattern

Raises:
    ValueError: in the following cases:

      * the NumPy array returned by the UDP does not satisfy the requirements described above (e.g., invalid
        shape, dimensions, etc.),
      * at least one element of the returned iterable Python object does not consist of a collection of exactly
        2 elements,
      * if the sparsity pattern returned by the UDP is invalid (specifically, if it is not strictly sorted lexicographically,
        or if the indices in the pattern are incompatible with the properties of the problem, or if the size of the
        returned pattern is different from the size recorded upon construction)
    OverflowError: if the NumPy array returned by the UDP contains integer values which are negative or outside an
      implementation-defined range
    unspecified: any exception thrown by:

      * the underlying C++ function,
      * the ``PyArray_FROM_OTF()`` function from the NumPy C API,
      * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
        signatures, etc.)

)";
}

std::string problem_has_hessians_docstring()
{
    return R"(has_hessians()

Check if the hessians are available in the UDP.

This method will return ``True`` if the hessians are available in the UDP, ``False`` otherwise.

The availability of the hessians is determined as follows:

* if the UDP does not provide a ``hessians()`` method, then this method will always return ``False``;
* if the UDP provides a ``hessians()`` method but it does not provide a ``has_hessians()`` method,
  then this method will always return ``True``;
* if the UDP provides both a ``hessians()`` and a ``has_hessians()`` method, then this method will return
  the output of the ``has_hessians()`` method of the UDP.

The optional ``has_hessians()`` method of the UDP must return a ``bool``. For information on how to
implement the ``hessians()`` method of the UDP, see :func:`~pygmo.problem.hessians()`.

Returns:
    ``bool``: a flag signalling the availability of the hessians in the UDP

)";
}

std::string problem_hessians_docstring()
{
    return R"(hessians(dv)

Hessians.

This method will compute the hessians of the input decision vector *dv* by invoking
the ``hessians()`` method of the UDP. The ``hessians()`` method of the UDP must return
a sparse representation of the hessians: the element :math:`l` of the returned vector contains
:math:`h^l_{ij} = \frac{\partial f^2_l}{\partial x_i\partial x_j}` in the order specified by the
:math:`l`-th element of the hessians sparsity pattern (a vector of index pairs :math:`(i,j)`)
as returned by :func:`~pygmo.problem.hessians_sparsity()`. Since
the hessians are symmetric, their sparse representation contains only lower triangular elements.

If the UDP provides a ``hessians()`` method, this method will forward *dv* to the ``hessians()``
method of the UDP after sanity checks. The output of the ``hessians()`` method of the UDP will
also be checked before being returned. If the UDP does not provide a ``hessians()`` method, an
error will be raised. A successful call of this method will increase the internal hessians
evaluation counter (see :func:`~pygmo.problem.get_hevals()`).

The ``hessians()`` method of the UDP must be able to take as input the decision vector as a 1D NumPy
array, and it must return the hessians vector as an iterable Python object (e.g., list, tuple, etc.).

Args:
    dv (array-like object): the decision vector whose hessians will be computed

Returns:
    ``list`` of 1D NumPy float array: the hessians of *dv*

Raises:
    ValueError: if the length of *dv* differs from the value returned by :func:`~pygmo.problem.get_nx()`, or
      the length of returned hessians does not match the corresponding hessians sparsity pattern dimensions, or
      the size of the return value is not equal to the fitness dimension
    NotImplementedError: if the UDP does not provide a ``hessians()`` method
    unspecified: any exception thrown by the ``hessians()`` method of the UDP, or by failures at the intersection
      between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string problem_has_hessians_sparsity_docstring()
{
    return R"(has_hessians_sparsity()

Check if the hessians sparsity is available in the UDP.

This method will return ``True`` if the hessians sparsity is available in the UDP, ``False`` otherwise.

The availability of the hessians sparsity is determined as follows:

* if the UDP does not provide a ``hessians_sparsity()`` method, then this method will always return ``False``;
* if the UDP provides a ``hessians_sparsity()`` method but it does not provide a ``has_hessians_sparsity()``
  method, then this method will always return ``True``;
* if the UDP provides both a ``hessians_sparsity()`` method and a ``has_hessians_sparsity()`` method,
  then this method will return the output of the ``has_hessians_sparsity()`` method of the UDP.

The optional ``has_hessians_sparsity()`` method of the UDP must return a ``bool``. For information on how to
implement the ``hessians_sparsity()`` method of the UDP, see :func:`~pygmo.problem.hessians_sparsity()`.

.. note::

   Regardless of what this method returns, the :func:`~pygmo.problem.hessians_sparsity()` method will always
   return a sparsity pattern: if the UDP does not provide the hessians sparsity, pygmo will assume that the sparsity
   pattern of the hessians is dense. See :func:`~pygmo.problem.hessians_sparsity()` for more details.

Returns:
    ``bool``: a flag signalling the availability of the hessians sparsity in the UDP

)";
}

std::string problem_hessians_sparsity_docstring()
{
    return R"(hessians_sparsity()

Hessians sparsity pattern.

This method will return the hessians sparsity pattern of the problem. Each component :math:`l` of the hessians
sparsity pattern is a lexicographically sorted collection of the indices :math:`(i,j)` of the non-zero elements of
:math:`h^l_{ij} = \frac{\partial f^l}{\partial x_i\partial x_j}`. Since the Hessian matrix is symmetric, only
lower triangular elements are allowed.

If :func:`~pygmo.problem.has_hessians_sparsity()` returns ``True``, then the ``hessians_sparsity()`` method of the
UDP will be invoked, and its result returned (after sanity checks). Otherwise, a dense pattern is assumed and
:math:`n_f` sparsity patterns containing :math:`((0,0),(1,0), (1,1), (2,0) ... (n_x-1,n_x-1))` will be returned.

The ``hessians_sparsity()`` method of the UDP must return an iterable Python object of any kind. Each element of the
returned object will then be interpreted as a sparsity pattern in the same way as described in
:func:`~pygmo.problem.gradient_sparsity()`. Specifically:

* if the element is a NumPy array, its shape must be :math:`(n,2)` (with :math:`n \geq 0`),
* if the element is itself an iterable Python object, then its elements must in turn be iterable Python objects
  containing each exactly 2 elements representing the indices :math:`(i,j)`.

Returns:
    ``list`` of 2D Numpy int array: the hessians sparsity patterns

Raises:
    ValueError: in the following cases:

      * the NumPy arrays returned by the UDP do not satisfy the requirements described above (e.g., invalid
        shape, dimensions, etc.),
      * at least one element of a returned iterable Python object does not consist of a collection of exactly
        2 elements,
      * if a sparsity pattern returned by the UDP is invalid (specifically, if it is not strictly sorted lexicographically,
        if the indices in the pattern are incompatible with the properties of the problem or if the size of the pattern
        differs from the size recorded upon construction)
    OverflowError: if the NumPy arrays returned by the UDP contain integer values which are negative or outside an
      implementation-defined range
    unspecified: any exception thrown by:

      * the underlying C++ function,
      * the ``PyArray_FROM_OTF()`` function from the NumPy C API,
      * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
        signatures, etc.)

)";
}

std::string problem_set_seed_docstring()
{
    return R"(set_seed(seed)

Set the seed for the stochastic variables.

This method will set the seed to be used in the fitness function to instantiate
all stochastic variables. If the UDP provides a ``set_seed()`` method, then
its ``set_seed()`` method will be invoked. Otherwise, an error will be raised.
The *seed* parameter must be non-negative.

The ``set_seed()`` method of the UDP must be able to take an ``int`` as input parameter.

Args:
    seed (``int``): the desired seed value

Raises:
    NotImplementedError: if the UDP does not provide a ``set_seed()`` method
    OverflowError: if *seed* is negative
    unspecified: any exception raised by the ``set_seed()`` method of the UDP or failures at the intersection
      between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string problem_has_set_seed_docstring()
{
    return R"(has_set_seed()

Check if the ``set_seed()`` method is available in the UDP.

This method will return ``True`` if the ``set_seed()`` method is available in the UDP, ``False`` otherwise.

The availability of the ``set_seed()`` method is determined as follows:

* if the UDP does not provide a ``set_seed()`` method, then this method will always return ``False``;
* if the UDP provides a ``set_seed()`` method but it does not provide a ``has_set_seed()`` method,
  then this method will always return ``True``;
* if the UDP provides both a ``set_seed()`` and a ``has_set_seed()`` method, then this method will return
  the output of the ``has_set_seed()`` method of the UDP.

The optional ``has_set_seed()`` method of the UDP must return a ``bool``. For information on how to
implement the ``set_seed()`` method of the UDP, see :func:`~pygmo.problem.set_seed()`.

Returns:
    ``bool``: a flag signalling the availability of the ``set_seed()`` method in the UDP

)";
}

std::string problem_feasibility_f_docstring()
{
    return R"(feasibility_f(f)

This method will check the feasibility of a fitness vector *f* against the tolerances returned by
:attr:`~pygmo.problem.c_tol`.

Args:
    f (array-like object): a fitness vector

Returns:
    ``bool``: ``True`` if the fitness vector is feasible, ``False`` otherwise

Raises:
    ValueError: if the size of *f* is not the same as the output of
      :func:`~pygmo.problem.get_nf()`

)";
}

std::string problem_feasibility_x_docstring()
{
    return R"(feasibility_x(x)

This method will check the feasibility of the fitness corresponding to a decision vector *x* against
the tolerances returned by :attr:`~pygmo.problem.c_tol`.

.. note:

This will cause one fitness evaluation.

Args:
    dv (array-like object): a decision vector

Returns:
    ``bool``: ``True`` if *x* results in a feasible fitness, ``False`` otherwise

Raises:
     unspecified: any exception thrown by :func:`~pygmo.problem.feasibility_f()` or
       :func:`~pygmo.problem.fitness()`

)";
}

std::string problem_get_name_docstring()
{
    return R"(get_name()

Problem's name.

If the UDP provides a ``get_name()`` method, then this method will return the output of its ``get_name()`` method.
Otherwise, an implementation-defined name based on the type of the UDP will be returned.

The ``get_name()`` method of the UDP must return a ``str``.

Returns:
    ``str``: the problem's name

)";
}

std::string problem_get_extra_info_docstring()
{
    return R"(get_extra_info()

Problem's extra info.

If the UDP provides a ``get_extra_info()`` method, then this method will return the output of its ``get_extra_info()``
method. Otherwise, an empty string will be returned.

The ``get_extra_info()`` method of the UDP must return a ``str``.

Returns:
  ``str``: extra info about the UDP

Raises:
  unspecified: any exception thrown by the ``get_extra_info()`` method of the UDP

)";
}

std::string problem_get_thread_safety_docstring()
{
    return R"(get_thread_safety()

Problem's thread safety level.

This method will return a value of the enum :class:`pygmo.thread_safety` which indicates the thread safety level
of the UDP. Unlike in C++, in Python it is not possible to re-implement this method in the UDP. That is, for C++
UDPs, the returned value will be the value returned by the ``get_thread_safety()`` method of the UDP. For Python
UDPs, the returned value will be unconditionally :attr:`pygmo.thread_safety.none`.

Returns:
    a value of :class:`pygmo.thread_safety`: the thread safety level of the UDP

)";
}

std::string problem_get_best_docstring(const std::string &name)
{
    return R"(best_known()

The best known solution for the )"
           + name + R"( problem.

Returns:
    1D NumPy float array: the best known solution for the )"
           + name + R"( problem

)";
}

std::string translate_docstring()
{
    return R"(The translate meta-problem.

This meta-problem translates the whole search space of an input :class:`pygmo.problem` or 
user-defined problem (UDP) by a fixed translation vector. :class:`~pygmo.translate` objects 
are user-defined problems that can be used in the construction of a :class:`pygmo.problem`.
)";
}

std::string translate_translation_docstring()
{
    return R"(Translation vector.

This read-only property contains an array of ``float`` representing the translation vector used in the
construction of this problem.

Returns:
    1D NumPy float array: the translation vector

)";
}

std::string algorithm_docstring()
{
    return R"(__init__(uda = null_algorithm())

Algorithm class.

This class represents an optimization algorithm. An algorithm can be
stochastic, deterministic, population based, derivative-free, using hessians,
using gradients, a meta-heuristic, evolutionary, etc.. Via this class pygmo offers
a common interface to all types of algorithms that can be applied to find solution
to a generic matematical programming problem as represented by the
:class:`~pygmo.problem` class.

In order to define an optimizaztion algorithm in pygmo, the user must first define a class
whose methods describe the properties of the algorithm and implement its logic.
In pygmo, we refer to such a class as a **user-defined algorithm**, or UDA for short. Once
defined and instantiated, a UDA can then be used to construct an instance of this class,
:class:`~pygmo.algorithm`, which provides a generic interface to optimization algorithms.

Every UDA must implement at least the following method:

.. code-block:: python

   def evolve(self, pop):
     ...

The ``evolve()`` method takes as input a :class:`~pygmo.population`, and it is expected to return
a new population generated by the *evolution* (or *optimisation*) of the original population.

Additional optional methods can be implemented in a UDA:

.. code-block:: python

   def has_set_seed(self):
     ...
   def set_seed(self, s):
     ...
   def has_set_verbosity(self):
     ...
   def set_verbosity(self, l):
     ...
   def get_name(self):
     ...
   def get_extra_info(self):
     ...

See the documentation of the corresponding methods in this class for details on how the optional
methods in the UDA should be implemented and on how they are used by :class:`~pygmo.algorithm`.
Note that the exposed C++ algorithms can also be used as UDAs, even if they do not expose any of the
mandatory or optional methods listed above (see :ref:`here <py_algorithms>` for the
full list of UDAs already coded in pygmo).

This class is the Python counterpart of the C++ class :cpp:class:`pagmo::algorithm`.

Args:
    uda: a user-defined algorithm, either C++ or Python

Raises:
    NotImplementedError: if *uda* does not implement the mandatory method detailed above
    unspecified: any exception thrown by:

      * methods of the UDA invoked during construction,
      * the deep copy of the UDA,
      * the constructor of the underlying C++ class,
      * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
        signatures, etc.)

)";
}

std::string algorithm_evolve_docstring()
{
    return R"(evolve(pop)

This method will invoke the ``evolve()`` method of the UDA. This is where the core of the optimization
(*evolution*) is made.

Args:
    pop (:class:`~pygmo.population`): starting population

Returns:
    :class:`~pygmo.population`: evolved population

Raises:
    unspecified: any exception thrown by the ``evolve()`` method of the UDA or by failures at the
      intersection between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string algorithm_set_seed_docstring()
{
    return R"(set_seed(seed)

Set the seed for the stochastic evolution.

This method will set the seed to be used in the ``evolve()`` method of the UDA for all stochastic variables. If the UDA
provides a ``set_seed()`` method, then its ``set_seed()`` method will be invoked. Otherwise, an error will be
raised. The *seed* parameter must be non-negative.

The ``set_seed()`` method of the UDA must be able to take an ``int`` as input parameter.

Args:
    seed (``int``): the random seed

Raises:
    NotImplementedError: if the UDA does not provide a ``set_seed()`` method
    unspecified: any exception raised by the ``set_seed()`` method of the UDA or failures at the intersection
      between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string algorithm_has_set_seed_docstring()
{
    return R"(has_set_seed()

Check if the ``set_seed()`` method is available in the UDA.

This method will return ``True`` if the ``set_seed()`` method is available in the UDA, ``False`` otherwise.

The availability of the ``set_seed()`` method is determined as follows:

* if the UDA does not provide a ``set_seed()`` method, then this method will always return ``False``;
* if the UDA provides a ``set_seed()`` method but it does not provide a ``has_set_seed()`` method,
  then this method will always return ``True``;
* if the UDA provides both a ``set_seed()`` and a ``has_set_seed()`` method, then this method will return
  the output of the ``has_set_seed()`` method of the UDA.

The optional ``has_set_seed()`` method of the UDA must return a ``bool``. For information on how to
implement the ``set_seed()`` method of the UDA, see :func:`~pygmo.algorithm.set_seed()`.

Returns:
    ``bool``: a flag signalling the availability of the ``set_seed()`` method in the UDA

)";
}

std::string algorithm_set_verbosity_docstring()
{
    return R"(set_verbosity(level)

Set the verbosity of logs and screen output.

This method will set the level of verbosity for the algorithm. If the UDA provides a ``set_verbosity()`` method,
then its ``set_verbosity()`` method will be invoked. Otherwise, an error will be raised.

The exact meaning of the input parameter *level* is dependent on the UDA.

The ``set_verbosity()`` method of the UDA must be able to take an ``int`` as input parameter.

Args:
    level (``int``): the desired verbosity level

Raises:
    NotImplementedError: if the UDA does not provide a ``set_verbosity()`` method
    unspecified: any exception raised by the ``set_verbosity()`` method of the UDA or failures at the intersection
      between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string algorithm_has_set_verbosity_docstring()
{
    return R"(has_set_verbosity()

Check if the ``set_verbosity()`` method is available in the UDA.

This method will return ``True`` if the ``set_verbosity()`` method is available in the UDA, ``False`` otherwise.

The availability of the ``set_verbosity()`` method is determined as follows:

* if the UDA does not provide a ``set_verbosity()`` method, then this method will always return ``False``;
* if the UDA provides a ``set_verbosity()`` method but it does not provide a ``has_set_verbosity()`` method,
  then this method will always return ``True``;
* if the UDA provides both a ``set_verbosity()`` and a ``has_set_verbosity()`` method, then this method will return
  the output of the ``has_set_verbosity()`` method of the UDA.

The optional ``has_set_verbosity()`` method of the UDA must return a ``bool``. For information on how to
implement the ``set_verbosity()`` method of the UDA, see :func:`~pygmo.algorithm.set_verbosity()`.

Returns:
    ``bool``: a flag signalling the availability of the ``set_verbosity()`` method in the UDA

)";
}

std::string algorithm_get_name_docstring()
{
    return R"(get_name()

Algorithm's name.

If the UDA provides a ``get_name()`` method, then this method will return the output of its ``get_name()`` method.
Otherwise, an implementation-defined name based on the type of the UDA will be returned.

The ``get_name()`` method of the UDA must return a ``str``.

Returns:
    ``str``: the algorithm's name

)";
}

std::string algorithm_get_extra_info_docstring()
{
    return R"(get_extra_info()

Algorithm's extra info.

If the UDA provides a ``get_extra_info()`` method, then this method will return the output of its ``get_extra_info()``
method. Otherwise, an empty string will be returned.

The ``get_extra_info()`` method of the UDA must return a ``str``.

Returns:
  ``str``: extra info about the UDA

Raises:
  unspecified: any exception thrown by the ``get_extra_info()`` method of the UDA

)";
}

std::string algorithm_get_thread_safety_docstring()
{
    return R"(get_thread_safety()

Algorithm's thread safety level.

This method will return a value of the enum :class:`pygmo.thread_safety` which indicates the thread safety level
of the UDA. Unlike in C++, in Python it is not possible to re-implement this method in the UDA. That is, for C++
UDAs, the returned value will be the value returned by the ``get_thread_safety()`` method of the UDA. For Python
UDAs, the returned value will be unconditionally :attr:`pygmo.thread_safety.none`.

Returns:
    a value of :class:`pygmo.thread_safety`: the thread safety level of the UDA

)";
}

std::string generic_uda_inner_algorithm_docstring()
{

    return R"(Inner algorithm of the meta-algorithm.

This read-only property gives direct access to the :class:`~pygmo.algorithm` stored within this meta-algorithm.

Returns:
    :class:`~pygmo.algorithm`: a reference to the inner algorithm

)";
}

std::string generic_udp_inner_problem_docstring()
{

    return R"(Inner problem of the meta-problem.

This read-only property gives direct access to the :class:`~pygmo.problem` stored within this meta-problem.

Returns:
    :class:`~pygmo.problem`: a reference to the inner problem

)";
}

std::string mbh_docstring()
{
    return R"(Monotonic Basin Hopping (generalized).

Monotonic basin hopping, or simply, basin hopping, is an algorithm rooted in the idea of mapping
the objective function :math:`f(\mathbf x_0)` into the local minima found starting from :math:`\mathbf x_0`.
This simple idea allows a substantial increase of efficiency in solving problems, such as the Lennard-Jones
cluster or the MGA-1DSM interplanetary trajectory problem that are conjectured to have a so-called
funnel structure.

In pygmo we provide an original generalization of this concept resulting in a meta-algorithm that operates
on any :class:`pygmo.population` using any suitable user-defined algorithm (UDA). When a population containing a single
individual is used and coupled with a local optimizer, the original method is recovered.
The pseudo code of our generalized version is:

.. code-block:: none

   > Select a pygmo population
   > Select a UDA
   > Store best individual
   > while i < stop_criteria
   > > Perturb the population in a selected neighbourhood
   > > Evolve the population using the algorithm
   > > if the best individual is improved
   > > > increment i
   > > > update best individual
   > > else
   > > > i = 0

:class:`pygmo.mbh` is a user-defined algorithm (UDA) that can be used to construct :class:`pygmo.algorithm` objects.

See: https://arxiv.org/pdf/cond-mat/9803344.pdf for the paper introducing the basin hopping idea for a Lennard-Jones
cluster optimization.

See also the docs of the C++ class :cpp:class:`pagmo::mbh`.

)";
}

std::string mbh_get_seed_docstring()
{
    return R"(get_seed()

Get the seed value that was used for the construction of this :class:`~pygmo.mbh`.

Returns:
    ``int``: the seed value

)";
}

std::string mbh_get_verbosity_docstring()
{
    return R"(get_verbosity()

Get the verbosity level value that was used for the construction of this :class:`~pygmo.mbh`.

Returns:
    ``int``: the verbosity level

)";
}

std::string mbh_set_perturb_docstring()
{
    return R"(set_perturb(perturb)

Set the perturbation vector.

Args:
    perturb (array-like object): perturb the perturbation to be applied to each component

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g., type conversion errors,
      mismatched function signatures, etc.)

)";
}

std::string mbh_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity parameter
(by default nothing is logged) which can be set calling :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm` constructed
with an :class:`~pygmo.mbh`. A verbosity level ``N > 0`` will log one line at the end of each call to the inner algorithm.

Returns:
    ``list`` of ``tuples``: at each call of the inner algorithm, the values ``Fevals``, ``Best``, ``Violated``, ``Viol. Norm`` and ``Trial``, where:

    * ``Fevals`` (``int``), the number of fitness evaluations made
    * ``Best`` (``float``), the objective function of the best fitness currently in the population
    * ``Violated`` (``int``), the number of constraints currently violated by the best solution
    * ``Viol. Norm`` (``float``), the norm of the violation (discounted already by the constraints tolerance)
    * ``Trial`` (``int``), the trial number (which will determine the algorithm stop)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(mbh(algorithm(de(gen = 10))))
    >>> algo.set_verbosity(3)
    >>> prob = problem(cec2013(prob_id = 1, dim = 20))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Fevals:          Best:      Violated:    Viol. Norm:         Trial:
        440        25162.3              0              0              0
        880          14318              0              0              0
       1320        11178.2              0              0              0
       1760        6613.71              0              0              0
       2200        6613.71              0              0              1
       2640        6124.62              0              0              0
       3080        6124.62              0              0              1

See also the docs of the relevant C++ method :cpp:func:`pagmo::mbh::get_log()`.

)";
}

std::string mbh_get_perturb_docstring()
{
    return R"(get_perturb()

Get the perturbation vector.

Returns:
    1D NumPy float array: the perturbation vector

)";
}

std::string cstrs_self_adaptive_docstring()
{
    return R"(This meta-algorithm implements a constraint handling technique that allows the use of any user-defined algorithm
(UDA) able to deal with single-objective unconstrained problems, on single-objective constrained problems. The
technique self-adapts its parameters during each successive call to the inner UDA basing its decisions on the entire
underlying population. The resulting approach is an alternative to using the meta-problem :class:`~pygmo.unconstrain`
to transform the constrained fitness into an unconstrained fitness.

The self-adaptive constraints handling meta-algorithm is largely based on the ideas of Faramani and Wright but it
extends their use to any-algorithm, in particular to non generational, population based, evolutionary approaches where
a steady-state reinsertion is used (i.e. as soon as an individual is found fit it is immediately reinserted into the
population and will influence the next offspring genetic material).

Each decision vector is assigned an infeasibility measure :math:`\iota` which accounts for the normalized violation of
all the constraints (discounted by the constraints tolerance as returned by :attr:`pygmo.problem.c_tol`). The
normalization factor used :math:`c_{j_{max}}` is the maximum violation of the :math:`j` constraint.

As in the original paper, three individuals in the evolving population are then used to penalize the single
objective.

.. math::
   \begin{array}{rl}
   \check X & \mbox{: the best decision vector} \\
   \hat X & \mbox{: the worst decision vector} \\
   \breve X & \mbox{: the decision vector with the highest objective}
   \end{array}

The best and worst decision vectors are defined accounting for their infeasibilities and for the value of the
objective function. Using the above definitions the overall pseudo code can be summarized as follows:

.. code-block:: none

   > Select a pygmo.population (related to a single-objective constrained problem)
   > Select a UDA (able to solve single-objective unconstrained problems)
   > while i < iter
   > > Compute the normalization factors (will depend on the current population)
   > > Compute the best, worst, highest (will depend on the current population)
   > > Evolve the population using the UDA and a penalized objective
   > > Reinsert the best decision vector from the previous evolution

:class:`pygmo.cstrs_self_adaptive` is a user-defined algorithm (UDA) that can be used to construct :class:`pygmo.algorithm` objects.

.. note::

   Self-adaptive constraints handling implements an internal cache to avoid the re-evaluation of the fitness
   for decision vectors already evaluated. This makes the final counter of fitness evaluations somewhat unpredictable.
   The number of function evaluation will be bounded to *iters* times the fevals made by one call to the inner UDA. The
   internal cache is reset at each iteration, but its size will grow unlimited during each call to
   the inner UDA evolve method.

.. note::

   Several modification were made to the original Faramani and Wright ideas to allow their approach to work on
   corner cases and with any UDAs. Most notably, a violation to the :math:`j`-th  constraint is ignored if all
   the decision vectors in the population satisfy that particular constraint (i.e. if :math:`c_{j_{max}} = 0`).

.. note::

   The performances of :class:`~pygmo.cstrs_self_adaptive` are highly dependent on the particular inner
   algorithm employed and in particular to its parameters (generations / iterations).

.. seealso::

   Farmani, Raziyeh, and Jonathan A. Wright. "Self-adaptive fitness formulation for constrained optimization." IEEE
   Transactions on Evolutionary Computation 7.5 (2003): 445-455.

See also the docs of the C++ class :cpp:class:`pagmo::cstrs_self_adaptive`.

)";
}

std::string cstrs_self_adaptive_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity parameter
(by default nothing is logged) which can be set calling :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm` constructed
with an :class:`~pygmo.cstrs_self_adaptive`. A verbosity level of ``N > 0`` will log one line each ``N`` ``iters``.

Returns:
    ``list`` of ``tuples``: at each call of the inner algorithm, the values ``Iters``, ``Fevals``, ``Best``, ``Infeasibility``, 
    ``Violated``, ``Viol. Norm`` and ``N. Feasible``, where:

    * ``Iters`` (``int``), the number of iterations made (i.e. calls to the evolve method of the inner algorithm)
    * ``Fevals`` (``int``), the number of fitness evaluations made
    * ``Best`` (``float``), the objective function of the best fitness currently in the population
    * ``Infeasibility`` (``float``), the aggregated (and normalized) infeasibility value of ``Best``
    * ``Violated`` (``int``), the number of constraints currently violated by the best solution
    * ``Viol. Norm`` (``float``), the norm of the violation (discounted already by the constraints tolerance)
    * ``N. Feasible`` (``int``), the number of feasible individuals currently in the population.

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(cstrs_self_adaptive(iters = 20, algo = de(10)))
    >>> algo.set_verbosity(3)
    >>> prob = problem(cec2006(prob_id = 1))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Iter:        Fevals:          Best: Infeasibility:      Violated:    Viol. Norm:   N. Feasible:
        1              0       -96.5435        0.34607              4        177.705              0 i
        4            600       -96.5435       0.360913              4        177.705              0 i
        7           1200       -96.5435        0.36434              4        177.705              0 i
       10           1800       -96.5435       0.362307              4        177.705              0 i
       13           2400       -23.2502       0.098049              4        37.1092              0 i
       16           3000       -23.2502       0.071571              4        37.1092              0 i
       19           3600       -23.2502       0.257604              4        37.1092              0 i
    >>> uda = algo.extract(moead)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 0, -96.54346700540063, 0.34606950943401493, 4, 177.70482046341274, 0), (4, 600, ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::cstrs_self_adaptive::get_log()`.

)";
}

std::string null_algorithm_docstring()
{
    return R"(__init__()

The null algorithm.

An algorithm used in the default-initialization of :class:`pygmo.algorithm` and of the meta-algorithms.

)";
}

std::string null_problem_docstring()
{
    return R"(__init__(nobj = 1, nec = 0, nic = 0)

The null problem.

A problem used in the default-initialization of :class:`pygmo.problem` and of the meta-problems.

Args:
    nobj (``int``): the number of objectives
    nec  (``int``): the number of equality constraints
    nic  (``int``): the number of inequality constraintsctives

Raises:
    ValueError: if *nobj*, *nec*, *nic* are negative or greater than an implementation-defined value or if *nobj* is zero
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string rosenbrock_docstring()
{
    return R"(__init__(dim = 2)

The Rosenbrock problem.

Args:
    dim (``int``): problem dimension

Raises:
    OverflowError: if *dim* is negative or greater than an implementation-defined value
    ValueError: if *dim* is less than 2

See also the docs of the C++ class :cpp:class:`pagmo::rosenbrock`.

)";
}

std::string minlp_rastrigin_docstring()
{
    return R"(__init__(dim_c = 1, dim_i = 1)

The scalable MINLP Rastrigin problem.

Args:
    dim_c (``int``): MINLP continuous dimension
    dim_i (``int``): MINLP integer dimension

Raises:
    OverflowError: if *dim_c* / *dim_i* is negative or greater than an implementation-defined value
    ValueError: if *dim_c* + *dim_i* is less than 1

See also the docs of the C++ class :cpp:class:`pagmo::minlp_rastrigin`.

)";
}

std::string zdt_p_distance_docstring()
{
    return R"(p_distance(point)

p_distance(pop)

Convergence metric for decision vectors (0 = on the optimal front)

Introduced by Martens and Izzo, this metric is able to measure "a distance" of any point from
the pareto front of any DTLZ problem analytically.

Args:
    point (array-like object): decision vector for which the p distance is requested
    pop (:class:`~pygmo.population`): population for which the average p distance is requested

Returns:
    ``float``: the distance (or average distance) from the Pareto front

See also the docs of the C++ class :func:`~pygmo.zdt.p_distance()`

)";
}

std::string dtlz_p_distance_docstring()
{
    return R"(p_distance(point)

p_distance(pop)

Convergence metric for decision vectors (0 = on the optimal front)

Introduced by Martens and Izzo, this metric is able to measure "a distance" of any point from
the pareto front of any DTLZ problem analytically.

Args:
    point (array-like object): decision vector for which the p distance is requested
    pop (:class:`~pygmo.population`): population for which the average p distance is requested

Returns:
    ``float``: the distance (or average distance) from the Pareto front

See also the docs of the C++ class :func:`~pygmo.dtlz.p_distance()`

)";
}

std::string dtlz_docstring()
{
    return R"(__init__(prob_id = 1, dim = 5, fdim = 3, alpha = 100)

The DTLZ problem suite problem.

Args:
    prob_id (``int``): DTLZ problem id 
    dim (``int``): problem dimension
    fdim (``int``): number of objectives
    alpha (``int``): controls density of solutions (used only by DTLZ4)

Raises:
    OverflowError: if *prob_id*, *dim*, *fdim* or *alpha* are negative or greater than an implementation-defined value
    ValueError: if *prob_id* is not in [1..7], *fdim* is smaller than 2, *dim* is smaller or equal to *fdim*.

See also the docs of the C++ class :cpp:class:`pagmo::dtlz`.

)";
}

std::string cec2014_docstring()
{
    return R"(__init__(prob_id = 1, dim = 2)

.. versionadded:: 2.8

The CEC 2014 problem suite (continuous, box-bounded, single-objective problems)

Args:
    prob_id (int): problem id (one of [1..30])
    dim (int): number of dimensions (one of [2, 10, 20, 30, 50, 100])

Raises:
    OverflowError: if *dim* or *prob_id* are negative or greater than an implementation-defined value
    ValueError: if *prob_id* is not in [1..28] or if *dim* is not in [2, 10, 20, 30, 50, 100] or if *dim* is 2 and *prob_id* is in [17,18,19,20,21,22,29,30]

See also the docs of the C++ class :cpp:class:`pagmo::cec2014`.

)";
}

std::string cec2013_docstring()
{
    return R"(__init__(prob_id = 1, dim = 2)

The CEC 2013 problem suite (continuous, box-bounded, single-objective problems)

Args:
    prob_id (``int``): problem id (one of [1..28])
    dim (``int``): number of dimensions (one of [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

Raises:
    OverflowError: if *dim* or *prob_id* are negative or greater than an implementation-defined value
    ValueError: if *prob_id* is not in [1..28] or if *dim* is not in [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

See also the docs of the C++ class :cpp:class:`pagmo::cec2013`.

)";
}

std::string cec2009_docstring()
{
    return R"(__init__(prob_id = 1, is_constrained = false, dim = 30u)

The CEC 2009 problem suite (continuous, constrained, single-objective problems)

Args:
    prob_id (``int``): problem id (one of [1..10])
    is_constrained (``bool``): selects the constrained version of the problems 
    dim (``int``): problem dimension

Raises:
    OverflowError: if *prob_id* or *dim* are negative or greater than an implementation-defined value
    ValueError: if *prob_id* is not in [1..10] or if *dim* is zero

See also the docs of the C++ class :cpp:class:`pagmo::cec2009`.

)";
}

std::string cec2006_docstring()
{
    return R"(__init__(prob_id = 1)

The CEC 2006 problem suite (continuous, constrained, single-objective problems)

Args:
    prob_id (``int``): problem id (one of [1..24])

Raises:
    OverflowError: if *prob_id* is negative or greater than an implementation-defined value
    ValueError: if *prob_id* is not in [1..24]

See also the docs of the C++ class :cpp:class:`pagmo::cec2006`.

)";
}

std::string luksan_vlcek1_docstring()
{
    return R"(__init__(dim = 3)

Implementation of Example 5.1 in the report from Luksan and Vlcek.

The problem is also known as the Chained Rosenbrock function with trigonometric-exponential constraints.

Its formulation in pygmo is written as:

.. math::
   \begin{array}{rl}
   \mbox{find:} & -5 \le x_i \le 5, \forall i=1..n \\
   \mbox{to minimize: } & \sum_{i=1}^{n-1}\left[100\left(x_i^2-x_{i+1}\right)^2 + \left(x_i-1\right)^2\right] \\
   \mbox{subject to:} &
    3x_{k+1}^3+2x_{k+2}-5+\sin(x_{k+1}-x_{k+2})\sin(x_{k+1}+x_{k+2}) + \\
    & +4x_{k+1}-x_k\exp(x_k-x_{k+1})-3 = 0, \forall k=1..n-2
   \end{array}

See: Luksan, L., and Jan Vlcek. "Sparse and partially separable test problems for unconstrained and equality
constrained optimization." (1999). http://hdl.handle.net/11104/0123965

Args:
    dim (``int``): problem dimension

Raises:
    OverflowError: if *dim* is negative or greater than an implementation-defined value

See also the docs of the C++ class :cpp:class:`pagmo::luksan_vlcek1`.

)";
}

std::string generic_uda_get_seed_docstring()
{
    return R"(get_seed()

This method will return the random seed used internally by this uda.

Returns:
    ``int``: the random seed of the population
)";
}

std::string bee_colony_docstring()
{
    return R"(__init__(gen = 1, limit = 1, seed = random)

Artificial Bee Colony.

Args:
    gen (``int``): number of generations
    limit (``int``): maximum number of trials for abandoning a source
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen*, *limit* or *seed* is negative or greater than an implementation-defined value
    ValueError: if *limit* is not greater than 0

See also the docs of the C++ class :cpp:class:`pagmo::bee_colony`.

)";
}

std::string bee_colony_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.bee_colony`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Current best``, ``Best``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Current best`` (``float``), the best fitness currently in the population
    * ``Best`` (``float``), the best fitness found

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(bee_colony(gen = 500, limit = 20))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best: Current Best:
       1             40         261363         261363
     101           4040        112.237        267.969
     201           8040        20.8885        265.122
     301          12040        20.6076        20.6076
     401          16040         18.252        140.079
    >>> uda = algo.extract(bee_colony)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 40, 183727.83934515435, 183727.83934515435), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::bee_colony::get_log()`.

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
    ftol (``float``): stopping criteria on the f tolerance (default is 1e-6)
    xtol (``float``): stopping criteria on the x tolerance (default is 1e-6)
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
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.de`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Best``, ``dx``, ``df``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Best`` (``float``), the best fitness function currently in the population
    * ``dx`` (``float``), the norm of the distance to the population mean of the mutant vectors
    * ``df`` (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(de(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best:            dx:            df:
      1             20         162446        65.2891    1.78686e+06
    101           2020        198.402         8.4454        572.161
    201           4020        21.1155        2.60629        24.5152
    301           6020        6.67069        0.51811        1.99744
    401           8020        3.60022       0.583444       0.554511
    Exit condition -- generations = 500
    >>> uda = algo.extract(de)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 20, 162446.0185265718, 65.28911664703388, 1786857.8926660626), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::de::get_log()`.

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
    ValueError: if *start_range* is not in (0, 1], if *stop_range* is not in (*start_range*, 1] or if *reduction_coeff* is not in (0,1)

See also the docs of the C++ class :cpp:class:`pagmo::compass_search`.

)";
}

std::string compass_search_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.compass_search`. A verbosity larger than 0 implies one log line at each improvment of the fitness or
change in the search range.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values``Fevals``, ``Best``, ``Range``, where:

    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Best`` (``float``), the best fitness function currently in the population
    * ``Range`` (``float``), the range used to vary the chromosome (relative to the box bounds width)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(compass_search(max_fevals = 500))
    >>> algo.set_verbosity(1)
    >>> prob = problem(hock_schittkowsky_71())
    >>> pop = population(prob, 1)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Fevals:          Best:      Violated:    Viol. Norm:         Range:
          4        110.785              1        2.40583            0.5
         12        110.785              1        2.40583           0.25
         20        110.785              1        2.40583          0.125
         22        91.0454              1        1.01855          0.125
         25        96.2795              1       0.229446          0.125
         33        96.2795              1       0.229446         0.0625
         41        96.2795              1       0.229446        0.03125
         45         94.971              1       0.127929        0.03125
         53         94.971              1       0.127929       0.015625
         56        95.6252              1      0.0458521       0.015625
         64        95.6252              1      0.0458521      0.0078125
         68        95.2981              1      0.0410151      0.0078125
         76        95.2981              1      0.0410151     0.00390625
         79        95.4617              1     0.00117433     0.00390625
         87        95.4617              1     0.00117433     0.00195312
         95        95.4617              1     0.00117433    0.000976562
        103        95.4617              1     0.00117433    0.000488281
        111        95.4617              1     0.00117433    0.000244141
        115        95.4515              0              0    0.000244141
        123        95.4515              0              0     0.00012207
        131        95.4515              0              0    6.10352e-05
        139        95.4515              0              0    3.05176e-05
        143        95.4502              0              0    3.05176e-05
        151        95.4502              0              0    1.52588e-05
        159        95.4502              0              0    7.62939e-06
    Exit condition -- range: 7.62939e-06 <= 1e-05
    >>> uda = algo.extract(compass_search)
    >>> uda.get_log() # doctest: +SKIP
    [(4, 110.785345345, 1, 2.405833534534, 0.5), (12, 110.785345345, 1, 2.405833534534, 0.25) ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::compass_search::get_log()`.

)";
}

std::string sade_docstring()
{
    return R"(__init__(gen = 1, variant = 2, variant_adptv = 1, ftol = 1e-6, xtol = 1e-6, memory = False, seed = random)

Self-adaptive Differential Evolution.

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
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.sade`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Best``, ``F``, ``CR``, ``dx``, ``df``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Best`` (``float``), the best fitness function currently in the population
    * ``F`` (``float``), the value of the adapted paramter F used to create the best so far
    * ``CR`` (``float``), the value of the adapted paramter CR used to create the best so far
    * ``dx`` (``float``), the norm of the distance to the population mean of the mutant vectors
    * ``df`` (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(sade(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problems.rosenbrock(10)
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best:             F:            CR:            dx:            df:
      1             20         297060       0.690031       0.294769        44.1494    2.30584e+06
    101           2020        97.4258        0.58354       0.591527        13.3115        441.545
    201           4020        8.79247         0.6678        0.53148        17.8822        121.676
    301           6020        6.84774       0.494549        0.98105        12.2781        40.9626
    401           8020         4.7861       0.428741       0.743813        12.2938        39.7791
    Exit condition -- generations = 500
    >>> uda = algo.extract(sade)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 20, 297059.6296130389, 0.690031071850855, 0.29476914701127666, 44.14940516578547, 2305836.7422693395), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::sade::get_log()`.

)";
}

std::string nsga2_docstring()
{
    return R"(__init__(gen = 1, cr = 0.95, eta_c = 10, m = 0.01, eta_m = 10, seed = random)

Non dominated Sorting Genetic Algorithm (NSGA-II).

Args:
    gen (``int``): number of generations
    cr (``float``): crossover probability
    eta_c (``float``): distribution index for crossover
    m (``float``): mutation probability
    eta_m (``float``): distribution index for mutation
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen* or *seed* are negative or greater than an implementation-defined value
    ValueError: if either:

      * *cr* is not in [0,1[.
      * *eta_c* is not in [0,100[.
      * *m* is not in [0,1].
      * *eta_m* is not in [0,100[.
    
See also the docs of the C++ class :cpp:class:`pagmo::nsga2`.

)";
}

std::string nsga2_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.nsga2`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``ideal_point``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``ideal_point`` (1D numpy array), the ideal point of the current population (cropped to max 5 dimensions only in the screen output)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(nsga2(gen=100))
    >>> algo.set_verbosity(20)
    >>> pop = population(zdt(1), 40)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:        ideal1:        ideal2:
       1              0      0.0033062        2.44966
      21            800    0.000275601       0.893137
      41           1600    3.15834e-05        0.44117
      61           2400     2.3664e-05       0.206365
      81           3200     2.3664e-05       0.133305
    >>> uda = algo.extract(nsga2)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 0, array([ 0.0033062 ,  2.44965599])), (21, 800, array([  2.75601086e-04 ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::nsga2::get_log`.

)";
}

std::string moead_docstring()
{
    return R"(__init__(gen = 1, weight_generation = "grid", decomposition = "tchebycheff", neighbours = 20, CR = 1, F = 0.5, eta_m = 20, realb = 0.9, limit = 2, preserve_diversity = true, seed = random)

Multi Objective Evolutionary Algorithms by Decomposition (the DE variant)

Args:
    gen (``int``): number of generations
    weight_generation (``str``): method used to generate the weights, one of "grid", "low discrepancy" or "random"
    decomposition (``str``): method used to decompose the objectives, one of "tchebycheff", "weighted" or "bi"
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
    ValueError: if either:
    
      * *decomposition* is not one of 'tchebycheff', 'weighted' or 'bi'.
      * *weight_generation* is not one of 'random', 'low discrepancy' or 'grid'.
      * *CR* or *F* or *realb* are not in [0.,1.] 
      * *eta_m* is negative

See also the docs of the C++ class :cpp:class:`pagmo::moead`.

)";
}

std::string moead_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.moead`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``ADR``, ``ideal_point``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``ADF`` (``float``), Average Decomposed Fitness, that is the average across all decomposed problem of the single objective decomposed fitness along the corresponding direction
    * ``ideal_point`` (``array``), the ideal point of the current population (cropped to max 5 dimensions only in the screen output)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(moead(gen=500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(zdt())
    >>> pop = population(prob, 40)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:           ADF:        ideal1:        ideal2:
      1              0        32.5747     0.00190532        2.65685
    101           4000        5.67751    2.56736e-09       0.468789
    201           8000        5.38297    2.56736e-09      0.0855025
    301          12000        5.05509    9.76581e-10      0.0574796
    401          16000        5.13126    9.76581e-10      0.0242256
    >>> uda = algo.extract(moead)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 0, 32.574745630075874, array([  1.90532430e-03,   2.65684834e+00])), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::moead::get_log()`.

)";
}

std::string cmaes_docstring()
{
    return R"(__init__(gen = 1, cc = -1, cs = -1, c1 = -1, cmu = -1, sigma0 = 0.5, ftol = 1e-6, xtol = 1e-6, memory = False, force_bounds = False, seed = random)

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
    force_bounds (``bool``): when true the box bounds are enforced. The fitness will never be called outside the bounds but the covariance matrix adaptation  mechanism will worsen
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen* is negative or greater than an implementation-defined value
    ValueError: if *cc*, *cs*, *c1*, *cmu* are not in [0,1] or -1

See also the docs of the C++ class :cpp:class:`pagmo::cmaes`.

)";
}

std::string cmaes_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.cmaes`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Best``, ``dx``, ``df``, ``sigma``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Best`` (``float``), the best fitness function currently in the population
    * ``dx`` (``float``), the norm of the distance to the population mean of the mutant vectors
    * ``df`` (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual
    * ``sigma`` (``float``), the current step-size

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(cmaes(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best:            dx:            df:         sigma:
      1              0         173924        33.6872    3.06519e+06            0.5
    101           2000        92.9612       0.583942        156.921      0.0382078
    201           4000        8.79819       0.117574          5.101      0.0228353
    301           6000        4.81377      0.0698366        1.34637      0.0297664
    401           8000        1.04445      0.0568541       0.514459      0.0649836
    Exit condition -- generations = 500
    >>> uda = algo.extract(cmaes)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 0, 173924.2840042722, 33.68717961390855, 3065192.3843070837, 0.5), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::cmaes::get_log()`.

)";
}

std::string xnes_docstring()
{
    return R"(__init__(gen = 1, eta_mu = -1, eta_sigma = -1, eta_b = -1, sigma0 = -1, ftol = 1e-6, xtol = 1e-6, memory = False, force_bounds = False, seed = random)

Exponential Evolution Strategies.

Args:
    gen (``int``): number of generations
    eta_mu (``float``): learning rate for mean update (if -1 will be automatically selected to be 1)
    eta_sigma (``float``): learning rate for step-size update (if -1 will be automatically selected)
    eta_b (``float``): learning rate for the covariance matrix update (if -1 will be automatically selected)
    sigma0 (``float``):  the initial search width will be sigma0 * (ub - lb) (if -1 will be automatically selected to be 1)
    ftol (``float``): stopping criteria on the x tolerance
    xtol (``float``): stopping criteria on the f tolerance
    memory (``bool``): when true the adapted parameters are not reset between successive calls to the evolve method
    force_bounds (``bool``): when true the box bounds are enforced. The fitness will never be called outside the bounds but the covariance matrix adaptation  mechanism will worsen
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen* is negative or greater than an implementation-defined value
    ValueError: if *eta_mu*, *eta_sigma*, *eta_b*, *sigma0* are not in ]0,1] or -1

See also the docs of the C++ class :cpp:class:`pagmo::xnes`.

)";
}

std::string xnes_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.xnes`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Best``, ``dx``, ``df``, ``sigma``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Best`` (``float``), the best fitness function currently in the population
    * ``dx`` (``float``), the norm of the distance to the population mean of the mutant vectors
    * ``df`` (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual
    * ``sigma`` (``float``), the current step-size

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(xnes(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best:            dx:            df:         sigma:
      1              0         173924        33.6872    3.06519e+06            0.5
    101           2000        92.9612       0.583942        156.921      0.0382078
    201           4000        8.79819       0.117574          5.101      0.0228353
    301           6000        4.81377      0.0698366        1.34637      0.0297664
    401           8000        1.04445      0.0568541       0.514459      0.0649836
    Exit condition -- generations = 500
    >>> uda = algo.extract(xnes)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 0, 173924.2840042722, 33.68717961390855, 3065192.3843070837, 0.5), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::xnes::get_log()`.

)";
}

std::string de1220_docstring()
{
    return R"(__init__(gen = 1, allowed_variants = [2,3,7,10,13,14,15,16], variant_adptv = 1, ftol = 1e-6, xtol = 1e-6, memory = False, seed = random)

Self-adaptive Differential Evolution, pygmo flavour (pDE).
The adaptation of the mutation variant is added to :class:`~pygmo.sade`

Args:
    gen (``int``): number of generations
    allowed_variants (array-like object): allowed mutation variants, each one being a number in [1, 18]
    variant_adptv (``int``): *F* and *CR* parameter adaptation scheme to be used (one of 1..2)
    ftol (``float``): stopping criteria on the x tolerance (default is 1e-6)
    xtol (``float``): stopping criteria on the f tolerance (default is 1e-6)
    memory (``bool``): when true the adapted parameters *CR* anf *F* are not reset between successive calls to the evolve method
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

The following adaptation schemes for the parameters *F* and *CR* are available:

+--------------------------------------+--------------------------------------+
| 1 - jDE                              | 2 - iDE                              |
+--------------------------------------+--------------------------------------+

See also the docs of the C++ class :cpp:class:`pagmo::de1220`.

)";
}

std::string de1220_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.de1220`. A verbosity of N implies a log line each N generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Best``, ``F``, ``CR``, ``Variant``, ``dx``, ``df``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Best`` (``float``), the best fitness function currently in the population
    * ``F`` (``float``), the value of the adapted paramter F used to create the best so far
    * ``CR`` (``float``), the value of the adapted paramter CR used to create the best so far
    * ``Variant`` (``int``), the mutation variant used to create the best so far
    * ``dx`` (``float``), the norm of the distance to the population mean of the mutant vectors
    * ``df`` (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(de1220(gen = 500))
    >>> algo.set_verbosity(100)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best:             F:            CR:       Variant:            dx:            df:
        1             20         285653        0.55135       0.441551             16        43.9719    2.02379e+06
    101           2020        12.2721       0.127285      0.0792493             14        3.22986        106.764
    201           4020        5.72927       0.148337       0.777806             14        2.72177        4.10793
    301           6020        4.85084        0.12193       0.996191              3        2.95555        3.85027
    401           8020        4.20638       0.235997       0.996259              3        3.60338        4.49432
    Exit condition -- generations = 500
    >>> uda = algo.extract(de1220)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 20, 285652.7928977573, 0.551350234239449, 0.4415510963067054, 16, 43.97185788345982, 2023791.5123259544), ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::de1220::get_log()`.

)";
}

std::string pso_docstring()
{
    return R"(__init__(gen = 1, omega = 0.7298, eta1 = 2.05, eta2 = 2.05, max_vel = 0.5, variant = 5, neighb_type = 2, neighb_param = 4, memory = False, seed = random)

Particle Swarm Optimization

Args:
    gen (``int``): number of generations
    omega (``float``): inertia weight (or constriction factor)
    eta1 (``float``): social component
    eta2 (``float``): cognitive component
    max_vel (``float``): maximum allowed particle velocities (normalized with respect to the bounds width)
    variant (``int``): algoritmic variant
    neighb_type (``int``): swarm topology (defining each particle's neighbours)
    neighb_param (``int``): topology parameter (defines how many neighbours to consider)
    memory (``bool``): when true the velocities are not reset between successive calls to the evolve method
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen* or *seed* is negative or greater than an implementation-defined value
    ValueError: if *omega* is not in the [0,1] interval, if *eta1*, *eta2* are not in the [0,4] interval, if *max_vel* is not in ]0,1]
    ValueError: *variant* is not one of 1 .. 6, if *neighb_type* is not one of 1 .. 4 or if *neighb_param* is zero

The following variants can be selected via the *variant* parameter:

+-----------------------------------------+-----------------------------------------+
| 1 - Canonical (with inertia weight)     | 2 - Same social and cognitive rand.     |
+-----------------------------------------+-----------------------------------------+
| 3 - Same rand. for all components       | 4 - Only one rand.                      |
+-----------------------------------------+-----------------------------------------+
| 5 - Canonical (with constriction fact.) | 6 - Fully Informed (FIPS)               |
+-----------------------------------------+-----------------------------------------+


The following topologies are selected by *neighb_type*:

+--------------------------------------+--------------------------------------+
| 1 - gbest                            | 2 - lbest                            |
+--------------------------------------+--------------------------------------+
| 3 - Von Neumann                      | 4 - Adaptive random                  |
+--------------------------------------+--------------------------------------+

The topology determines (together with the topology parameter) which particles need to be considered
when computing the social component of the velocity update.

)";
}

std::string pso_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.pso`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``gbest``, ``Mean Vel.``, ``Mean lbest``, ``Avg. Dist.``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``gbest`` (``float``), the best fitness function found so far by the the swarm
    * ``Mean Vel.`` (``float``), the average particle velocity (normalized)
    * ``Mean lbest`` (``float``), the average fitness of the current particle locations
    * ``Avg. Dist.`` (``float``), the average distance between particles (normalized)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(pso(gen = 500))
    >>> algo.set_verbosity(50)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:         gbest:     Mean Vel.:    Mean lbest:    Avg. Dist.:
       1             40        72473.3       0.173892         677427       0.281744
      51           1040        135.867      0.0183806        748.001       0.065826
     101           2040        12.6726     0.00291046        84.9531      0.0339452
     151           3040         8.4405    0.000852588        33.5161      0.0191379
     201           4040        7.56943    0.000778264         28.213     0.00789202
     251           5040         6.8089     0.00435521        22.7988     0.00107112
     301           6040         6.3692    0.000289725        17.3763     0.00325571
     351           7040        6.09414    0.000187343        16.8875     0.00172307
     401           8040        5.78415    0.000524536        16.5073     0.00234197
     451           9040         5.4662     0.00018305        16.2339    0.000958182
    >>> uda = algo.extract(pso)
    >>> uda.get_log() # doctest: +SKIP
    [(1,40,72473.32713790605,0.1738915144248373,677427.3504996448,0.2817443174278134), (51,1040,...

See also the docs of the relevant C++ method :cpp:func:`pagmo::pso::get_log()`.

)";
}

//----------
std::string pso_gen_docstring()
{
    return R"(__init__(gen = 1, omega = 0.7298, eta1 = 2.05, eta2 = 2.05, max_vel = 0.5, variant = 5, neighb_type = 2, neighb_param = 4, memory = False, seed = random)

Particle Swarm Optimization (generational) is identical to :class:`~pygmo.pso`, but does update the velocities of each particle before new particle positions are computed (taking
into consideration all updated particle velocities). Each particle is thus evaluated on the same seed within a generation as opposed to the standard PSO which evaluates single particle 
at a time. Consequently, the generational PSO algorithm is suited for stochastic optimization problems.


Args:
    gen (``int``): number of generations
    omega (``float``): inertia weight (or constriction factor)
    eta1 (``float``): social component
    eta2 (``float``): cognitive component
    max_vel (``float``): maximum allowed particle velocities (normalized with respect to the bounds width)
    variant (``int``): algoritmic variant
    neighb_type (``int``): swarm topology (defining each particle's neighbours)
    neighb_param (``int``): topology parameter (defines how many neighbours to consider)
    memory (``bool``): when true the velocities are not reset between successive calls to the evolve method
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *gen* or *seed* is negative or greater than an implementation-defined value
    ValueError: if *omega* is not in the [0,1] interval, if *eta1*, *eta2* are not in the [0,1] interval, if *max_vel* is not in ]0,1]
    ValueError: *variant* is not one of 1 .. 6, if *neighb_type* is not one of 1 .. 4 or if *neighb_param* is zero

The following variants can be selected via the *variant* parameter:

+-----------------------------------------+-----------------------------------------+
| 1 - Canonical (with inertia weight)     | 2 - Same social and cognitive rand.     |
+-----------------------------------------+-----------------------------------------+
| 3 - Same rand. for all components       | 4 - Only one rand.                      |
+-----------------------------------------+-----------------------------------------+
| 5 - Canonical (with constriction fact.) | 6 - Fully Informed (FIPS)               |
+-----------------------------------------+-----------------------------------------+


The following topologies are selected by *neighb_type*:

+--------------------------------------+--------------------------------------+
| 1 - gbest                            | 2 - lbest                            |
+--------------------------------------+--------------------------------------+
| 3 - Von Neumann                      | 4 - Adaptive random                  |
+--------------------------------------+--------------------------------------+

The topology determines (together with the topology parameter) which particles need to be considered
when computing the social component of the velocity update.

)";
}

std::string pso_gen_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm`
constructed with a :class:`~pygmo.pso`. A verbosity of ``N`` implies a log line each ``N`` generations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``gbest``, ``Mean Vel.``, ``Mean lbest``, ``Avg. Dist.``, where:

    * ``Gen`` (``int``), generation number
    * ``Fevals`` (``int``), number of functions evaluation made
    * ``gbest`` (``float``), the best fitness function found so far by the the swarm
    * ``Mean Vel.`` (``float``), the average particle velocity (normalized)
    * ``Mean lbest`` (``float``), the average fitness of the current particle locations
    * ``Avg. Dist.`` (``float``), the average distance between particles (normalized)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(pso(gen = 500))
    >>> algo.set_verbosity(50)
    >>> prob = problem(rosenbrock(10))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:         gbest:     Mean Vel.:    Mean lbest:    Avg. Dist.:
       1             40        72473.3       0.173892         677427       0.281744
      51           1040        135.867      0.0183806        748.001       0.065826
     101           2040        12.6726     0.00291046        84.9531      0.0339452
     151           3040         8.4405    0.000852588        33.5161      0.0191379
     201           4040        7.56943    0.000778264         28.213     0.00789202
     251           5040         6.8089     0.00435521        22.7988     0.00107112
     301           6040         6.3692    0.000289725        17.3763     0.00325571
     351           7040        6.09414    0.000187343        16.8875     0.00172307
     401           8040        5.78415    0.000524536        16.5073     0.00234197
     451           9040         5.4662     0.00018305        16.2339    0.000958182
    >>> uda = algo.extract(pso)
    >>> uda.get_log() # doctest: +SKIP
    [(1,40,72473.32713790605,0.1738915144248373,677427.3504996448,0.2817443174278134), (51,1040,...

See also the docs of the relevant C++ method :cpp:func:`pagmo::pso::get_log()`.

)";
}

std::string simulated_annealing_docstring()
{
    return R"(__init__(Ts = 10., Tf = .1, n_T_adj = 10, n_range_adj = 10, bin_size = 10, start_range = 1., seed = random)

Simulated Annealing (Corana's version)

Args:
    Ts (``float``): starting temperature
    Tf (``float``): final temperature
    n_T_adj (``int``): number of temperature adjustments in the annealing schedule
    n_range_adj (``int``): number of adjustments of the search range performed at a constant temperature
    bin_size (``int``): number of mutations that are used to compute the acceptance rate
    start_range (``float``): starting range for mutating the decision vector
    seed (``int``): seed used by the internal random number generator (default is random)

Raises:
    OverflowError: if *n_T_adj*, *n_range_adj* or *bin_size* are negative or greater than an implementation-defined value
    ValueError: if *Ts* is not in (0, inf), if *Tf* is not in (0, inf), if *Tf* > *Ts* or if *start_range* is not in (0,1]
    ValueError: if *n_T_adj* is not strictly positive or if *n_range_adj* is not strictly positive

See also the docs of the C++ class :cpp:class:`pagmo::simulated_annealing`.
)";
}

std::string simulated_annealing_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. 
The log frequency depends on the verbosity parameter (by default nothing is logged) which can be set calling
the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm` constructed with a
:class:`~pygmo.simulated_annealing`. A verbosity larger than 0 will produce a log with one entry
each verbosity fitness evaluations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Fevals``, ``Best``, ``Current``, ``Mean range``, ``Temperature``, where:

    * ``Fevals`` (``int``), number of functions evaluation made
    * ``Best`` (``float``), the best fitness function found so far
    * ``Current`` (``float``), last fitness sampled
    * ``Mean range`` (``float``), the mean search range across the decision vector components (relative to the box bounds width)
    * ``Temperature`` (``float``), the current temperature

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(simulated_annealing(Ts=10., Tf=1e-5, n_T_adj = 100))
    >>> algo.set_verbosity(5000)
    >>> prob = problem(rosenbrock(dim = 10))
    >>> pop = population(prob, 1)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Fevals:          Best:       Current:    Mean range:   Temperature:
         57           5937           5937           0.48             10
      10033        9.50937        28.6775      0.0325519        2.51189
      15033        7.87389        14.3951      0.0131132        1.25893
      20033        7.87389        8.68616      0.0120491       0.630957
      25033        2.90084        4.43344     0.00676893       0.316228
      30033       0.963616        1.36471     0.00355931       0.158489
      35033       0.265868        0.63457     0.00202753      0.0794328
      40033        0.13894       0.383283     0.00172611      0.0398107
      45033       0.108051       0.169876    0.000870499      0.0199526
      50033      0.0391731      0.0895308     0.00084195           0.01
      55033      0.0217027      0.0303561    0.000596116     0.00501187
      60033     0.00670073     0.00914824    0.000342754     0.00251189
      65033      0.0012298     0.00791511    0.000275182     0.00125893
      70033     0.00112816     0.00396297    0.000192117    0.000630957
      75033    0.000183055     0.00139717    0.000135137    0.000316228
      80033    0.000174868     0.00192479    0.000109781    0.000158489
      85033       7.83e-05    0.000494225    8.20723e-05    7.94328e-05
      90033    5.35153e-05    0.000120148    5.76009e-05    3.98107e-05
      95033    5.35153e-05    9.10958e-05    3.18624e-05    1.99526e-05
      99933    2.34849e-05    8.72206e-05    2.59215e-05    1.14815e-05
    >>> uda = algo.extract(simulated_annealing)
    >>> uda.get_log() # doctest: +SKIP
    [(57, 5936.999957947842, 5936.999957947842, 0.47999999999999987, 10.0), (10033, ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::simulated_annealing::get_log()`.

)";
}

std::string random_decision_vector_docstring()
{
    return R"(random_decision_vector_docstring(lb, ub, nix = 0)

Creates a random decision vector within some bounds using pygmo's global rng. If
both the lower and upper bounds are finite numbers, then the :math:`i`-th
component of the randomly generated pagmo::vector_double will be such that
:math:`lb_i \le x_i < ub_i`. If :math:`lb_i == ub_i` then :math:`lb_i` is
returned. If an integer part *nix* is specified then the last *nix* components
are guaranteed to be integers within the specified (integer) bounds.

Args:
    lb (array-like object): the lower bounds
    ub (array-like object): the upper bounds
    nix (``int``): the integer size

Raises:
    ValueError: if *nix* is negative or greater than an implementation-defined value
    ValueError: if *lb* and *ub* are malformed (unequal lenght, zero size, *nix* larger than ``len(lb)`` or bounds are not integers in their last *nix* components)
    TypeError: if *lb* or *ub* cannot be converted to a vector of floats

Returns:
    1D NumPy float array: the random decision vector
)";
}

std::string decompose_docstring()
{
    return R"(The decompose meta-problem.

This meta-problem *decomposes* a multi-objective input user-defined problem,
resulting in a single-objective user-defined problem with a fitness function combining the
original fitness functions. In particular, three different *decomposition methods* are here
made available:

* weighted decomposition,
* Tchebycheff decomposition,
* boundary interception method (with penalty constraint).

In the case of :math:`n` objectives, we indicate with: :math:`\mathbf f(\mathbf x) = [f_1(\mathbf x), \ldots, f_n(\mathbf
x)]` the vector containing the original multiple objectives, with: :math:`\boldsymbol \lambda = (\lambda_1, \ldots,
\lambda_n)` an :math:`n`-dimensional weight vector and with: :math:`\mathbf z^* = (z^*_1, \ldots, z^*_n)`
an :math:`n`-dimensional reference point. We also ussume :math:`\lambda_i > 0, \forall i=1..n` and :math:`\sum_i \lambda_i =
1`.

The decomposed problem is thus a single objective optimization problem having the following single objective,
according to the decomposition method chosen:

* weighted decomposition: :math:`f_d(\mathbf x) = \boldsymbol \lambda \cdot \mathbf f`,
* Tchebycheff decomposition: :math:`f_d(\mathbf x) = \max_{1 \leq i \leq m} \lambda_i \vert f_i(\mathbf x) - z^*_i \vert`,
* boundary interception method (with penalty constraint): :math:`f_d(\mathbf x) = d_1 + \theta d_2`,


where :math:`d_1 = (\mathbf f - \mathbf z^*) \cdot \hat {\mathbf i}_{\lambda}`,
:math:`d_2 = \vert (\mathbf f - \mathbf z^*) - d_1 \hat {\mathbf i}_{\lambda})\vert` and
:math:`\hat {\mathbf i}_{\lambda} = \frac{\boldsymbol \lambda}{\vert \boldsymbol \lambda \vert}`.

.. note:

The reference point :math:`z^*` is often taken as the ideal point and as such
it may be allowed to change during the course of the optimization / evolution. The argument adapt_ideal activates
this behaviour so that whenever a new ideal point is found :math:`z^*` is adapted accordingly.

.. note:

The use of :class:`~pygmo.decompose` discards gradients and hessians so that if the original user defined problem
implements them, they will not be available in the decomposed problem. The reason for this behaviour is that
the Tchebycheff decomposition is not differentiable. Also, the use of this class was originally intended for
derivative-free optimization.

See: "Q. Zhang -- MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"

See: https://en.wikipedia.org/wiki/Multi-objective_optimization#Scalarizing
)";
}

std::string decompose_decompose_fitness_docstring()
{
    return R"(decompose_fitness(f, weight, ref_point)

Returns the decomposed fitness vector.

Args:
    f (array-like object): fitness vector
    weight (array-like object): the weight to be used in the decomposition
    ref_point (array-like object): the reference point to be used if either ``'tchebycheff'`` or ``'bi'`` was
      indicated as a decomposition method (its value is ignored if ``'weighted'`` was indicated)

Returns:
    1D NumPy float array: the decomposed fitness vector

Raises:
    ValueError: if *f*, *weight* and *ref_point* have different sizes
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string decompose_original_fitness_docstring()
{
    return R"(original_fitness(x)

Fitness of the original problem.

Returns the fitness of the original multi-objective problem used to construct the decomposed problem.

Args:
    x (array-like object): input decision vector

Returns:
    1D NumPy float array: the fitness of the original multi-objective problem

Raises:
    unspecified: any exception thrown by the original fitness computation, or by failures at the
      intersection between C++ and Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string decompose_z_docstring()
{
    return R"(Current reference point.

This read-only property contains the reference point to be used for the decomposition. This is only
used for Tchebycheff and boundary interception decomposition methods.

.. note:

The reference point is adapted at each call of the fitness.

Returns:
    1D NumPy float array: the reference point

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string unconstrain_docstring()
{
    return R"(The unconstrain meta-problem.

This meta-problem transforms a constrained problem into an unconstrained problem applying one of the following methods:

* Death penalty: simply penalizes all objectives by the same high value if the fitness vector is infeasible.
* Kuri's death penalty: defined by Angel Kuri Morales et al., penalizes all objectives according to the rate of satisfied constraints.
* Weighted violations penalty: penalizes all objectives by the weighted sum of the constraint violations.
* Ignore the constraints: simply ignores the constraints.
* Ignore the objectives: ignores the objectives and defines as a new single objective the overall constraints violation (i.e. the sum of the L2 norms of the equalities and inequalities violations)

.. note:

The use of :class:`~pygmo.unconstrain` discards gradients and hessians so that if the original user defined problem
implements them, they will not be available in the unconstrained problem. The reason for this behaviour is that,
in general, the methods implemented may not be differentiable. Also, the use of this class was originally intended for
derivative-free optimization.

See: Coello Coello, C. A. (2002). Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: 
a survey of the state of the art. Computer methods in applied mechanics and engineering, 191(11), 1245-1287.

See: Kuri Morales, A. and Quezada, C.C. A Universal eclectic genetic algorithm for constrained optimization,
Proceedings 6th European Congress on Intelligent Techniques & Soft Computing, EUFIT'98, 518-522, 1998.

)";
}

std::string fast_non_dominated_sorting_docstring()
{
    return R"(fast_non_dominated_sorting(points)

Runs the fast non dominated sorting algorithm on the input *points*

Args:
    points (2d-array-like object): the input points

Raises:
    ValueError: if *points* is malformed
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    ``tuple``: (*ndf*, *dl*, *dc*, *ndr*), where:

    * *ndf* (``list`` of 1D NumPy int array): the non dominated fronts
    * *dl* (``list`` of 1D NumPy int array): the domination list
    * *dc* (1D NumPy int array): the domination count
    * *ndr* (1D NumPy int array): the non domination ranks

Examples:
    >>> import pygmo as pg
    >>> ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = [[0,1],[-1,3],[2.3,-0.2],[1.1,-0.12],[1.1, 2.12],[-1.1,-1.1]])
)";
}

std::string pareto_dominance_docstring()
{
    return R"(pareto_dominance(obj1, obj2)

Returns ``True`` if *obj1* Pareto dominates *obj2*, ``False`` otherwise. Minimization
is assumed.

Each pair of corresponding elements in  *obj1* and *obj2* is compared: if all
elements in *obj1* are less or equal to the corresponding element in *obj2*,
but at least one is different, ``True`` will be returned. Otherwise, ``False`` will be returned.

Args:
    obj1 (array-like object): the first list of objectives
    obj2 (array-like object): the second list of objectives

Raises:
    ValueError: if the dimensions of *obj1* and *obj2* are different
    TypeError: if *obj1* or *obj2* cannot be converted to a vector of vector floats

Returns:
    ``bool``: ``True`` if *obj1* is dominating *obj2*, ``False`` otherwise.

Examples:
    >>> import pygmo as pg
    >>> pg.pareto_dominance(obj1 = [1,2], obj2 = [2,2])
    True

)";
}

std::string non_dominated_front_2d_docstring()
{
    return R"(non_dominated_front_2d(points)

Finds the non dominated front of a set of two dimensional objectives. Complexity is :math:`\mathcal{O}(N \log N)`
and is thus lower than the complexity of calling :func:`~pygmo.fast_non_dominated_sorting()`

See: Jensen, Mikkel T. "Reducing the run-time complexity of multiobjective EAs: The NSGA-II and other algorithms."
IEEE Transactions on Evolutionary Computation 7.5 (2003): 503-515.

Args:
    points (2d-array-like object): the input points

Raises:
    ValueError: if *points* contain anything else than 2 dimensional objectives
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    (``list`` of 1D NumPy int array): the non dominated fronts

Examples:
    >>> import pygmo as pg
    >>> pg.non_dominated_front_2d(points = [[0,5],[1,4],[2,3],[3,2],[4,1],[2,2]])
    array([0, 1, 5, 4], dtype=uint64)
)";
}

std::string crowding_distance_docstring()
{
    return R"(non_dominated_front_2d(points)

An implementation of the crowding distance. Complexity is :math:`O(M N \log N)` where :math:`M` is the number of
objectives and :math:`N` is the number of individuals. The function assumes *points* contain a non-dominated front. 
Failiure to meet this condition will result in undefined behaviour.

See: Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm for multi-objective
optimization: NSGA-II." Parallel problem solving from nature PPSN VI. Springer Berlin Heidelberg, 2000.

Args:
    points (2d-array-like object): the input points

Raises:
    ValueError: if *points* does not contain at least two points, or is malformed
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    (``list`` of 1D NumPy int array): the non dominated fronts

Examples:
    >>> import pygmo as pg
    >>> pg.crowding_distance(points = [[0,5],[1,4],[2,3],[3,2],[4,1]])
    array([ inf,   1.,   1.,   1.,  inf])
)";
}

std::string sort_population_mo_docstring()
{
    return R"(sort_population_mo(points)

Sorts a multi-objective, unconstrained,  population (intended here as a 2D array-like
containing objective vectors) with respect to the following strict ordering:

- :math:`f_1 \prec f_2` if the non domination ranks are such that :math:`i_1 < i_2`. In case :math:`i_1 = i_2`,
   then :math:`f_1 \prec f_2` if the crowding distances are such that :math:`d_1 > d_2`.

Complexity is :math:`\mathcal{O}(M N^2)` where :math:`M` is the size of the objective vector and :math:`N` is the number of individuals.

.. note::

   This function will also work for single objective optimization, i.e. with objective vector
   of size 1, in which case, though, it is more efficient to sort using python built-in sorting methods.

Args:
    points (2d-array-like object): the input objective vectors

Raises:
    unspecified: all exceptions thrown by pagmo::fast_non_dominated_sorting and pagmo::crowding_distance
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    (``list`` of 1D NumPy int array): the indexes of the sorted objectives vectors.

Examples:
    >>> import pygmo as pg
    >>> pop = pg.population(prob = pg.dtlz(prob_id = 3, dim=10, fdim=4), size = 20)
    >>> pg.sort_population_mo(points = pop.get_f()) # doctest: +SKIP
    array([ 4,  7, 14, 15, 16, 18,  9, 13,  5,  3,  6,  2, 12,  0,  1, 19, 17, 8, 10, 11], dtype=uint64)
)";
}

std::string select_best_N_mo_docstring()
{
    return R"(select_best_N_mo(points, N)

Returns (unordered) the best N individuals out of a multi-objective, unconstrained population, (intended here
as a 2D array-like containing objective vectors). The strict ordering used is the same as that defined
in :func:`~pygmo.sort_population_mo()`

Complexity is :math:`\mathcal{O}(M N^2)` where :math:`M` is the number of objectives and :math:`N` is the number of individuals.

While the complexity is the same as that of :func:`~pygmo.sort_population_mo()`, this function is to be preferred when 
possible in that it avoids to compute the crowidng distance for all individuals and only computes it for the last
non-dominated front containing individuals included in the best N.

Args:
    points (2d-array-like object): the input objective vectors
    N (``int``): The size of the returned list of bests.

Raises:
    unspecified: all exceptions thrown by :cpp:func:`pagmo::fast_non_dominated_sorting()` and :cpp:func:`pagmo::crowding_distance()`
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    (``list`` of 1D NumPy int array): the indexes of the *N* best objectives vectors.

Examples:
    >>> import pygmo as pg
    >>> pop = pg.population(prob = pg.dtlz(prob_id = 3, dim=10, fdim=4), size = 20)
    >>> pg.select_best_N_mo(points = pop.get_f(), N = 13) # doctest: +SKIP
    array([ 2,  3,  4,  5,  6,  7,  9, 12, 13, 14, 15, 16, 18], dtype=uint64)
)";
}

std::string decompose_objectives_docstring()
{
    return R"(decompose_objectives(objs, weights, ref_point, method)

Decomposes a vector of objectives.

A vector of objectives is reduced to one only objective using a decomposition technique.

Three different possibilities for *method* are here made available:

- weighted decomposition,
- Tchebycheff decomposition,
- boundary interception method (with penalty constraint).

In the case of :math:`n` objectives, we indicate with: :math:`\mathbf f(\mathbf x) = [f_1(\mathbf x), \ldots,
f_n(\mathbf x)]` the vector containing the original multiple objectives, with: :math:`\boldsymbol \lambda =
(\lambda_1, \ldots, \lambda_n)` an :math:`n`-dimensional weight vector and with: :math:`\mathbf z^* = (z^*_1, \ldots,
z^*_n)` an :math:`n`-dimensional reference point. We also ussume :math:`\lambda_i > 0, \forall i=1..n` and :math:`\sum_i
\lambda_i = 1`.

The resulting single objective is thus defined as:

- weighted decomposition: :math:`f_d(\mathbf x) = \boldsymbol \lambda \cdot \mathbf f`
- Tchebycheff decomposition: :math:`f_d(\mathbf x) = \max_{1 \leq i \leq m} \lambda_i \vert f_i(\mathbf x) - z^*_i \vert`
- boundary interception method (with penalty constraint): :math:`f_d(\mathbf x) = d_1 + \theta d_2`

where :math:`d_1 = (\mathbf f - \mathbf z^*) \cdot \hat {\mathbf i}_{\lambda}` ,
:math:`d_2 = \vert (\mathbf f - \mathbf z^*) - d_1 \hat {\mathbf i}_{\lambda})\vert` , and 
:math:`\hat {\mathbf i}_{\lambda} = \frac{\boldsymbol \lambda}{\vert \boldsymbol \lambda \vert}`

Args:
    objs (array-like object): the objective vectors
    weights (array-like object): the weights :math:`\boldsymbol \lambda`
    ref_point (array-like object): the reference point :math:`\mathbf z^*` . It is not used if *method* is ``"weighted"``
    method (``string``): the decomposition method: one of ``"weighted"``, ``"tchebycheff"`` or ``"bi"``

Raises:
    ValueError: if *objs*, *weight* and *ref_point* have different sizes or if *method* is not one of ``"weighted"``, ``"tchebycheff"`` or ``"bi"``.
    TypeError: if *weights* or *ref_point* or *objs* cannot be converted to a vector of floats.

Returns:
    1D NumPy float array:  a one dimensional array containing the decomposed objective.

Examples:
    >>> import pygmo as pg
    >>> pg.decompose_objectives(objs = [1,2,3], weights = [0.1,0.1,0.8], ref_point=[5,5,5], method = "weighted") # doctest: +SKIP
    array([ 2.7])
)";
}

std::string decomposition_weights_docstring()
{
    return R"(decomposition_weights(n_f, n_w, method, seed)

Generates the requested number of weight vectors to be used to decompose a multi-objective problem. Three methods are available:

- ``"grid"`` generates weights on an uniform grid. This method may only be used when the number of requested weights to be genrated is such that a uniform grid is indeed possible. 
  In two dimensions this is always the case, but in larger dimensions uniform grids are possible only in special cases
- ``"random"`` generates weights randomly distributing them uniformly on the simplex (weights are such that :math:`\sum_i \lambda_i = 1`) 
- ``"low discrepancy"`` generates weights using a low-discrepancy sequence to, eventually, obtain a better coverage of the Pareto front. Halton sequence is used since
  low dimensionalities are expected in the number of objectives (i.e. less than 20), hence Halton sequence is deemed as appropriate.

.. note::  
   All methods are guaranteed to generate weights on the simplex (:math:`\sum_i \lambda_i = 1`). All weight generation methods are guaranteed
   to generate the canonical weights [1,0,0,...], [0,1,0,..], ... first.
 
Args:
    n_f (``int``): the objective vectors
    n_w (``int``): the weights :math:`\boldsymbol \lambda`
    method (``string``): the reference point :math:`\mathbf z^*`. It is not used if *method* is ``"weighted"``
    seed (``int``): the decomposition method: one of ``"weighted"``, ``"tchebycheff"`` or ``"bi"``

Raises:
    OverflowError: if *n_f*, *n_w* or *seed* are negative or greater than an implementation-defined value
    ValueError: if *n_f* and *n_w* are not compatible with the selected weight generation method or if *method* is not
    one of ``"grid"``, ``"random"`` or ``"low discrepancy"``


Returns:
    1D NumPy float array:  the weights

Examples:
    >>> import pygmo as pg
    >>> pg.decomposition_weights(n_f = 2, n_w = 6, method = "low discrepancy", seed = 33) # doctest: +SKIP
    array([[ 1.   ,  0.   ],
           [ 0.   ,  1.   ],
           [ 0.25 ,  0.75 ],
           [ 0.75 ,  0.25 ],
           [ 0.125,  0.875],
           [ 0.625,  0.375]])
)";
}

std::string nadir_docstring()
{
    return R"(nadir(points)

Computes the nadir point of a set of points, i.e objective vectors. The nadir is that point that has the maximum
value of the objective function in the points of the non-dominated front.

Complexity is :math:`\mathcal{O}(MN^2)` where :math:`M` is the number of objectives and :math:`N` is the number of points.

Args:
    points (2d-array-like object): the input points

Raises:
    ValueError: if *points* is malformed
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    1D NumPy float array: the nadir point

)";
}

std::string ideal_docstring()
{
    return R"(ideal(points)

Computes the ideal point of a set of points, i.e objective vectors. The ideal point is that point that has, in each 
component, the minimum value of the objective functions of the input points.

Complexity is :math:`\mathcal{O}(MN)` where :math:`M` is the number of objectives and :math:`N` is the number of points.

Args:
    points (2d-array-like object): the input points

Raises:
    ValueError: if *points* is malformed
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    1D NumPy float array: the ideal point

)";
}

std::string compare_fc_docstring()
{
    return R"(compare_fc(f1, f2, nec, tol)

Compares two fitness vectors in a single-objective, constrained, case.

The following strict ordering is used:

- :math:`f_1 \prec f_2` if :math:`f_1` is feasible and :math:`f_2` is not.
- :math:`f_1 \prec f_2` if :math:`f_1` is they are both infeasible, but :math:`f_1`
  violates less constraints than :math:`f_2`, or in case they both violate the same
  number of constraints, if the :math:`L_2` norm of the overall constraint violation
  is smaller.
- :math:`f_1 \prec f_2` if both fitness vectors are feasible and the objective value
  in :math:`f_1` is smaller than the objectve value in :math:`f_2`

.. note::
   the fitness vectors are assumed to contain exactly one objective, \p neq equality constraints and the rest (if any) inequality constraints.

Args:
    f1 (array-like object): the first fitness vector
    f2 (array-like object): the second fitness vector
    nec (``int``): the number of equality consraints in the fitness vectors
    tol (array-like object): tolerances to be accounted for in the constraints

Raises:
    OverflowError: if *nec* is negative or greater than an implementation-defined value
    ValueError: if *f1* and *f2* do not have equal size :math:`n`, if *f1* does not have at least size 1, 
    if *neq* is larger than :math:`n-1` (too many constraints) or if the size of *tol* is not :math:`n - 1`
    TypeError: if *f1*, *f2* or *tol* cannot be converted to a vector of floats

Returns:
    ``bool``: ``True`` if *f1* is better than *f2*, ``False`` otherwise.

Examples:
    >>> import pygmo as pg
    >>> pg.compare_fc(f1 = [1,1,1], f2 = [1,2.1,-1.2], nec = 1, tol = [0]*2)
    False
)";
}

std::string sort_population_con_docstring()
{
    return R"(sort_population_con(input_f, nec, tol)

Sorts a population (intended here as a 2D array-like
containing fitness vectors) assuming a single-objective, constrained case. 

The following strict ordering is used (same as the one used in :func:`pygmo.compare_fc()`):

- :math:`f_1 \prec f_2` if :math:`f_1` is feasible and :math:`f_2` is not.
- :math:`f_1 \prec f_2` if :math:`f_1` is they are both infeasible, but :math:`f_1`
  violates less constraints than :math:`f_2`, or in case they both violate the same
  number of constraints, if the :math:`L_2` norm of the overall constraint violation
  is smaller.
- :math:`f_1 \prec f_2` if both fitness vectors are feasible and the objective value
  in :math:`f_1` is smaller than the objectve value in :math:`f_2`

.. note::
   the fitness vectors are assumed to contain exactly one objective, \p neq equality constraints and the rest (if any) inequality constraints.

Args:
    input_f (2-D array-like object): the fitness vectors
    nec (``int``): the number of equality constraints in the fitness vectors
    tol (array-like object): tolerances to be accounted for in the constraints

Raises:
    OverflowError: if *nec* is negative or greater than an implementation-defined value
    ValueError: if the input fitness vectors do not have all the same size :math:`n >=1`, or if *neq* is larger than :math:`n-1` (too many constraints)
    or if the size of *tol* is not equal to :math:`n-1`
    TypeError: if *input_f* cannot be converted to a vector of vector of floats or *tol* cannot be converted to a vector of floats.

Returns:
    ``list`` of 1D NumPy int array: the indexes of the sorted fitnesses vectors.

Examples:
    >>> import pygmo as pg
    >>> idxs = pg.sort_population_con(input_f = [[1.2,0.1,-1],[0.2,1.1,1.1],[2,-0.5,-2]], nec = 1, tol = [1e-8]*2)
    >>> print(idxs)
    [0 2 1]
)";
}

std::string estimate_sparsity_docstring()
{
    return R"(estimate_sparsity(callable, x, dx = 1e-8)

Performs a numerical estimation of the sparsity pattern of same callable object by numerically
computing it around the input point *x* and detecting the components that are changed.

The *callable* must accept an iterable as input and return an array-like object

Note that estimate_sparsity may fail to detect the real sparsity as it only considers one variation around the input
point. It is of use, though, in tests or cases where its not possible to write the sparsity or where the user is
confident the estimate will be correct.

Args:
    callable (a callable object): The function we want to estimate sparsity (typically a fitness).
    x (array-like object): decision vector to use when testing for sparisty.
    dx (``float``): To detect the sparsity each component of *x* will be changed by :math:`\max(|x_i|,1) dx`.

Raises:
    unspecified: any exception thrown by the *callable* object when called on *x*.
    TypeError: if *x* cannot be converted to a vector of floats or *callable* is not callable.

Returns:
    2D NumPy float array: the sparsity_pattern of *callable* detected around *x*

Examples:
    >>> import pygmo as pg
    >>> def my_fun(x):
    ...     return [x[0]+x[3], x[2], x[1]]
    >>> pg.estimate_sparsity(callable = my_fun, x = [0.1,0.1,0.1,0.1], dx = 1e-8)
    array([[0, 0],
           [0, 3],
           [1, 2],
           [2, 1]])

)";
}

std::string estimate_gradient_docstring()
{
    return R"(estimate_gradient(callable, x, dx = 1e-8)

Performs a numerical estimation of the sparsity pattern of same callable object by numerically
computing it around the input point *x* and detecting the components that are changed.

The *callable* must accept an iterable as input and return an array-like object. The gradient returned will be dense
and contain, in the lexicographic order requested by :func:`~pygmo.problem.gradient()`, :math:`\frac{df_i}{dx_j}`.

The numerical approximation of each derivative is made by central difference, according to the formula:

.. math::
   \frac{df}{dx} \approx \frac{f(x+dx) - f(x-dx)}{2dx} + O(dx^2)

The overall cost, in terms of calls to *callable* will thus be :math:`n` where :math:`n` is the size of *x*.

Args:
    callable (a callable object): The function we want to estimate sparsity (typically a fitness).
    x (array-like object): decision vector to use when testing for sparisty.
    dx (``float``): To detect the sparsity each component of *x* will be changed by :math:`\max(|x_i|,1) dx`.

Raises:
    unspecified: any exception thrown by the *callable* object when called on *x*.
    TypeError: if *x* cannot be converted to a vector of floats or *callable* is not callable.

Returns:
    2D NumPy float array: the dense gradient of *callable* detected around *x*

Examples:
    >>> import pygmo as pg
    >>> def my_fun(x):
    ...     return [x[0]+x[3], x[2], x[1]]
    >>> pg.estimate_gradient(callable = my_fun, x = [0]*4, dx = 1e-8) # doctest: +NORMALIZE_WHITESPACE
    array([1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.])
)";
}

std::string estimate_gradient_h_docstring()
{
    return R"(estimate_gradient_h(callable, x, dx = 1e-8)

Performs a numerical estimation of the sparsity pattern of same callable object by numerically
computing it around the input point *x* and detecting the components that are changed.

The *callable* must accept an iterable as input and return an array-like object. The gradient returned will be dense
and contain, in the lexicographic order requested by :func:`~pygmo.problem.gradient`, :math:`\frac{df_i}{dx_j}`.

The numerical approximation of each derivative is made by central difference, according to the formula:

.. math::
   \frac{df}{dx} \approx \frac 32 m_1 - \frac 35 m_2 +\frac 1{10} m_3 + O(dx^6)

where:

.. math::
   m_i = \frac{f(x + i dx) - f(x-i dx)}{2i dx}

The overall cost, in terms of calls to *callable* will thus be 6:math:`n` where :math:`n` is the size of *x*.

Args:
    callable (a callable object): The function we want to estimate sparsity (typically a fitness).
    x (array-like object): decision vector to use when testing for sparisty.
    dx (``float``): To detect the sparsity each component of *x* will be changed by :math:`\max(|x_i|,1) dx`.

Raises:
    unspecified: any exception thrown by the *callable* object when called on *x*.
    TypeError: if *x* cannot be converted to a vector of floats or *callable* is not callable.

Returns:
    2D NumPy float array: the dense gradient of *callable* detected around *x*

Examples:
    >>> import pygmo as pg
    >>> def my_fun(x):
    ...     return [x[0]+x[3], x[2], x[1]]
    >>> pg.estimate_gradient_h(callable = my_fun, x = [0]*4, dx = 1e-2) # doctest: +NORMALIZE_WHITESPACE
    array([1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.])
)";
}

std::string set_global_rng_seed_docstring()
{
    return R"(set_global_rng_seed(seed)

In pygmo it is, in general, possible to control the seed of all random generators by a dedicated *seed* kwarg passed on via various
constructors. If no *seed* is passed pygmo randomly creates a seed for you using its global random number generator. 

This function allows to be able to reset the seed of auch a global random number generator. This can be useful to create a deterministic behaviour of pygmo easily. 

.. note::
   In complex parallel evolutions obtaining a deterministic behaviour is not possible even setting the global seed as
   pygmo implements an asynchronous model for parallel execution and the exact interplay between threads and processes cannot
   be reproduced deterministically.

Examples:
    >>> import pygmo as pg
    >>> pg.set_global_rng_seed(seed = 32)
    >>> pop = pg.population(prob = pg.ackley(5), size = 20)
    >>> print(pop.champion_f)
    [17.26891503]
    >>> pg.set_global_rng_seed(seed = 32)
    >>> pop = pg.population(prob = pg.ackley(5), size = 20)
    >>> print(pop.champion_f)
    [17.26891503]
    )";
}

std::string hvwfg_docstring()
{
    return R"(__init__(stop_dimension = 2)

The hypervolume algorithm from the Walking Fish Group (2011 version).

This object can be passed as parameter to the various methods of the 
class :class:`~pygmo.hypervolume` as it derives from the hidden base
class :class:`~pygmo._hv_algorithm`

Args:
    stop_dimension (``int``): the input population

Raises:
    OverflowError: if *stop_dimension* is negative or greater than an implementation-defined value

Examples:
    >>> import pygmo as pg
    >>> hv_algo = pg.hvwfg(stop_dimension = 2)

See also the docs of the C++ class :cpp:class:`pagmo::hvwfg`.

)";
}

std::string hv2d_docstring()
{
    return R"(__init__()

Exact hypervolume algorithm for two dimensional points.

This object can be passed as parameter to the various methods of the 
class :class:`~pygmo.hypervolume` as it derives from the hidden base
class :class:`~pygmo._hv_algorithm`

Examples:
    >>> import pygmo as pg
    >>> hv_algo = pg.hv2d()

See also the docs of the C++ class :cpp:class:`pagmo::hv2d`.

)";
}

std::string hv3d_docstring()
{
    return R"(__init__()

Exact hypervolume algorithm for three dimensional points.

This object can be passed as parameter to the various methods of the 
class :class:`~pygmo.hypervolume` as it derives from the hidden base
class :class:`~pygmo._hv_algorithm`

Examples:
    >>> import pygmo as pg
    >>> hv_algo = pg.hv3d()

See also the docs of the C++ class :cpp:class:`pagmo::hv3d`.

)";
}

std::string bf_approx_docstring()
{
    return R"(__init__()

Bringmann-Friedrich approximation method. Implementation of the Bringmann-Friedrich approximation scheme (FPRAS),
reduced to the special case of approximating the least contributor.

This object can be passed as parameter to the various methods of the 
class :class:`~pygmo.hypervolume` as it derives from the hidden base
class :class:`~pygmo._hv_algorithm`

Examples:
    >>> import pygmo as pg
    >>> hv_algo = pg.bf_approx()

See also the docs of the C++ class :cpp:class:`pagmo::bf_approx`.

)";
}

std::string bf_fpras_docstring()
{
    return R"(__init__(eps = 1e-2, delta = 1e-2, seed = random)

Bringmann-Friedrich approximation method. Implementation of the Bringmann-Friedrich approximation scheme (FPRAS),
reduced to the special case of approximating the hypervolume indicator.

This object can be passed as parameter to the various methods of the 
class :class:`~pygmo.hypervolume` as it derives from the hidden base
class :class:`~pygmo._hv_algorithm`

Examples:
    >>> import pygmo as pg
    >>> hv_algo = pg.bf_fpras(eps = 1e-2, delta = 1e-2)

See also the docs of the C++ class :cpp:class:`pagmo::bf_fpras`.

)";
}

std::string hv_init1_docstring()
{
    return R"(__init__(pop)

Constructor from population

Args:
    pop (:class:`~pygmo.population`): the input population

Raises:
    ValueError: if *pop* contains a single-objective or a constrained problem

Examples:
    >>> from pygmo import *
    >>> pop = population(prob = zdt(prob_id = 1), size = 20)
    >>> hv = hypervolume(pop = pop)

See also the docs of the C++ class :cpp:class:`pagmo::hypervolume`.

)";
}

std::string hv_init2_docstring()
{
    return R"(__init__(points)

Constructor from points

Args:
    points (2d array-like object): the points

Raises:
    ValueError: if *points* is inconsistent

Examples:
    >>> from pygmo import *
    >>> points = [[1,2],[0.5, 3],[0.1,3.1]]
    >>> hv = hypervolume(points = points)

See also the docs of the C++ class :cpp:class:`pagmo::hypervolume`.

)";
}

std::string hv_compute_docstring()
{
    return R"(hypervolume.compute(ref_point, hv_algo = auto)

Computes the hypervolume with the supplied algorithm. If no algorithm
is supplied,  then an exact hypervolume algorithm is automatically selected
specific for the point dimension.

Args:
    ref_point (2d array-like object): the points
    hv_algo (deriving from :class:`~pygmo._hv_algorithm`): hypervolume algorithm to be used

Returns:
    ``float``: the computed hypervolume assuming *ref_point* as reference point

Raises:
    ValueError: if *ref_point* is not dominated by the nadir point

See also the docs of the C++ class :cpp:func:`pagmo::hypervolume::compute`.

)";
}

std::string hv_contributions_docstring()
{
    return R"(hypervolume.contributions(ref_point, hv_algo = auto)

This method returns the exclusive contribution to the hypervolume of every point.
According to *hv_algo* this computation can be implemented optimally (as opposed to calling
for :func:`~pygmo.hypervolume.exclusive` in a loop).

Args:
    ref_point (2d array-like object): the points
    hv_algo (deriving from :class:`~pygmo._hv_algorithm`): hypervolume algorithm to be used

Returns:
    1D NumPy float array: the contribution of all points to the hypervolume

Raises:
    ValueError: if *ref_point* is not suitable

See also the docs of the C++ class :cpp:func:`pagmo::hypervolume::contributions`.

)";
}

std::string hv_exclusive_docstring()
{
    return R"(hypervolume.exclusive(idx, ref_point, hv_algo = auto)

Computes the exclusive contribution to the hypervolume of a particular point.

Args:
    idx (``int``): index of the point
    ref_point (array-like object): the reference point
    hv_algo (deriving from :class:`~pygmo._hv_algorithm`): hypervolume algorithm to be used


Returns:
    1D NumPy float array: the contribution of all points to the hypervolume

Raises:
    ValueError: if *ref_point* is not suitable or if *idx* is out of bounds
    OverflowError: if *idx* is negative or greater than an implementation-defined value

See also the docs of the C++ class :cpp:func:`pagmo::hypervolume::exclusive`.

)";
}

std::string hv_greatest_contributor_docstring()
{
    return R"(hypervolume.greatest_contributor(ref_point, hv_algo = auto)

Computes the point contributing the most to the total hypervolume.

Args:
    ref_point (array-like object): the reference point
    hv_algo (deriving from :class:`~pygmo._hv_algorithm`): hypervolume algorithm to be used

Raises:
    ValueError: if *ref_point* is not suitable

See also the docs of the C++ class :cpp:func:`pagmo::hypervolume::greatest_contributor`.

)";
}

std::string hv_least_contributor_docstring()
{
    return R"(hypervolume.least_contributor(ref_point, hv_algo = auto)

Computes the point contributing the least to the total hypervolume.

Args:
    ref_point (array-like object): the reference point
    hv_algo (deriving from :class:`~pygmo._hv_algorithm`): hypervolume algorithm to be used

Raises:
    ValueError: if *ref_point* is not suitable

See also the docs of the C++ class :cpp:func:`pagmo::hypervolume::least_contributor`.

)";
}

std::string hv_refpoint_docstring()
{
    return R"(hypervolume.refpoint(offset = 0)

Calculates a mock refpoint by taking the maximum in each dimension over all points saved in the hypervolume object.
The result is a point that is necessarily dominated by all other points, and thus can be used for hypervolume computations.

.. note:

This point is different from the one computed by :func:`~pygmo.nadir()` as only the non dominated front is considered
in that method (also its complexity is thus higher)

Args:
    offset (``float``): the reference point

Returns:
    1D NumPy float array: the reference point

See also the docs of the C++ class :cpp:func:`pagmo::hypervolume::refpoint`.

)";
}

std::string island_docstring()
{
    return R"(Island class.

In the pygmo jargon, an island is a class that encapsulates three entities:

* a user-defined island (**UDI**),
* an :class:`~pygmo.algorithm`,
* a :class:`~pygmo.population`.

Through the UDI, the island class manages the asynchronous evolution (or optimisation)
of its :class:`~pygmo.population` via the algorithm's :func:`~pygmo.algorithm.evolve()`
method. Depending on the UDI, the evolution might take place in a separate thread (e.g., if the UDI is a
:class:`~pygmo.thread_island`), in a separate process (e.g., if the UDI is a
:class:`~pygmo.mp_island`) or even in a separate machine (e.g., if the UDI is a
:class:`~pygmo.ipyparallel_island`). The evolution is always asynchronous (i.e., running in the
"background") and it is initiated by a call to the :func:`~pygmo.island.evolve()` method. At any
time the user can query the state of the island and fetch its internal data members. The user can explicitly
wait for pending evolutions to conclude by calling the :func:`~pygmo.island.wait()` and
:func:`~pygmo.island.wait_check()` methods. The status of ongoing evolutions in the island can be queried via
the :attr:`~pygmo.island.status` attribute.

Typically, pygmo users will employ an already-available UDI in conjunction with this class (see :ref:`here <py_islands>`
for a full list), but advanced users can implement their own UDI types. A user-defined island must implement
the following method:

.. code-block:: python

   def run_evolve(self, algo, pop):
     ...

The ``run_evolve()`` method of the UDI will use the input :class:`~pygmo.algorithm`'s
:func:`~pygmo.algorithm.evolve()` method to evolve the input :class:`~pygmo.population` and, once the evolution
is finished, it will return the algorithm used for the evolution and the evolved :class:`~pygmo.population`.

In addition to the mandatory ``run_evolve()`` method, a UDI may implement the following optional methods:

.. code-block:: python

   def get_name(self):
     ...
   def get_extra_info(self):
     ...

See the documentation of the corresponding methods in this class for details on how the optional
methods in the UDI are used by :class:`~pygmo.island`.

Note that, due to the asynchronous nature of :class:`~pygmo.island`, a UDI has certain requirements regarding
thread safety. Specifically, ``run_evolve()`` is always called in a separate thread of execution, and consequently:

* multiple UDI objects may be calling their own ``run_evolve()`` method concurrently,
* in a specific UDI object, any method from the public API of the UDI may be called while ``run_evolve()`` is
  running concurrently in another thread. Thus, UDI writers must ensure that actions such as copying
  the UDI, calling the optional methods (such as ``get_name()``), etc. can be safely performed while the island
  is evolving.

An island can be initialised in a variety of ways using keyword arguments:

* if the arguments list is empty, a default :class:`~pygmo.island` is constructed, containing a
  :class:`~pygmo.thread_island` UDI, a :class:`~pygmo.null_algorithm` algorithm and an empty
  population with problem type :class:`~pygmo.null_problem`;
* if the arguments list contains *algo*, *pop* and, optionally, *udi*, then the constructor will initialise
  an :class:`~pygmo.island` containing the specified algorithm, population and UDI. If the *udi* parameter
  is not supplied, the UDI type is chosen according to a heuristic which depends on the platform, the
  Python version and the supplied *algo* and *pop* parameters:

  * if *algo* and *pop*'s problem provide at least the :attr:`~pygmo.thread_safety.basic` thread safety guarantee,
    then :class:`~pygmo.thread_island` will be selected as UDI type;
  * otherwise, if the current platform is Windows or the Python version is at least 3.4, then :class:`~pygmo.mp_island`
    will be selected as UDI type, else :class:`~pygmo.ipyparallel_island` will be chosen;
* if the arguments list contains *algo*, *prob*, *size* and, optionally, *udi* and *seed*, then a :class:`~pygmo.population`
  will be constructed from *prob*, *size* and *seed*, and the construction will then proceed in the same way detailed
  above (i.e., *algo* and the newly-created population are used to initialise the island's algorithm and population,
  and the UDI, if not specified, will be chosen according to the heuristic detailed above).

If the keyword arguments list is invalid, a :exc:`KeyError` exception will be raised.

This class is the Python counterpart of the C++ class :cpp:class:`pagmo::island`.

)";
}

std::string island_evolve_docstring()
{
    return R"(evolve(n = 1)

Launch evolution.

This method will evolve the island’s :class:`~pygmo.population` using the island’s :class:`~pygmo.algorithm`.
The evolution happens asynchronously: a call to :func:`~pygmo.island.evolve()` will create an evolution task that
will be pushed to a queue, and then return immediately. The tasks in the queue are consumed by a separate thread of execution
managed by the :class:`~pygmo.island` object. Each task will invoke the ``run_evolve()`` method of the UDI *n*
times consecutively to perform the actual evolution. The island's algorithm and population will be updated at the
end of each ``run_evolve()`` invocation. Exceptions raised inside the tasks are stored within the island object,
and can be re-raised by calling :func:`~pygmo.island.wait_check()`.

It is possible to call this method multiple times to enqueue multiple evolution tasks, which will be consumed in a FIFO (first-in
first-out) fashion. The user may call :func:`~pygmo.island.wait()` or :func:`~pygmo.island.wait_check()` to block until all
tasks have been completed, and to fetch exceptions raised during the execution of the tasks. The :attr:`~pygmo.island.status`
attribute can be used to query the status of the asynchronous operations in the island.

Args:
     n (``int``): the number of times the ``run_evolve()`` method of the UDI will be called within the evolution task

Raises:
    OverflowError: if *n* is negative or larger than an implementation-defined value
    unspecified: any exception thrown by the underlying C++ method, or by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string island_wait_check_docstring()
{
    return R"(wait_check()

Block until evolution ends and re-raise the first stored exception.

If one task enqueued after the last call to :func:`~pygmo.island.wait_check()` threw an exception, the exception will be re-thrown
by this method. If more than one task enqueued after the last call to :func:`~pygmo.island.wait_check()` threw an exception,
this method will re-throw the exception raised by the first enqueued task that threw, and the exceptions
from all the other tasks that threw will be ignored.

Note that :func:`~pygmo.island.wait_check()` resets the status of the island: after a call to :func:`~pygmo.island.wait_check()`,
:attr:`~pygmo.island.status` will always return :attr:`pygmo.evolve_status.idle`.

Raises:
    unspecified: any exception thrown by evolution tasks or by the underlying C++ method

)";
}

std::string island_wait_docstring()
{
    return R"(wait()

This method will block until all the evolution tasks enqueued via :func:`~pygmo.island.evolve()` have been completed.
Exceptions thrown by the enqueued tasks can be re-raised via :func:`~pygmo.island.wait_check()`: they will **not** be
re-thrown by this method. Also, contrary to :func:`~pygmo.island.wait_check()`, this method will **not** reset the
status of the island: after a call to :func:`~pygmo.island.wait()`, :attr:`~pygmo.island.status` will always return
either :attr:`pygmo.evolve_status.idle` or :attr:`pygmo.evolve_status.idle_error`.

)";
}

std::string island_status_docstring()
{
    return R"(Status of the island.

This read-only property will return an :class:`~pygmo.evolve_status` flag indicating the current status of
asynchronous operations in the island. The flag will be:

* :attr:`~pygmo.evolve_status.idle` if the island is currently not evolving and no exceptions
  were thrown by evolution tasks since the last call to :func:`~pygmo.island.wait_check()`;
* :attr:`~pygmo.evolve_status.busy` if the island is evolving and no exceptions
  have (yet) been thrown by evolution tasks since the last call to :func:`~pygmo.island.wait_check()`;
* :attr:`~pygmo.evolve_status.idle_error` if the island is currently not evolving and at least one
  evolution task threw an exception since the last call to :func:`~pygmo.island.wait_check()`;
* :attr:`~pygmo.evolve_status.busy_error` if the island is currently evolving and at least one
  evolution task has already thrown an exception since the last call to :func:`~pygmo.island.wait_check()`.

Note that after a call to :func:`~pygmo.island.wait_check()`, :attr:`~pygmo.island.status` will always return
:attr:`~pygmo.evolve_status.idle`, and after a call to :func:`~pygmo.island.wait()`, :attr:`~pygmo.island.status`
will always return either :attr:`~pygmo.evolve_status.idle` or :attr:`~pygmo.evolve_status.idle_error`.

Returns:
    :class:`~pygmo.evolve_status`: a flag indicating the current status of asynchronous operations in the island

)";
}

std::string island_get_algorithm_docstring()
{
    return R"(get_algorithm()

Get the algorithm.

It is safe to call this method while the island is evolving.

Returns:
    :class:`~pygmo.algorithm`: a copy of the island's algorithm

Raises:
    unspecified: any exception thrown by the underlying C++ method

)";
}

std::string island_set_algorithm_docstring()
{
    return R"(set_algorithm(algo)

Set the algorithm.

It is safe to call this method while the island is evolving.

Args:
    algo (:class:`~pygmo.algorithm`): the algorithm that will be copied into the island

Raises:
    unspecified: any exception thrown by the underlying C++ method

)";
}

std::string island_get_population_docstring()
{
    return R"(get_population()

Get the population.

It is safe to call this method while the island is evolving.

Returns:
    :class:`~pygmo.population`: a copy of the island's population

Raises:
    unspecified: any exception thrown by the underlying C++ method

)";
}

std::string island_set_population_docstring()
{
    return R"(set_population(pop)

Set the population.

It is safe to call this method while the island is evolving.

Args:
    pop (:class:`~pygmo.population`): the population that will be copied into the island

Raises:
    unspecified: any exception thrown by the underlying C++ method

)";
}

std::string island_get_thread_safety_docstring()
{
    return R"(get_thread_safety()

It is safe to call this method while the island is evolving.

Returns:
    ``tuple``: a tuple containing the :class:`~pygmo.thread_safety` levels of the island's algorithm and problem

Raises:
    unspecified: any exception thrown by the underlying C++ method

)";
}

std::string island_get_name_docstring()
{
    return R"(get_name()

Island's name.

If the UDI provides a ``get_name()`` method, then this method will return the output of its ``get_name()`` method.
Otherwise, an implementation-defined name based on the type of the UDI will be returned.

It is safe to call this method while the island is evolving.

The ``get_name()`` method of the UDI must return a ``str``.

Returns:
    ``str``: the name of the UDI

Raises:
    unspecified: any exception thrown by the ``get_name()`` method of the UDI

)";
}

std::string island_get_extra_info_docstring()
{
    return R"(get_extra_info()

Island's extra info.

If the UDI provides a ``get_extra_info()`` method, then this method will return the output of its ``get_extra_info()``
method. Otherwise, an empty string will be returned.

It is safe to call this method while the island is evolving.

The ``get_extra_info()`` method of the UDI must return a ``str``.

Returns:
    ``str``: extra info about the UDI

Raises:
    unspecified: any exception thrown by the ``get_extra_info()`` method of the UDI

)";
}

std::string thread_island_docstring()
{
    return R"(__init__()

Thread island.

This class is a user-defined island (UDI) that will run evolutions directly inside
the separate thread of execution within :class:`pygmo.island`. Evolution tasks running on this
UDI must involve :class:`~pygmo.algorithm` and :class:`~pygmo.problem` instances
that provide at least the :attr:`~pygmo.thread_safety.basic` thread safety guarantee, otherwise
errors will be raised during the evolution.

See also the documentation of the corresponding C++ class :cpp:class:`pagmo::thread_island`.

)";
}

std::string archipelago_docstring()
{
    return R"(Archipelago.

An archipelago is a collection of :class:`~pygmo.island` objects which provides a convenient way to perform
multiple optimisations in parallel.

The interface of :class:`~pygmo.archipelago` mirrors partially the interface of :class:`~pygmo.island`: the
evolution is initiated by a call to :func:`~pygmo.archipelago.evolve()`, and at any time the user can query the
state of the archipelago and access its island members. The user can explicitly wait for pending evolutions
to conclude by calling the :func:`~pygmo.archipelago.wait()` and :func:`~pygmo.archipelago.wait_check()` methods.
The status of ongoing evolutions in the archipelago can be queried via the :attr:`~pygmo.archipelago.status` attribute.

)";
}

std::string archipelago_evolve_docstring()
{
    return R"(evolve(n = 1)

Evolve archipelago.

This method will call :func:`pygmo.island.evolve()` on all the islands of the archipelago.
The input parameter *n* will be passed to the invocations of :func:`pygmo.island.evolve()` for each island.
The :attr:`~pygmo.archipelago.status` attribute can be used to query the status of the asynchronous operations in the
archipelago.

Args:
     n (``int``): the parameter that will be passed to :func:`pygmo.island.evolve()`

Raises:
    unspecified: any exception thrown by :func:`pygmo.island.evolve()`

)";
}

std::string archipelago_status_docstring()
{
    return R"(Status of the archipelago.

This read-only property will return an :class:`~pygmo.evolve_status` flag indicating the current status of
asynchronous operations in the archipelago. The flag will be:

* :attr:`~pygmo.evolve_status.idle` if, for all the islands in the archipelago, :attr:`pygmo.island.status`
  returns :attr:`~pygmo.evolve_status.idle`;
* :attr:`~pygmo.evolve_status.busy` if, for at least one island in the archipelago, :attr:`pygmo.island.status`
  returns :attr:`~pygmo.evolve_status.busy`, and for no island :attr:`pygmo.island.status` returns an error status;
* :attr:`~pygmo.evolve_status.idle_error` if no island in the archipelago is busy and for at least one island
  :attr:`pygmo.island.status` returns :attr:`~pygmo.evolve_status.idle_error`;
* :attr:`~pygmo.evolve_status.busy_error` if, for at least one island in the archipelago, :attr:`pygmo.island.status`
  returns an error status and at least one island is busy.

Note that after a call to :func:`~pygmo.archipelago.wait_check()`, :attr:`pygmo.archipelago.status` will always return
:attr:`~pygmo.evolve_status.idle`, and after a call to :func:`~pygmo.archipelago.wait()`, :attr:`pygmo.archipelago.status`
will always return either :attr:`~pygmo.evolve_status.idle` or :attr:`~pygmo.evolve_status.idle_error`.

Returns:
    :class:`~pygmo.evolve_status`:  a flag indicating the current status of asynchronous operations in the archipelago

)";
}

std::string archipelago_wait_docstring()
{
    return R"(wait()

Block until all evolutions have finished.

This method will call :func:`pygmo.island.wait()` on all the islands of the archipelago. Exceptions thrown by island
evolutions can be re-raised via :func:`~pygmo.archipelago.wait_check()`: they will **not** be re-thrown by this method.
Also, contrary to :func:`~pygmo.archipelago.wait_check()`, this method will **not** reset the status of the archipelago:
after a call to :func:`~pygmo.archipelago.wait()`, the :attr:`~pygmo.archipelago.status` attribute will
always return either :attr:`pygmo.evolve_status.idle` or :attr:`pygmo.evolve_status.idle_error`.

)";
}

std::string archipelago_wait_check_docstring()
{
    return R"(wait_check()

Block until all evolutions have finished and raise the first exception that was encountered.

This method will call :func:`pygmo.island.wait_check()` on all the islands of the archipelago (following
the order in which the islands were inserted into the archipelago).
The first exception raised by :func:`pygmo.island.wait_check()` will be re-raised by this method,
and all the exceptions thrown by the other calls to :func:`pygmo.island.wait_check()` will be ignored.

Note that :func:`~pygmo.archipelago.wait_check()` resets the status of the archipelago: after a call to
:func:`~pygmo.archipelago.wait_check()`, the :attr:`~pygmo.archipelago.status` attribute will
always return :attr:`pygmo.evolve_status.idle`.

Raises:
    unspecified: any exception thrown by any evolution task queued in the archipelago's
      islands

)";
}

std::string archipelago_getitem_docstring()
{
    return R"(__getitem__(i)

This subscript operator can be used to access the *i*-th island of the archipelago (that is, the *i*-th island that was
inserted via :func:`~pygmo.archipelago.push_back()`).

Raises:
    IndexError: if *i* is greater than the size of the archipelago

)";
}

std::string archipelago_get_champions_f_docstring()
{
    return R"(get_champions_f()

Get the fitness vectors of the islands' champions.

Returns:
    ``list`` of 1D NumPy float arrays: the fitness vectors of the islands' champions

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g., type conversion errors,
      mismatched function signatures, etc.)

)";
}

std::string archipelago_get_champions_x_docstring()
{
    return R"(get_champions_x()

Get the decision vectors of the islands' champions.

Returns:
    ``list`` of 1D NumPy float arrays: the decision vectors of the islands' champions

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g., type conversion errors,
      mismatched function signatures, etc.)

)";
}

std::string nlopt_docstring()
{
    return R"(__init__(solver = "cobyla")

NLopt algorithms.

This user-defined algorithm wraps a selection of solvers from the
`NLopt <https://nlopt.readthedocs.io/en/latest/>`__ library, focusing on
local optimisation (both gradient-based and derivative-free). The complete list of supported
NLopt algorithms is:

* COBYLA,
* BOBYQA,
* NEWUOA + bound constraints,
* PRAXIS,
* Nelder-Mead simplex,
* sbplx,
* MMA (Method of Moving Asymptotes),
* CCSA,
* SLSQP,
* low-storage BFGS,
* preconditioned truncated Newton,
* shifted limited-memory variable-metric,
* augmented Lagrangian algorithm.

The desired NLopt solver is selected upon construction of an :class:`~pygmo.nlopt` algorithm. Various properties
of the solver (e.g., the stopping criteria) can be configured via class attributes. Multiple
stopping criteria can be active at the same time: the optimisation will stop as soon as at least one stopping criterion
is satisfied. By default, only the ``xtol_rel`` stopping criterion is active (see :attr:`~pygmo.nlopt.xtol_rel`).

All NLopt solvers support only single-objective optimisation, and, as usual in pygmo, minimisation
is always assumed. The gradient-based algorithms require the optimisation problem to provide a gradient.
Some solvers support equality and/or inequality constraints. The constraints' tolerances will
be set to those specified in the :class:`~pygmo.problem` being optimised (see :attr:`pygmo.problem.c_tol`).

In order to support pygmo's population-based optimisation model, the ``evolve()`` method will select
a single individual from the input :class:`~pygmo.population` to be optimised by the NLopt solver.
If the optimisation produces a better individual (as established by :func:`~pygmo.compare_fc()`),
the optimised individual will be inserted back into the population.
The selection and replacement strategies can be configured via the :attr:`~pygmo.nlopt.selection`
and :attr:`~pygmo.nlopt.replacement` attributes.

.. note::

   This user-defined algorithm is available only if pygmo was compiled with the ``PAGMO_WITH_NLOPT`` option
   enabled (see the :ref:`installation instructions <install>`).

.. seealso::

   The `NLopt website <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`__ contains a detailed description
   of each supported solver.

This constructor will initialise an :class:`~pygmo.nlopt` object which will use the NLopt algorithm specified by
the input string *solver*, the ``"best"`` individual selection strategy and the ``"best"`` individual
replacement strategy. *solver* is translated to an NLopt algorithm type according to the following
translation table:

================================  ====================================
*solver* string                   NLopt algorithm
================================  ====================================
``"cobyla"``                      ``NLOPT_LN_COBYLA``
``"bobyqa"``                      ``NLOPT_LN_BOBYQA``
``"newuoa"``                      ``NLOPT_LN_NEWUOA``
``"newuoa_bound"``                ``NLOPT_LN_NEWUOA_BOUND``
``"praxis"``                      ``NLOPT_LN_PRAXIS``
``"neldermead"``                  ``NLOPT_LN_NELDERMEAD``
``"sbplx"``                       ``NLOPT_LN_SBPLX``
``"mma"``                         ``NLOPT_LD_MMA``
``"ccsaq"``                       ``NLOPT_LD_CCSAQ``
``"slsqp"``                       ``NLOPT_LD_SLSQP``
``"lbfgs"``                       ``NLOPT_LD_LBFGS``
``"tnewton_precond_restart"``     ``NLOPT_LD_TNEWTON_PRECOND_RESTART``
``"tnewton_precond"``             ``NLOPT_LD_TNEWTON_PRECOND``
``"tnewton_restart"``             ``NLOPT_LD_TNEWTON_RESTART``
``"tnewton"``                     ``NLOPT_LD_TNEWTON``
``"var2"``                        ``NLOPT_LD_VAR2``
``"var1"``                        ``NLOPT_LD_VAR1``
``"auglag"``                      ``NLOPT_AUGLAG``
``"auglag_eq"``                   ``NLOPT_AUGLAG_EQ``
================================  ====================================

The parameters of the selected solver can be configured via the attributes of this class.

See also the docs of the C++ class :cpp:class:`pagmo::nlopt`.

.. seealso::

   The `NLopt website <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`__ contains a detailed
   description of each supported solver.

Args:
    solver (``str``): the name of the NLopt algorithm that will be used by this :class:`~pygmo.nlopt` object

Raises:
    RuntimeError: if the NLopt version is not at least 2
    ValueError: if *solver* is not one of the allowed algorithm names
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> nl = nlopt('slsqp')
    >>> nl.xtol_rel = 1E-6 # Change the default value of the xtol_rel stopping criterion
    >>> nl.xtol_rel # doctest: +SKIP
    1E-6
    >>> algo = algorithm(nl)
    >>> algo.set_verbosity(1)
    >>> prob = problem(luksan_vlcek1(20))
    >>> prob.c_tol = [1E-6] * 18 # Set constraints tolerance to 1E-6
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
       objevals:       objval:      violated:    viol. norm:
               1       95959.4             18        538.227 i
               2       89282.7             18        5177.42 i
               3         75580             18        464.206 i
               4         75580             18        464.206 i
               5       77737.6             18        1095.94 i
               6         41162             18        350.446 i
               7         41162             18        350.446 i
               8         67881             18        362.454 i
               9       30502.2             18        249.762 i
              10       30502.2             18        249.762 i
              11       7266.73             18        95.5946 i
              12        4510.3             18        42.2385 i
              13       2400.66             18        35.2507 i
              14       34051.9             18        749.355 i
              15       1657.41             18        32.1575 i
              16       1657.41             18        32.1575 i
              17       1564.44             18        12.5042 i
              18       275.987             14        6.22676 i
              19       232.765             12         12.442 i
              20       161.892             15        4.00744 i
              21       161.892             15        4.00744 i
              22       17.6821             11        1.78909 i
              23       7.71103              5       0.130386 i
              24       6.24758              4     0.00736759 i
              25       6.23325              1    5.12547e-05 i
              26        6.2325              0              0
              27       6.23246              0              0
              28       6.23246              0              0
              29       6.23246              0              0
              30       6.23246              0              0
    <BLANKLINE>
    Optimisation return status: NLOPT_XTOL_REACHED (value = 4, Optimization stopped because xtol_rel or xtol_abs was reached)
    <BLANKLINE>

)";
}

std::string nlopt_stopval_docstring()
{
    return R"(``stopval`` stopping criterion.

The ``stopval`` stopping criterion instructs the solver to stop when an objective value less than
or equal to ``stopval`` is found. Defaults to the C constant ``-HUGE_VAL`` (that is, this stopping criterion
is disabled by default).

Returns:
    ``float``: the value of the ``stopval`` stopping criterion

Raises:
    ValueError: if, when setting this property, a ``NaN`` is passed
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_ftol_rel_docstring()
{
    return R"(``ftol_rel`` stopping criterion.

The ``ftol_rel`` stopping criterion instructs the solver to stop when an optimization step (or an estimate of the
optimum) changes the objective function value by less than ``ftol_rel`` multiplied by the absolute value of the
function value. Defaults to 0 (that is, this stopping criterion is disabled by default).

Returns:
    ``float``: the value of the ``ftol_rel`` stopping criterion

Raises:
    ValueError: if, when setting this property, a ``NaN`` is passed
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_ftol_abs_docstring()
{
    return R"(``ftol_abs`` stopping criterion.

The ``ftol_abs`` stopping criterion instructs the solver to stop when an optimization step
(or an estimate of the optimum) changes the function value by less than ``ftol_abs``.
Defaults to 0 (that is, this stopping criterion is disabled by default).

Returns:
    ``float``: the value of the ``ftol_abs`` stopping criterion

Raises:
    ValueError: if, when setting this property, a ``NaN`` is passed
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_xtol_rel_docstring()
{
    return R"(``xtol_rel`` stopping criterion.

The ``xtol_rel`` stopping criterion instructs the solver to stop when an optimization step (or an estimate of the
optimum) changes every parameter by less than ``xtol_rel`` multiplied by the absolute value of the parameter.
Defaults to 1E-8.

Returns:
    ``float``: the value of the ``xtol_rel`` stopping criterion

Raises:
    ValueError: if, when setting this property, a ``NaN`` is passed
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_xtol_abs_docstring()
{
    return R"(``xtol_abs`` stopping criterion.

The ``xtol_abs`` stopping criterion instructs the solver to stop when an optimization step (or an estimate of the
optimum) changes every parameter by less than ``xtol_abs``. Defaults to 0 (that is, this stopping criterion is disabled
by default).

Returns:
    ``float``: the value of the ``xtol_abs`` stopping criterion

Raises:
    ValueError: if, when setting this property, a ``NaN`` is passed
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_maxeval_docstring()
{
    return R"(``maxeval`` stopping criterion.

The ``maxeval`` stopping criterion instructs the solver to stop when the number of function evaluations exceeds
``maxeval``. Defaults to 0 (that is, this stopping criterion is disabled by default).

Returns:
    ``int``: the value of the ``maxeval`` stopping criterion

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_maxtime_docstring()
{
    return R"(``maxtime`` stopping criterion.

The ``maxtime`` stopping criterion instructs the solver to stop when the optimization time (in seconds) exceeds
``maxtime``. Defaults to 0 (that is, this stopping criterion is disabled by default).

Returns:
    ``int``: the value of the ``maxtime`` stopping criterion

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string bls_selection_docstring(const std::string &algo)
{
    return R"(Individual selection policy.

This attribute represents the policy that is used in the ``evolve()`` method to select the individual
that will be optimised. The attribute can be either a string or an integral.

If the attribute is a string, it must be one of ``"best"``, ``"worst"`` and ``"random"``:

* ``"best"`` will select the best individual in the population,
* ``"worst"`` will select the worst individual in the population,
* ``"random"`` will randomly choose one individual in the population.

:func:`~pygmo.)"
           + algo + R"(.set_random_sr_seed()` can be used to seed the random number generator
used by the ``"random"`` policy.

If the attribute is an integer, it represents the index (in the population) of the individual that is selected
for optimisation.

Returns:
    ``int`` or ``str``: the individual selection policy or index

Raises:
    OverflowError: if the attribute is set to an integer which is negative or too large
    ValueError: if the attribute is set to an invalid string
    TypeError: if the attribute is set to a value of an invalid type
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string bls_replacement_docstring(const std::string &algo)
{
    return R"(Individual replacement policy.

This attribute represents the policy that is used in the ``evolve()`` method to select the individual
that will be replaced by the optimised individual. The attribute can be either a string or an integral.

If the attribute is a string, it must be one of ``"best"``, ``"worst"`` and ``"random"``:

* ``"best"`` will select the best individual in the population,
* ``"worst"`` will select the worst individual in the population,
* ``"random"`` will randomly choose one individual in the population.

:func:`~pygmo.)"
           + algo + R"(.set_random_sr_seed()` can be used to seed the random number generator
used by the ``"random"`` policy.

If the attribute is an integer, it represents the index (in the population) of the individual that will be
replaced by the optimised individual.

Returns:
    ``int`` or ``str``: the individual replacement policy or index

Raises:
    OverflowError: if the attribute is set to an integer which is negative or too large
    ValueError: if the attribute is set to an invalid string
    TypeError: if the attribute is set to a value of an invalid type
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string bls_set_random_sr_seed_docstring(const std::string &algo)
{
    return R"(set_random_sr_seed(seed)

Set the seed for the ``"random"`` selection/replacement policies.

Args:
    seed (``int``): the value that will be used to seed the random number generator used by the ``"random"``
      election/replacement policies (see :attr:`~pygmo.)"
           + algo + R"(.selection` and
      :attr:`~pygmo.)"
           + algo + R"(.replacement`)

Raises:
    OverflowError: if the attribute is set to an integer which is negative or too large
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_get_log_docstring()
{
    return R"(get_log()

Optimisation log.

The optimisation log is a collection of log data lines. A log data line is a tuple consisting of:

* the number of objective function evaluations made so far,
* the objective function value for the current decision vector,
* the number of constraints violated by the current decision vector,
* the constraints violation norm for the current decision vector,
* a boolean flag signalling the feasibility of the current decision vector.

Returns:
    ``list``: the optimisation log

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_get_last_opt_result_docstring()
{
    return R"(get_last_opt_result()

Get the result of the last optimisation.

Returns:
    ``int``: the NLopt return code for the last optimisation run, or ``NLOPT_SUCCESS`` if no optimisations have been run yet

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_get_solver_name_docstring()
{
    return R"(get_solver_name()

Get the name of the NLopt solver used during construction.

Returns:
    ``str``: the name of the NLopt solver used during construction

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string nlopt_local_optimizer_docstring()
{
    return R"(Local optimizer.

Some NLopt algorithms rely on other NLopt algorithms as local/subsidiary optimizers.
This property, of type :class:`~pygmo.nlopt`, allows to set such local optimizer.
By default, no local optimizer is specified, and the property is set to ``None``.

.. note::

   At the present time, only the ``"auglag"`` and ``"auglag_eq"`` solvers make use
   of a local optimizer. Setting a local optimizer on any other solver will have no effect.

.. note::

   The objective function, bounds, and nonlinear-constraint parameters of the local
   optimizer are ignored (as they are provided by the parent optimizer). Conversely, the stopping
   criteria should be specified in the local optimizer.The verbosity of
   the local optimizer is also forcibly set to zero during the optimisation.

Returns:
    :class:`~pygmo.nlopt`: the local optimizer, or ``None`` if not set

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python
      (e.g., type conversion errors, mismatched function signatures, etc.), when setting the property

)";
}

std::string sea_docstring()
{
    return R"(__init__(gen = 1, seed = random)

(N+1)-ES simple evolutionary algorithm.

Args:
    gen (``int``): number of generations to consider (each generation will compute the objective function once)
    seed (``int``): seed used by the internal random number generator

Raises:
    OverflowError: if *gen* or *seed* are negative or greater than an implementation-defined value
    unspecified: any exception thrown by failures at the intersection between C++ and Python
      (e.g., type conversion errors, mismatched function signatures, etc.)

See also the docs of the C++ class :cpp:class:`pagmo::sea`.

)";
}

std::string sea_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. 
The log frequency depends on the verbosity parameter (by default nothing is logged) which can be set calling
the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm` constructed with a
:class:`~pygmo.sea`. 
A verbosity larger than 1 will produce a log with one entry each verbosity fitness evaluations.
A verbosity equal to 1 will produce a log with one entry at each improvement of the fitness.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Best``, ``Improvement``, ``Mutations``

    * ``Gen`` (``int``), generation.
    * ``Fevals`` (``int``), number of functions evaluation made.
    * ``Best`` (``float``), the best fitness function found so far.
    * ``Improvement`` (``float``), improvement made by the last mutation.
    * ``Mutations`` (``float``), number of mutated components for the decision vector.

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(sea(500))
    >>> algo.set_verbosity(50)
    >>> prob = problem(schwefel(dim = 20))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best:   Improvement:     Mutations:
       1              1        6363.44        2890.49              2
    1001           1001        1039.92       -562.407              3
    2001           2001        358.966         -632.6              2
    3001           3001         106.08       -995.927              3
    4001           4001         83.391         -266.8              1
    5001           5001        62.4994       -1018.38              3
    6001           6001        39.2851       -732.695              2
    7001           7001        37.2185       -518.847              1
    8001           8001        20.9452        -450.75              1
    9001           9001        17.9193       -270.679              1
    >>> uda = algo.extract(sea)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 1, 6363.442036625835, 2890.4854414320716, 2), (1001, 1001, ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::sea::get_log()`.
)";
}

std::string ihs_docstring()
{
    return R"(__init__(gen = 1, phmcr = 0.85, ppar_min = 0.35, ppar_max=0.99, bw_min=1e-5, bw_max=1., seed = random)

Harmony search (HS) is a metaheuristic algorithm said to mimick the improvisation process of musicians.
In the metaphor, each musician (i.e., each variable) plays (i.e., generates) a note (i.e., a value)
for finding a best harmony (i.e., the global optimum) all together.

This pygmo UDA implements the so-called improved harmony search algorithm (IHS), in which the probability
of picking the variables from the decision vector and the amount of mutation to which they are subject
vary (respectively linearly and exponentially) at each call of the ``evolve()`` method.

In this algorithm the number of fitness function evaluations is equal to the number of iterations.
All the individuals in the input population participate in the evolution. A new individual is generated
at every iteration, substituting the current worst individual of the population if better.

.. warning::

   The HS algorithm can and has been  criticized, not for its performances,
   but for the use of a metaphor that does not add anything to existing ones. The HS
   algorithm essentially applies mutation and crossover operators to a background population and as such
   should have been developed in the context of Evolutionary Strategies or Genetic Algorithms and studied
   in that context. The use of the musicians metaphor only obscures its internal functioning
   making theoretical results from ES and GA erroneously seem as unapplicable to HS.

.. note::

   The original IHS algorithm was designed to solve unconstrained, deterministic single objective problems.
   In pygmo, the algorithm was modified to tackle also multi-objective, constrained (box and non linearly).
   Such extension is original with pygmo.

Args:
    gen (``int``): number of generations to consider (each generation will compute the objective function once)
    phmcr (``float``): probability of choosing from memory (similar to a crossover probability)
    ppar_min (``float``): minimum pitch adjustment rate. (similar to a mutation rate)
    ppar_max (``float``): maximum pitch adjustment rate. (similar to a mutation rate)
    bw_min (``float``): minimum distance bandwidth. (similar to a mutation width)
    bw_max (``float``): maximum distance bandwidth. (similar to a mutation width)
    seed (``int``): seed used by the internal random number generator

Raises:
    OverflowError: if *gen* or *seed* are negative or greater than an implementation-defined value
    ValueError: if *phmcr* is not in the ]0,1[ interval, *ppar_min* or *ppar_max* are not in the ]0,1[ 
        interval, min/max quantities are less than/greater than max/min quantities, *bw_min* is negative.
    unspecified: any exception thrown by failures at the intersection between C++ and Python
      (e.g., type conversion errors, mismatched function signatures, etc.)

See also the docs of the C++ class :cpp:class:`pagmo::ihs`.

)";
}

std::string ihs_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. 
The log frequency depends on the verbosity parameter (by default nothing is logged) which can be set calling
the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm` constructed with a
:class:`~pygmo.ihs`. 
A verbosity larger than 1 will produce a log with one entry each verbosity fitness evaluations.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Fevals``, ``ppar``, ``bw``, ``dx``, ``df``,  ``Violated``, ``Viol. Norm``,``ideal``

    * ``Fevals`` (``int``), number of functions evaluation made.
    * ``ppar`` (``float``), the pitch adjustment rate.
    * ``bw`` (``float``), the distance bandwidth.
    * ``dx`` (``float``), the population flatness evaluated as the distance between the decisions vector of the best and of the worst individual (or -1 in a multiobjective case).
    * ``df`` (``float``), the population flatness evaluated as the distance between the fitness of the best and of the worst individual (or -1 in a multiobjective case).
    * ``Violated`` (``int``), the number of constraints violated by the current decision vector.
    * ``Viol. Norm`` (``float``), the constraints violation norm for the current decision vector.
    * ``ideal_point`` (1D numpy array), the ideal point of the current population (cropped to max 5 dimensions only in the screen output)

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(ihs(20000))
    >>> algo.set_verbosity(2000)
    >>> prob = problem(hock_schittkowsky_71())
    >>> prob.c_tol = [1e-1]*2
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Fevals:          ppar:            bw:            dx:            df:      Violated:    Viol. Norm:        ideal1:
          1       0.350032       0.999425        4.88642        14.0397              0              0        43.2982
       2001       0.414032       0.316046        5.56101        25.7009              0              0        33.4251
       4001       0.478032      0.0999425          5.036        26.9657              0              0        19.0052
       6001       0.542032      0.0316046        3.77292        23.9992              0              0        19.0052
       8001       0.606032     0.00999425        3.97937        16.0803              0              0        18.1803
      10001       0.670032     0.00316046        1.15023        1.57947              0              0        17.8626
      12001       0.734032    0.000999425       0.017882      0.0185438              0              0        17.5894
      14001       0.798032    0.000316046     0.00531358      0.0074745              0              0        17.5795
      16001       0.862032    9.99425e-05     0.00270865     0.00155563              0              0        17.5766
      18001       0.926032    3.16046e-05     0.00186637     0.00167523              0              0        17.5748
    >>> uda = algo.extract(ihs)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 0.35003234534534, 0.9994245193792801, 4.886415773459253, 14.0397487316794, ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::ihs::get_log()`.
)";
}

std::string sga_docstring()
{
    return R"(__init__(gen = 1, cr = .90, eta_c = 1., m = 0.02, param_m = 1., param_s = 2, crossover = "exponential", mutation = "polynomial", selection = "tournament", seed = random)

A Simple Genetic Algorithm

.. versionadded:: 2.2

Approximately during the same decades as Evolutionary Strategies (see :class:`~pygmo.sea`) were studied, 
a different group led by John Holland, and later by his student David Goldberg, introduced and
studied an algorithmic framework called "genetic algorithms" that were, essentially, leveraging on
the same idea but introducing also crossover as a genetic operator. This led to a few decades of
confusion and discussions on what was an evolutionary startegy and what a genetic algorithm and on
whether the crossover was a useful operator or mutation only algorithms were to be preferred.

In pygmo we provide a rather classical implementation of a genetic algorithm, letting the user choose between
selected crossover types, selection schemes and mutation types.

The pseudo code of our version is:

.. code-block:: none

   > Start from a population (pop) of dimension N
   > while i < gen
   > > Selection: create a new population (pop2) with N individuals selected from pop (with repetition allowed)
   > > Crossover: create a new population (pop3) with N individuals obtained applying crossover to pop2
   > > Mutation:  create a new population (pop4) with N individuals obtained applying mutation to pop3
   > > Evaluate all new chromosomes in pop4
   > > Reinsertion: set pop to contain the best N individuals taken from pop and pop4

The various blocks of pygmo genetic algorithm are listed below:

*Selection*: two selection methods are provided: ``tournament`` and ``truncated``. ``Tournament`` selection works by
selecting each offspring as the one having the minimal fitness in a random group of size *param_s*. The ``truncated``
selection, instead, works selecting the best *param_s* chromosomes in the entire population over and over.
We have deliberately not implemented the popular roulette wheel selection as we are of the opinion that such
a system does not generalize much being highly sensitive to the fitness scaling.

*Crossover*: four different crossover schemes are provided:``single``, ``exponential``, ``binomial``, ``sbx``. The
``single`` point crossover, works selecting a random point in the parent chromosome and, with probability *cr*, inserting the
partner chromosome thereafter. The ``exponential`` crossover is taken from the algorithm differential evolution,
implemented, in pygmo, as :class:`~pygmo.de`. It essentially selects a random point in the parent chromosome and inserts,
in each successive gene, the partner values with probability  *cr* up to when it stops. The binomial crossover
inserts each gene from the partner with probability *cr*. The simulated binary crossover (called ``sbx``), is taken
from the NSGA-II algorithm, implemented in pygmo as :class:`~pygmo.nsga2`, and makes use of an additional parameter called
distribution index *eta_c*.

*Mutation*: three different mutations schemes are provided: ``uniform``, ``gaussian`` and ``polynomial``. Uniform mutation
simply randomly samples from the bounds. Gaussian muattion samples around each gene using a normal distribution
with standard deviation proportional to the *param_m* and the bounds width. The last scheme is the ``polynomial``
mutation from Deb.

*Reinsertion*: the only reinsertion strategy provided is what we call pure elitism. After each generation
all parents and children are put in the same pool and only the best are passed to the next generation.
 
.. note:

   This algorithm will work only for box bounded problems.

Args:
    gen (``int``): number of generations.
    cr (``float``): crossover probability.
    eta_c (``float``): distribution index for ``sbx`` crossover. This parameter is inactive if other types of crossover are selected.
    m (``float``): mutation probability.
    param_m (``float``): distribution index (``polynomial`` mutation), gaussian width (``gaussian`` mutation) or inactive (``uniform`` mutation)
    param_s (``float``): the number of best individuals to use in "truncated" selection or the size of the tournament in ``tournament`` selection.
    crossover (``str``): the crossover strategy. One of ``exponential``, ``binomial``, ``single`` or ``sbx``
    mutation (``str``): the mutation strategy. One of ``gaussian``, ``polynomial`` or ``uniform``.
    selection (``str``): the selection strategy. One of ``tournament``, "truncated".
    seed (``int``): seed used by the internal random number generator

Raises:
    OverflowError: if *gen* or *seed* are negative or greater than an implementation-defined value
    ValueError: if *cr* is not in [0,1], if *eta_c* is not in [1,100], if *m* is not in [0,1], input_f *mutation* 
      is not one of ``gaussian``, ``uniform`` or ``polynomial``, if *selection* not one of "roulette", 
      "truncated" or *crossover* is not one of ``exponential``, ``binomial``, ``sbx``, ``single``, if *param_m* is
      not in [0,1] and *mutation* is not ``polynomial``, if *mutation* is not in [1,100] and *mutation* is ``polynomial``.
    unspecified: any exception thrown by failures at the intersection between C++ and Python
      (e.g., type conversion errors, mismatched function signatures, etc.)

See also the docs of the C++ class :cpp:class:`pagmo::sga`.
)";
}

std::string sga_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. 
The log frequency depends on the verbosity parameter (by default nothing is logged) which can be set calling
the method :func:`~pygmo.algorithm.set_verbosity()` on an :class:`~pygmo.algorithm` constructed with a
:class:`~pygmo.sga`. 
A verbosity larger than 1 will produce a log with one entry each verbosity fitness evaluations.
A verbosity equal to 1 will produce a log with one entry at each improvement of the fitness.

Returns:
    ``list`` of ``tuples``: at each logged epoch, the values ``Gen``, ``Fevals``, ``Best``, ``Improvement``

    ``Gen`` (``int``), generation.
    ``Fevals`` (``int``), number of functions evaluation made.
    ``Best`` (``float``), the best fitness function found so far.
    ``Improvement`` (``float``), improvement made by the last generation.

Examples:
    >>> from pygmo import *
    >>> algo = algorithm(sga(gen = 500))
    >>> algo.set_verbosity(50)
    >>> prob = problem(schwefel(dim = 20))
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Gen:        Fevals:          Best:   Improvement:     Mutations:
       1              1        6363.44        2890.49              2
    1001           1001        1039.92       -562.407              3
    2001           2001        358.966         -632.6              2
    3001           3001         106.08       -995.927              3
    4001           4001         83.391         -266.8              1
    5001           5001        62.4994       -1018.38              3
    6001           6001        39.2851       -732.695              2
    7001           7001        37.2185       -518.847              1
    8001           8001        20.9452        -450.75              1
    9001           9001        17.9193       -270.679              1
    >>> uda = algo.extract(sea)
    >>> uda.get_log() # doctest: +SKIP
    [(1, 1, 6363.442036625835, 2890.4854414320716, 2), (1001, 1001, ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::sga::get_log()`.
)";
}

std::string ipopt_docstring()
{
    return R"(__init__()

Ipopt.

.. versionadded:: 2.2

This class is a user-defined algorithm (UDA) that wraps the Ipopt (Interior Point OPTimizer) solver,
a software package for large-scale nonlinear optimization. Ipopt is a powerful solver that
is able to handle robustly and efficiently constrained nonlinear opimization problems at high dimensionalities.

Ipopt supports only single-objective minimisation, and it requires the availability of the gradient in the
optimisation problem. If possible, for best results the Hessians should be provided as well (but Ipopt
can estimate numerically the Hessians if needed).

In order to support pygmo's population-based optimisation model, the ``evolve()`` method will select
a single individual from the input :class:`~pygmo.population` to be optimised.
If the optimisation produces a better individual (as established by :func:`~pygmo.compare_fc()`),
the optimised individual will be inserted back into the population. The selection and replacement strategies
can be configured via the :attr:`~pygmo.ipopt.selection` and :attr:`~pygmo.ipopt.replacement` attributes.

Ipopt supports a large amount of options for the configuration of the optimisation run. The options
are divided into three categories:

* *string* options (i.e., the type of the option is ``str``),
* *integer* options (i.e., the type of the option is ``int``),
* *numeric* options (i.e., the type of the option is ``float``).

The full list of options is available on the `Ipopt website <https://www.coin-or.org/Ipopt/documentation/node40.html>`__.
:class:`pygmo.ipopt` allows to configure any Ipopt option via methods such as :func:`~pygmo.ipopt.set_string_options()`,
:func:`~pygmo.ipopt.set_string_option()`, :func:`~pygmo.ipopt.set_integer_options()`, etc., which need to be used before
invoking the ``evolve()`` method.

If the user does not set any option, :class:`pygmo.ipopt` use Ipopt's default values for the options (see the
`documentation <https://www.coin-or.org/Ipopt/documentation/node40.html>`__), with the following
modifications:

* if the ``"print_level"`` integer option is **not** set by the user, it will be set to 0 by :class:`pygmo.ipopt` (this will
  suppress most screen output produced by the solver - note that we support an alternative form of logging via
  the :func:`pygmo.algorithm.set_verbosity()` machinery);
* if the ``"hessian_approximation"`` string option is **not** set by the user and the optimisation problem does
  **not** provide the Hessians, then the option will be set to ``"limited-memory"`` by :class:`pygmo.ipopt`. This makes it
  possible to optimise problems without Hessians out-of-the-box (i.e., Ipopt will approximate numerically the
  Hessians for you);
* if the ``"constr_viol_tol"`` numeric option is **not** set by the user and the optimisation problem is constrained,
  then :class:`pygmo.ipopt` will compute the minimum value ``min_tol`` in the vector returned by :attr:`pygmo.problem.c_tol`
  for the optimisation problem at hand. If ``min_tol`` is nonzero, then the ``"constr_viol_tol"`` Ipopt option will
  be set to ``min_tol``, otherwise the default Ipopt value (1E-4) will be used for the option. This ensures that,
  if the constraint tolerance is not explicitly set by the user, a solution deemed feasible by Ipopt is also
  deemed feasible by pygmo (but the opposite is not necessarily true).

.. note::

   This user-defined algorithm is available only if pygmo was compiled with the ``PAGMO_WITH_IPOPT`` option
   enabled (see the :ref:`installation instructions <install>`).

.. note::

   Ipopt is not thread-safe, and thus it cannot be used in a :class:`pygmo.thread_island`.

.. seealso::

   https://projects.coin-or.org/Ipopt.

See also the docs of the C++ class :cpp:class:`pagmo::ipopt`.

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_numeric_option("tol",1E-9) # Change the relative convergence tolerance
    >>> ip.get_numeric_options() # doctest: +SKIP
    {'tol': 1e-09}
    >>> algo = algorithm(ip)
    >>> algo.set_verbosity(1)
    >>> prob = problem(luksan_vlcek1(20))
    >>> prob.c_tol = [1E-6] * 18 # Set constraints tolerance to 1E-6
    >>> pop = population(prob, 20)
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    <BLANKLINE>
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
    Ipopt is released as open source code under the Eclipse Public License (EPL).
            For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    <BLANKLINE>
    <BLANKLINE>
    objevals:        objval:      violated:    viol. norm:
            1         201174             18         1075.3 i
            2         209320             18        691.814 i
            3        36222.3             18        341.639 i
            4        11158.1             18        121.097 i
            5        4270.38             18        46.4742 i
            6        2054.03             18        20.7306 i
            7        705.959             18        5.43118 i
            8        37.8304             18        1.52099 i
            9        2.89066             12       0.128862 i
           10       0.300807              3      0.0165902 i
           11     0.00430279              3    0.000496496 i
           12    7.54121e-06              2    9.70735e-06 i
           13    4.34249e-08              0              0
           14    3.71925e-10              0              0
           15    3.54406e-13              0              0
           16    2.37071e-18              0              0
    <BLANKLINE>
    Optimisation return status: Solve_Succeeded (value = 0)
    <BLANKLINE>
)";
}

std::string ipopt_get_log_docstring()
{
    return R"(get_log()

Optimisation log.

The optimisation log is a collection of log data lines. A log data line is a tuple consisting of:

* the number of objective function evaluations made so far,
* the objective function value for the current decision vector,
* the number of constraints violated by the current decision vector,
* the constraints violation norm for the current decision vector,
* a boolean flag signalling the feasibility of the current decision vector.

Returns:
    ``list``: the optimisation log

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

.. warning::

   The number of constraints violated, the constraints violation norm and the feasibility flag stored in the log
   are all determined via the facilities and the tolerances specified within :class:`pygmo.problem`. That
   is, they might not necessarily be consistent with Ipopt's notion of feasibility. See the explanation
   of how the ``"constr_viol_tol"`` numeric option is handled in :class:`pygmo.ipopt`.

.. note::

   Ipopt supports its own logging format and protocol, including the ability to print to screen and write to file.
   Ipopt's screen logging is disabled by default (i.e., the Ipopt verbosity setting is set to 0 - see
   :class:`pygmo.ipopt`). On-screen logging can be enabled via the ``"print_level"`` string option.

)";
}

std::string ipopt_get_last_opt_result_docstring()
{
    return R"(get_last_opt_result()

Get the result of the last optimisation.

Returns:
    ``int``: the Ipopt return code for the last optimisation run, or ``Ipopt::Solve_Succeeded`` if no optimisations have been run yet

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.get_last_opt_result()
    0

)";
}

std::string ipopt_set_string_option_docstring()
{
    return R"(set_string_option(name, value)

Set string option.

This method will set the optimisation string option *name* to *value*.
The optimisation options are passed to the Ipopt API when calling the ``evolve()`` method.

Args:
    name (``str``): the name of the option
    value (``str``): the value of the option

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_string_option("hessian_approximation","limited-memory")
    >>> algorithm(ip) # doctest: +NORMALIZE_WHITESPACE
    Algorithm name: Ipopt: Interior Point Optimization [deterministic]
        Thread safety: none
    <BLANKLINE>
    Extra info:
        Last optimisation return code: Solve_Succeeded (value = 0)
        Verbosity: 0
        Individual selection policy: best
        Individual replacement policy: best
        String options: {hessian_approximation : limited-memory}
    <BLANKLINE>
)";
}

std::string ipopt_set_string_options_docstring()
{
    return R"(set_string_options(opts)

Set string options.

This method will set the optimisation string options contained in *opts*.
It is equivalent to calling :func:`~pygmo.ipopt.set_string_option()` passing all the name-value pairs in *opts*
as arguments.

Args:
    opts (``dict`` of ``str``-``str`` pairs): the name-value map that will be used to set the options

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_string_options({"hessian_approximation":"limited-memory", "limited_memory_initialization":"scalar1"})
    >>> algorithm(ip) # doctest: +NORMALIZE_WHITESPACE
    Algorithm name: Ipopt: Interior Point Optimization [deterministic]
            Thread safety: none
    <BLANKLINE>
    Extra info:
            Last optimisation return code: Solve_Succeeded (value = 0)
            Verbosity: 0
            Individual selection policy: best
            Individual replacement policy: best
            String options: {hessian_approximation : limited-memory,  limited_memory_initialization : scalar1}

)";
}

std::string ipopt_get_string_options_docstring()
{
    return R"(get_string_options()

Get string options.

Returns:
    ``dict`` of ``str``-``str`` pairs: a name-value dictionary of optimisation string options

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_string_option("hessian_approximation","limited-memory")
    >>> ip.get_string_options()
    {'hessian_approximation': 'limited-memory'}

)";
}

std::string ipopt_reset_string_options_docstring()
{
    return R"(reset_string_options()

Clear all string options.

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_string_option("hessian_approximation","limited-memory")
    >>> ip.get_string_options()
    {'hessian_approximation': 'limited-memory'}
    >>> ip.reset_string_options()
    >>> ip.get_string_options()
    {}

)";
}

std::string ipopt_set_integer_option_docstring()
{
    return R"(set_integer_option(name, value)

Set integer option.

This method will set the optimisation integer option *name* to *value*.
The optimisation options are passed to the Ipopt API when calling the ``evolve()`` method.

Args:
    name (``str``): the name of the option
    value (``int``): the value of the option

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_integer_option("print_level",3)
    >>> algorithm(ip) # doctest: +NORMALIZE_WHITESPACE
    Algorithm name: Ipopt: Interior Point Optimization [deterministic]
            Thread safety: none
    <BLANKLINE>
    Extra info:
            Last optimisation return code: Solve_Succeeded (value = 0)
            Verbosity: 0
            Individual selection policy: best
            Individual replacement policy: best
            Integer options: {print_level : 3}

)";
}

std::string ipopt_set_integer_options_docstring()
{
    return R"(set_integer_options(opts)

Set integer options.

This method will set the optimisation integer options contained in *opts*.
It is equivalent to calling :func:`~pygmo.ipopt.set_integer_option()` passing all the name-value pairs in *opts*
as arguments.

Args:
    opts (``dict`` of ``str``-``int`` pairs): the name-value map that will be used to set the options

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_integer_options({"filter_reset_trigger":4, "print_level":3})
    >>> algorithm(ip) # doctest: +NORMALIZE_WHITESPACE
    Algorithm name: Ipopt: Interior Point Optimization [deterministic]
            Thread safety: none
    <BLANKLINE>
    Extra info:
            Last optimisation return code: Solve_Succeeded (value = 0)
            Verbosity: 0
            Individual selection policy: best
            Individual replacement policy: best
            Integer options: {filter_reset_trigger : 4,  print_level : 3}

)";
}

std::string ipopt_get_integer_options_docstring()
{
    return R"(get_integer_options()

Get integer options.

Returns:
    ``dict`` of ``str``-``int`` pairs: a name-value dictionary of optimisation integer options

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_integer_option("print_level",3)
    >>> ip.get_integer_options()
    {'print_level': 3}

)";
}

std::string ipopt_reset_integer_options_docstring()
{
    return R"(reset_integer_options()

Clear all integer options.

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_integer_option("print_level",3)
    >>> ip.get_integer_options()
    {'print_level': 3}
    >>> ip.reset_integer_options()
    >>> ip.get_integer_options()
    {}

)";
}

std::string ipopt_set_numeric_option_docstring()
{
    return R"(set_numeric_option(name, value)

Set numeric option.

This method will set the optimisation numeric option *name* to *value*.
The optimisation options are passed to the Ipopt API when calling the ``evolve()`` method.

Args:
    name (``str``): the name of the option
    value (``float``): the value of the option

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_numeric_option("tol",1E-6)
    >>> algorithm(ip) # doctest: +SKIP
    Algorithm name: Ipopt: Interior Point Optimization [deterministic]
            Thread safety: none
    <BLANKLINE>
    Extra info:
            Last optimisation return code: Solve_Succeeded (value = 0)
            Verbosity: 0
            Individual selection policy: best
            Individual replacement policy: best
            Numeric options: {tol : 1E-6}

)";
}

std::string ipopt_set_numeric_options_docstring()
{
    return R"(set_numeric_options(opts)

Set numeric options.

This method will set the optimisation numeric options contained in *opts*.
It is equivalent to calling :func:`~pygmo.ipopt.set_numeric_option()` passing all the name-value pairs in *opts*
as arguments.

Args:
    opts (``dict`` of ``str``-``float`` pairs): the name-value map that will be used to set the options

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_numeric_options({"tol":1E-4, "constr_viol_tol":1E-3})
    >>> algorithm(ip) # doctest: +SKIP
    Algorithm name: Ipopt: Interior Point Optimization [deterministic]
            Thread safety: none
    <BLANKLINE>
    Extra info:
            Last optimisation return code: Solve_Succeeded (value = 0)
            Verbosity: 0
            Individual selection policy: best
            Individual replacement policy: best
            Numeric options: {constr_viol_tol : 1E-3,  tol : 1E-4}

)";
}

std::string ipopt_get_numeric_options_docstring()
{
    return R"(get_numeric_options()

Get numeric options.

Returns:
    ``dict`` of ``str``-``float`` pairs: a name-value dictionary of optimisation numeric options

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_numeric_option("tol",1E-4)
    >>> ip.get_numeric_options() # doctest: +SKIP
    {'tol': 1E-4}

)";
}

std::string ipopt_reset_numeric_options_docstring()
{
    return R"(reset_numeric_options()

Clear all numeric options.

Examples:
    >>> from pygmo import *
    >>> ip = ipopt()
    >>> ip.set_numeric_option("tol",1E-4)
    >>> ip.get_numeric_options() # doctest: +SKIP
    {'tol': 1E-4}
    >>> ip.reset_numeric_options()
    >>> ip.get_numeric_options()
    {}
)";
}

} // namespace pygmo
