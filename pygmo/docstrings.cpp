/* Copyright 2017 PaGMO development team

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

#include "docstrings.hpp"

namespace pygmo
{

std::string population_docstring()
{
    return R"(The population class.

This class represents a population of individuals, i.e., potential candidate solutions to a given problem. In pygmo an
individual is determined:

* by a unique ID used to track him across generations and migrations,
* by a chromosome (a decision vector),
* by the fitness of the chromosome as evaluated by a :class:`~pygmo.core.problem` and thus including objectives,
  equality constraints and inequality constraints if present.

A special mechanism is implemented to track the best individual that has ever been part of the population. Such an individual
is called *champion* and its decision vector and fitness vector are automatically kept updated. The *champion* is not necessarily
an individual currently in the population. The *champion* is only defined and accessible via the population interface if the
:class:`~pygmo.core.problem` currently contained in the :class:`~pygmo.core.population` is single objective.

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
    unspecified: any exception thrown by :func:`pygmo.core.problem.fitness()` or by failures at the intersection between C++ and
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
    unspecified: any exception thrown by :func:`pygmo.core.problem.fitness()` or by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string population_best_idx_docstring()
{
    return R"(best_idx(tol = 0.)

Index of the best individual.

If the problem is single-objective and unconstrained, the best is simply the individual with the smallest fitness. If the problem
is, instead, single objective, but with constraints, the best will be defined using the criteria specified in :cpp:func:`pagmo::sort_population_con()`.
If the problem is multi-objective one single best is not well defined. In this case the user can still obtain a strict ordering of the population
individuals by calling the :cpp:func:`pagmo::sort_population_mo()` function.

Args:
    tol (``float`` or array-like object): scalar tolerance or vector of tolerances to be applied to each constraints

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
is, instead, single objective, but with constraints, the worst will be defined using the criteria specified in :cpp:func:`pagmo::sort_population_con()`.
If the problem is multi-objective one single worst is not well defined. In this case the user can still obtain a strict ordering of the population
individuals by calling the :cpp:func:`pagmo::sort_population_mo()` function.

Args:
    tol (``float`` or array-like object): scalar tolerance or vector of tolerances to be applied to each constraints

Returns:
    ``int``: the index of the worst individual

Raises:
     ValueError: if the problem is multiobjective and thus a worst individual is not well defined, or if the population is empty
     unspecified: any exception thrown by :cpp:func:`pagmo::sort_population_con()`

)";
}

std::string population_champion_x_docstring()
{
    return R"(Champion's decision vector.

This read-only property contains an array of ``float`` representing the decision vector of the population's champion.

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

**NOTE**: the user must make sure that the input fitness *f* makes sense as pygmo will only check its dimension.

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

**NOTE**: a call to this method triggers one fitness function evaluation.

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

This property gives direct access to the :class:`~pygmo.core.problem` stored within the population.

Returns:
    :class:`~pygmo.core.problem`: a reference to the internal problem

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.) when setting the property

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
    return R"(__init__(udp = null_problem)

Problem class.

This class represents a generic *mathematical programming* or *evolutionary optimization* problem in the form:

.. math::
   \begin{array}{rl}
   \mbox{find:}      & \mathbf {lb} \le \mathbf x \le \mathbf{ub}\\
   \mbox{to minimize: } & \mathbf f(\mathbf x, s) \in \mathbb R^{n_{obj}}\\
   \mbox{subject to:} & \mathbf {c}_e(\mathbf x, s) = 0 \\
                     & \mathbf {c}_i(\mathbf x, s) \le 0
   \end{array}

where :math:`\mathbf x \in \mathbb R^{n_x}` is called *decision vector* or
*chromosome*, :math:`\mathbf{lb}, \mathbf{ub} \in \mathbb R^{n_x}` are the *box-bounds*,
:math:`\mathbf f: \mathbb R^{n_x} \rightarrow \mathbb R^{n_{obj}}` define the *objectives*,
:math:`\mathbf c_e:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ec}}` are non linear *equality constraints*,
and :math:`\mathbf c_i:  \mathbb R^{n_x} \rightarrow \mathbb R^{n_{ic}}` are non linear *inequality constraints*.
Note that the objectives and constraints may also depend from an added value :math:`s` seeding the
values of any number of stochastic variables. This allows also for stochastic programming
tasks to be represented by this class.

In order to define an optimizaztion problem in pygmo, the user must first define a class
whose methods describe the properties of the problem and allow to compute
the objective function, the gradient, the constraints, etc. In pygmo, we refer to such
a class as a **user-defined problem**, or UDP for short. Once defined and instantiated,
a UDP can then be used to construct an instance of this class, :class:`~pygmo.core.problem`, which
provides a generic interface to optimization problems.

Every UDP must implement at least the following two methods:

.. code-block:: python

   def fitness(self, dv):
     ...
   def get_bounds(self):
     ...

The ``fitness()`` method is expected to return the fitness of the input decision vector, while
``get_bounds()`` is expected to return the box bounds of the problem,
:math:`(\mathbf{lb}, \mathbf{ub})`, which also implicitly define the dimension of the problem.
The ``fitness()`` and ``get_bounds()`` methods of the UDP are accessible from the corresponding
:func:`pygmo.core.problem.fitness()` and :func:`pygmo.core.problem.get_bounds()`
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
methods in the UDP should be implemented and on how they are used by :class:`~pygmo.core.problem`.
Note that the exposed C++ problems can also be used as UDPs, even if they do not expose any of the
mandatory or optional methods listed above (see :ref:`here <py_problems>` for the
full list of UDPs already coded in pygmo).

This class is the Python counterpart of the C++ class :cpp:class:`pagmo::problem`.

Args:
    udp: a user-defined problem (either C++ or Python - note that *udp* will be deep-copied
      and stored inside the :class:`~pygmo.core.problem` instance)

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

In addition to invoking the ``fitness()`` method of the UDP, this method will perform sanity checks on
*dv* and on the returned fitness vector. A successful call of this method will increase the internal fitness
evaluation counter (see :func:`~pygmo.core.problem.get_fevals()`).

The ``fitness()`` method of the UDP must be able to take as input the decision vector as a 1D NumPy array, and it must
return the fitness vector as an iterable Python object (e.g., 1D NumPy array, list, tuple, etc.).

Args:
    dv (array-like object): the decision vector (chromosome) to be evaluated

Returns:
    1D NumPy float array: the fitness of *dv*

Raises:
    ValueError: if either the length of *dv* differs from the value returned by :func:`~pygmo.core.problem.get_nx()`, or
      the length of the returned fitness vector differs from the value returned by :func:`~pygmo.core.problem.get_nf()`
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
of a :class:`~pygmo.core.problem`.

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
of a :class:`~pygmo.core.problem`.

Returns:
    ``int``: the number of objectives of the problem

)";
}

std::string problem_get_nx_docstring()
{
    return R"(get_nx()

Dimension of the problem.

This method will return :math:`n_{x}`, the dimension of the problem as established by the length of
the bounds returned by :func:`~pygmo.core.problem.get_bounds()`.

Returns:
    ``int``: the dimension of the problem

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
of a :class:`~pygmo.core.problem`.

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
of a :class:`~pygmo.core.problem`.

Returns:
    ``int``: the number of inequality constraints of the problem

)";
}

std::string problem_get_nc_docstring()
{
    return R"(get_nc()

Total number of constraints.

This method will return the sum of the output of :func:`~pygmo.core.problem.get_nic()` and
:func:`~pygmo.core.problem.get_nec()` (i.e., the total number of constraints).

Returns:
    ``int``: the total number of constraints of the problem

)";
}

std::string problem_c_tol_docstring()
{
    return R"(Constraints tolerance.

This property contains an array of ``float`` that are used when checking for constraint feasibility.
The dimension of the array is :math:`n_{ec} + n_{ic}`, and the array is zero-filled on problem
construction.

Returns:
    1D NumPy float array: the constraints tolerance

Raises:
    ValueError: if, when setting this property, the size of the input array differs from the number
      of constraints of the problem or if any element of the array is negative or NaN
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string problem_get_fevals_docstring()
{
    return R"(Number of fitness evaluations.

Each time a call to :func:`~pygmo.core.problem.fitness()` successfully completes, an internal counter
is increased by one. The counter is initialised to zero upon problem construction and it is never
reset. Copy operations copy the counter as well.

Returns:
    ``int`` : the number of times :func:`~pygmo.core.problem.fitness()` was successfully called

)";
}

std::string problem_get_gevals_docstring()
{
    return R"(Number of gradient evaluations.

Each time a call to :func:`~pygmo.core.problem.gradient()` successfully completes, an internal counter
is increased by one. The counter is initialised to zero upon problem construction and it is never
reset. Copy operations copy the counter as well.

Returns:
    ``int`` : the number of times :func:`~pygmo.core.problem.gradient()` was successfully called

)";
}

std::string problem_get_hevals_docstring()
{
    return R"(Number of hessians evaluations.

Each time a call to :func:`~pygmo.core.problem.hessians()` successfully completes, an internal counter
is increased by one. The counter is initialised to zero upon problem construction and it is never
reset. Copy operations copy the counter as well.

Returns:
    ``int`` : the number of times :func:`~pygmo.core.problem.hessians()` was successfully called

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
implement the ``gradient()`` method of the UDP, see :func:`~pygmo.core.problem.gradient()`.

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
:func:`~pygmo.core.problem.gradient_sparsity()`.

If the UDP provides a ``gradient()`` method, this method will forward *dv* to the ``gradient()``
method of the UDP after sanity checks. The output of the ``gradient()`` method of the UDP will
also be checked before being returned. If the UDP does not provide a ``gradient()`` method, an
error will be raised. A successful call of this method will increase the internal gradient
evaluation counter (see :func:`~pygmo.core.problem.get_gevals()`).

The ``gradient()`` method of the UDP must be able to take as input the decision vector as a 1D NumPy
array, and it must return the gradient vector as an iterable Python object (e.g., 1D NumPy array,
list, tuple, etc.).

Args:
    dv (array-like object): the decision vector whose gradient will be computed

Returns:
    1D NumPy float array: the gradient of *dv*

Raises:
    ValueError: if either the length of *dv* differs from the value returned by :func:`~pygmo.core.problem.get_nx()`, or
      the returned gradient vector does not have the same size as the vector returned by
      :func:`~pygmo.core.problem.gradient_sparsity()`
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
implement the ``gradient_sparsity()`` method of the UDP, see :func:`~pygmo.core.problem.gradient_sparsity()`.

**NOTE** regardless of what this method returns, the :func:`~pygmo.core.problem.gradient_sparsity()` method will always
return a sparsity pattern: if the UDP does not provide the gradient sparsity, pygmo will assume that the sparsity
pattern of the gradient is dense. See :func:`~pygmo.core.problem.gradient_sparsity()` for more details.

Returns:
    ``bool``: a flag signalling the availability of the gradient sparsity in the UDP

)";
}

std::string problem_gradient_sparsity_docstring()
{
    return R"(gradient_sparsity()

Gradient sparsity pattern.

This method will return the gradient sparsity pattern of the problem. The gradient sparsity pattern is a
collection of the indices :math:`(i,j)` of the non-zero elements of :math:`g_{ij} = \frac{\partial f_i}{\partial x_j}`.

If :func:`~pygmo.core.problem.has_gradient_sparsity()` returns ``True``, then the ``gradient_sparsity()`` method of the
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
      * if the sparsity pattern returned by the UDP is invalid (specifically, if it contains duplicate pairs of indices
        or if the indices in the pattern are incompatible with the properties of the problem)
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
implement the ``hessians()`` method of the UDP, see :func:`~pygmo.core.problem.hessians()`.

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
as returned by :func:`~pygmo.core.problem.hessians_sparsity()`. Since
the hessians are symmetric, their sparse representation contains only lower triangular elements.

If the UDP provides a ``hessians()`` method, this method will forward *dv* to the ``hessians()``
method of the UDP after sanity checks. The output of the ``hessians()`` method of the UDP will
also be checked before being returned. If the UDP does not provide a ``hessians()`` method, an
error will be raised. A successful call of this method will increase the internal hessians
evaluation counter (see :func:`~pygmo.core.problem.get_hevals()`).

The ``hessians()`` method of the UDP must be able to take as input the decision vector as a 1D NumPy
array, and it must return the hessians vector as an iterable Python object (e.g., list, tuple, etc.).

Args:
    dv (array-like object): the decision vector whose hessians will be computed

Returns:
    ``list`` of 1D NumPy float array: the hessians of *dv*

Raises:
    ValueError: if either the length of *dv* differs from the value returned by :func:`~pygmo.core.problem.get_nx()`, or
      the length of returned hessians does not match the corresponding hessians sparsity pattern dimensions
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
implement the ``hessians_sparsity()`` method of the UDP, see :func:`~pygmo.core.problem.hessians_sparsity()`.

**NOTE** regardless of what this method returns, the :func:`~pygmo.core.problem.hessians_sparsity()` method will always
return a sparsity pattern: if the UDP does not provide the hessians sparsity, pygmo will assume that the sparsity
pattern of the hessians is dense. See :func:`~pygmo.core.problem.hessians_sparsity()` for more details.

Returns:
    ``bool``: a flag signalling the availability of the hessians sparsity in the UDP

)";
}

std::string problem_hessians_sparsity_docstring()
{
    return R"(hessians_sparsity()

Hessians sparsity pattern.

This method will return the hessians sparsity pattern of the problem. Each component :math:`l` of the hessians
sparsity pattern is a collection of the indices :math:`(i,j)` of the non-zero elements of
:math:`h^l_{ij} = \frac{\partial f^l}{\partial x_i\partial x_j}`. Since the Hessian matrix is symmetric, only
lower triangular elements are allowed.

If :func:`~pygmo.core.problem.has_hessians_sparsity()` returns ``True``, then the ``hessians_sparsity()`` method of the
UDP will be invoked, and its result returned (after sanity checks). Otherwise, a dense pattern is assumed and
:math:`n_f` sparsity patterns containing :math:`((0,0),(1,0), (1,1), (2,0) ... (n_x-1,n_x-1))` will be returned.

The ``hessians_sparsity()`` method of the UDP must return an iterable Python object of any kind. Each element of the
returned object will then be interpreted as a sparsity pattern in the same way as described in
:func:`~pygmo.core.problem.gradient_sparsity()`. Specifically:

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
      * if a sparsity pattern returned by the UDP is invalid (specifically, if it contains duplicate pairs of indices
        or if the indices in the pattern are incompatible with the properties of the problem)
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
implement the ``set_seed()`` method of the UDP, see :func:`~pygmo.core.problem.set_seed()`.

Returns:
    ``bool``: a flag signalling the availability of the ``set_seed()`` method in the UDP

)";
}

std::string problem_feasibility_f_docstring()
{
    return R"(feasibility_f(f)

This method will check the feasibility of a fitness vector *f* against the tolerances returned by
:attr:`~pygmo.core.problem.c_tol`.

Args:
    f (array-like object): a fitness vector

Returns:
    ``bool``: ``True`` if the fitness vector is feasible, ``False`` otherwise

Raises:
    ValueError: if the size of *f* is not the same as the output of
      :func:`~pymog.core.problem.get_nf()`

)";
}

std::string problem_feasibility_x_docstring()
{
    return R"(feasibility_x(x)

This method will check the feasibility of the fitness corresponding to a decision vector *x* against
the tolerances returned by :attr:`~pygmo.core.problem.c_tol`.

**NOTE** This will cause one fitness evaluation.

Args:
    dv (array-like object): a decision vector

Returns:
    ``bool``: ``True`` if *x* results in a feasible fitness, ``False`` otherwise

Raises:
     unspecified: any exception thrown by :func:`~pygmo.core.problem.feasibility_f()` or
       :func:`~pygmo.core.problem.fitness()`

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
    return R"(__init__(udp = null_problem(), translation = [0.])

The translate meta-problem.

This meta-problem translates the whole search space of an input user-defined problem (UDP) by a fixed
translation vector. :class:`~pygmo.core.translate` objects are user-defined problems that
can be used in the definition of a :class:`pygmo.core.problem`.

The constructor admits two forms,

* no arguments,
* exactly two arguments.

Any other combination of arguments will raise an error.

Args:
    udp: a user-defined problem (either C++ or Python - note that *udp* will be deep-copied
      and stored inside the :class:`~pygmo.core.translate` instance)
    translation (array-like object): an array containing the translation to be applied

Raises:
    ValueError: if the length of *translation* is not equal to the dimension of *udp*
    unspecified: any exception thrown by:

      * the constructor of :class:`pygmo.core.problem`,
      * the constructor of the underlying C++ class,
      * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
        signatures, etc.)

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
    return R"(__init__(uda = null_algorithm)

Algorithm class.

This class represents an optimization algorithm. An algorithm can be
stochastic, deterministic, population based, derivative-free, using hessians,
using gradients, a meta-heuristic, evolutionary, etc.. Via this class pygmo offers
a common interface to all types of algorithms that can be applied to find solution
to a generic matematical programming problem as represented by the
:class:`~pygmo.core.problem` class.

In order to define an optimizaztion algorithm in pygmo, the user must first define a class
whose methods describe the properties of the algorithm and implement its logic.
In pygmo, we refer to such a class as a **user-defined algorithm**, or UDA for short. Once
defined and instantiated, a UDA can then be used to construct an instance of this class,
:class:`~pygmo.core.algorithm`, which provides a generic interface to optimization algorithms.

Every UDA must implement at least the following method:

.. code-block:: python

   def evolve(self, pop):
     ...

The ``evolve()`` method takes as input a :class:`~pygmo.core.population`, and it is expected to return
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
methods in the UDA should be implemented and on how they are used by :class:`~pygmo.core.algorithm`.
Note that the exposed C++ algorithms can also be used as UDAs, even if they do not expose any of the
mandatory or optional methods listed above (see :ref:`here <py_algorithms>` for the
full list of UDAs already coded in pygmo).

This class is the Python counterpart of the C++ class :cpp:class:`pagmo::algorithm`.

Args:
    uda: a user-defined algorithm (either C++ or Python - note that *uda* will be deep-copied
      and stored inside the :class:`~pygmo.core.algorithm` instance)

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
    pop (:class:`~pygmo.core.population`): starting population

Returns:
    :class:`~pygmo.core.population`: evolved population

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
implement the ``set_seed()`` method of the UDA, see :func:`~pygmo.core.algorithm.set_seed()`.

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
implement the ``set_verbosity()`` method of the UDA, see :func:`~pygmo.core.algorithm.set_verbosity()`.

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

std::string mbh_docstring()
{
    return R"(__init__(uda = compass_search(), stop = 5, perturb = 1e-2, seed = random)

Monotonic Basin Hopping (generalized).

Monotonic basin hopping, or simply, basin hopping, is an algorithm rooted in the idea of mapping
the objective function :math:`f(\mathbf x_0)` into the local minima found starting from :math:`\mathbf x_0`.
This simple idea allows a substantial increase of efficiency in solving problems, such as the Lennard-Jones
cluster or the MGA-1DSM interplanetary trajectory problem that are conjectured to have a so-called
funnel structure.

In pygmo we provide an original generalization of this concept resulting in a meta-algorithm that operates
on any :class:`pygmo.core.population` using any suitable user-defined algorithm (UDA). When a population containing a single
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

:class:`pygmo.core.mbh` is a user-defined algorithm (UDA) that can be used to construct :class:`pygmo.core.algorithm` objects.

See: http://arxiv.org/pdf/cond-mat/9803344 for the paper introducing the basin hopping idea for a Lennard-Jones
cluster optimization.

See also the docs of the C++ class :cpp:class:`pagmo::mbh`.

The constructor admits two forms:

* no arguments,
* three mandatory arguments and one optional argument (the seed).

Any other combination of arguments will raise an exception.

Args:
    uda: a user-defined algorithm (either C++ or Python - note that *uda* will be deep-copied
      and stored inside the :class:`~pygmo.core.mbh` instance)
    stop (``int``): consecutive runs of the inner algorithm that need to result in no improvement for
      :class:`~pygmo.core.mbh` to stop
    perturb (``float`` or array-like object): perturb the perturbation to be applied to each component
    seed (``int``): seed used by the internal random number generator

Raises:
    ValueError: if *perturb* (or one of its components, if *perturb* is an array) is not in the
      (0,1] range
    unspecified: any exception thrown by the constructor of :class:`pygmo.core.algorithm`, or by
      failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
      signatures, etc.)

)";
}

std::string mbh_get_seed_docstring()
{
    return R"(get_seed()

Get the seed value that was used for the construction of this :class:`~pygmo.core.mbh`.

Returns:
    ``int``: the seed value

)";
}

std::string mbh_get_verbosity_docstring()
{
    return R"(get_verbosity()

Get the verbosity level value that was used for the construction of this :class:`~pygmo.core.mbh`.

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
(by default nothing is logged) which can be set calling :func:`~pygmo.core.algorithm.set_verbosity()` on a :class:`~pygmo.core.algorithm` constructed
with an :class:`~pygmo.core.mbh`. A verbosity level ``N > 0`` will log one line at the end of each call to the inner algorithm.

Returns:
    ``list`` of ``tuples``: at each call of the inner algorithm, the values ``Fevals``, ``Best``, ``Violated``, ``Viol. Norm`` and ``Trial``, where:

    * ``Fevals`` (``int``), the number of fitness evaluations made
    * ``Best`` (``float``), the objective function of the best fitness currently in the population
    * ``Violated`` (``int``), the number of constraints currently violated by the best solution
    * ``Viol. Norm`` (``float``), the norm of the violation (discounted already by the constraints tolerance)
    * ``Trial`` (``int``), the trial number (which will determine the algorithm stop)

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

std::string null_algorithm_docstring()
{
    return R"(__init__()

The null algorithm.

An algorithm used in the default-initialization of :class:`pygmo.core.algorithm` and of the meta-algorithms.

)";
}

std::string null_problem_docstring()
{
    return R"(__init__(nobj = 1, nec = 0, nic = 0)

The null problem.

A problem used in the default-initialization of :class:`pygmo.core.problem` and of the meta-problems.

Args:
    nobj (``int``): the number of objectives
    nec  (``int``): the number of equality constraints
    nic  (``int``): the number of inequality constraintsctives

Raises:
    ValueError: if *nobj*, *nec*, *nic* are not positive or if *nobj* is zero
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

std::string zdt_p_distance_docstring()
{
    return R"(p_distance(point)

p_distance(pop)

Convergence metric for decision vectors (0 = on the optimal front)

Introduced by Martens and Izzo, this metric is able to measure "a distance" of any point from
the pareto front of any DTLZ problem analytically.

Args:
    point (array-like object): decision vector for which the p distance is requested
    pop (:class:`~pygmo.core.population`): population for which the average p distance is requested

Returns:
    ``float``: the distance (or average distance) from the Pareto front

See also the docs of the C++ class :func:`~pygmo.core.zdt.p_distance()`

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
    pop (:class:`~pygmo.core.population`): population for which the average p distance is requested

Returns:
    ``float``: the distance (or average distance) from the Pareto front

See also the docs of the C++ class :func:`~pygmo.core.dtlz.p_distance()`

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
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.bee_colony`. A verbosity of ``N`` implies a log line each ``N`` generations.

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
    >>> pop = algo.evolve(pop)
    Gen:        Fevals:  Current best:          Best:
       1             40         183728         183728
     101           4040        506.757        26.4234
     201           8040        55.6282        14.9136
     301          12040         65.554        14.9136
     401          16040        191.654        14.9136
    >>> al = algo.extract(bee_colony)
    >>> al.get_log()
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
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.de`. A verbosity of ``N`` implies a log line each ``N`` generations.

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
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.compass_search`. A verbosity larger than 0 implies one log line at each improvment of the fitness or
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
    >>> pop = algo.evolve(pop)
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
    >>> al = algo.extract(compass_search)
    >>> al.get_log()
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
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.sade`. A verbosity of ``N`` implies a log line each ``N`` generations.

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

See also the docs of the relevant C++ method :cpp:func:`pagmo::sade::get_log()`.

)";
}

std::string nsga2_docstring()
{
    return R"(__init__(gen = 1, cr = 0.95, eta_c = 10, m = 0.01, eta_m = 10, int_dim = 0, seed = random)

Non dominated Sorting Genetic Algorithm (NSGA-II).

Args:
    gen (``int``): number of generations
    cr (``float``): crossover probability
    eta_c (``float``): distribution index for crossover
    m (``float``): mutation probability
    eta_m (``float``): distribution index for mutation
    int_dim (``int``): the dimension of the decision vector to be considered as integer (the last int_dim entries will be treated as integers when mutation and crossover are applied)
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
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.nsga2`. A verbosity of ``N`` implies a log line each ``N`` generations.

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
    >>> pop = algo.evolve(pop)
    Gen:        Fevals:        ideal1:        ideal2:
       1              0      0.0033062        2.44966
      21            800    0.000275601       0.893137
      41           1600    3.15834e-05        0.44117
      61           2400     2.3664e-05       0.206365
      81           3200     2.3664e-05       0.133305
    >>> al = algo.extract(nsga2)
    >>> al.get_log()
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
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.moead`. A verbosity of ``N`` implies a log line each ``N`` generations.

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

See also the docs of the relevant C++ method :cpp:func:`pagmo::moead::get_log()`.

)";
}

std::string cmaes_docstring()
{
    return R"(__init__(gen = 1, cc = -1, cs = -1, c1 = -1, cmu = -1, sigma0 = 0.5, ftol = 1e-6, xtol = 1e-6, memory = False, seed = random)

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
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()``. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.cmaes`. A verbosity of ``N`` implies a log line each ``N`` generations.

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

See also the docs of the relevant C++ method :cpp:func:`pagmo::cmaes::get_log()`.

)";
}

std::string de1220_docstring()
{
    return R"(__init__(gen = 1, allowed_variants = [2,3,7,10,13,14,15,16], variant_adptv = 1, ftol = 1e-6, xtol = 1e-6, memory = False, seed = random)

Self-adaptive Differential Evolution, pygmo flavour (pDE).
The adaptation of the mutation variant is added to :class:`~pygmo.core.sade`

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
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.de1220`. A verbosity of N implies a log line each N generations.

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

std::string pso_get_log_docstring()
{
    return R"(get_log()

Returns a log containing relevant parameters recorded during the last call to ``evolve()`` and printed to screen. The log frequency depends on the verbosity
parameter (by default nothing is logged) which can be set calling the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm`
constructed with a :class:`~pygmo.core.de1220`. A verbosity of ``N`` implies a log line each ``N`` generations.

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
    >>> pop = algo.evolve(pop)
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
    >>> al = algo.extract(de1220)
    >>> al.get_log()
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
the method :func:`~pygmo.core.algorithm.set_verbosity()` on an :class:`~pygmo.core.algorithm` constructed with a
:class:`~pygmo.core.simulated_annealing`. A verbosity larger than 0 will produce a log with one entry
each verbosity function evaluations.

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
    >>> pop = algo.evolve(pop)
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
    >>> uda.get_log()
    [(57, 5936.999957947842, 5936.999957947842, 0.47999999999999987, 10.0), (10033, ...

See also the docs of the relevant C++ method :cpp:func:`pagmo::simulated_annealing::get_log()`.

)";
}

std::string decompose_docstring()
{
    return R"(__init__(udp = null_problem(nobj = 2), weight = [0.5, 0.5], z = [0.,0.], method = 'weighted', adapt_ideal = False)

The decompose meta-problem.

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

**NOTE** The reference point :math:`z^*` is often taken as the ideal point and as such
it may be allowed to change during the course of the optimization / evolution. The argument adapt_ideal activates
this behaviour so that whenever a new ideal point is found :math:`z^*` is adapted accordingly.

**NOTE** The use of :class:`~pygmo.core.decompose` discards gradients and hessians so that if the original user defined problem
implements them, they will not be available in the decomposed problem. The reason for this behaviour is that
the Tchebycheff decomposition is not differentiable. Also, the use of this class was originally intended for
derivative-free optimization.

See: "Q. Zhang -- MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"

See: https://en.wikipedia.org/wiki/Multi-objective_optimization#Scalarizing_multi-objective_optimization_problems

The constructor admits two forms:

* no arguments,
* two mandatory arguments and three optional arguments.

Any other combination of arguments will raise an exception.

Args:
    udp: a user-defined problem (either C++ or Python - note that *udp* will be deep-copied
      and stored inside the :class:`~pygmo.core.decompose` instance)
    weight (array-like object): the vector of weights :math:`\boldsymbol \lambda`
    z (array-like object): the reference point :math:`\mathbf z^*`
    method (``str``): a string containing the decomposition method chosen
    adapt_ideal (``bool``): when ``True``, the reference point is adapted at each fitness evaluation
      to be the ideal point

Raises:
    ValueError: if either:

      * *udp* is single objective or constrained,
      * *method* is not one of [``'weighted'``, ``'tchebycheff'``, ``'bi'``],
      * *weight* is not of size :math:`n`,
      * *z* is not of size :math:`n`,
      * *weight* is not such that :math:`\lambda_i > 0, \forall i=1..n`,
      * *weight* is not such that :math:`\sum_i \lambda_i = 1`
    unspecified: any exception thrown by:

      * the constructor of :class:`pygmo.core.problem`,
      * the constructor of the underlying C++ class,
      * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
        signatures, etc.)

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

**NOTE** The reference point is adapted at each call of the fitness.

Returns:
    1D NumPy float array: the reference point

Raises:
    unspecified: any exception thrown by failures at the intersection between C++ and Python (e.g.,
      type conversion errors, mismatched function signatures, etc.)

)";
}

std::string unconstrain_docstring()
{
    return R"(__init__(udp = null_problem(nobj=2, nec=3, nic=4), method = "death penalty", weights = [])

The unconstrain meta-problem.

This meta-problem transforms a constrained problem into an unconstrained problem applying one of the following methods:

* Death penalty: simply penalizes all objectives by the same high value if the fitness vector is infeasible.
* Kuri's death penalty: defined by Angel Kuri Morales et al., penalizes all objectives according to the rate of satisfied constraints.
* Weighted violations penalty: penalizes all objectives by the weighted sum of the constraint violations.
* Ignore the constraints: simply ignores the constraints.
* Ignore the objectives: ignores the objectives and defines as a new single objective the overall constraints violation (i.e. the sum of the L2 norms of the equalities and inequalities violations)

**NOTE** The use of :class:`~pygmo.core.unconstrain` discards gradients and hessians so that if the original user defined problem
implements them, they will not be available in the unconstrained problem. The reason for this behaviour is that,
in general, the methods implemented may not be differentiable. Also, the use of this class was originally intended for
derivative-free optimization.

See: Coello Coello, C. A. (2002). Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: 
a survey of the state of the art. Computer methods in applied mechanics and engineering, 191(11), 1245-1287.

See: Kuri Morales, A. and Quezada, C.C. A Universal eclectic genetic algorithm for constrained optimization,
Proceedings 6th European Congress on Intelligent Techniques & Soft Computing, EUFIT'98, 518-522, 1998.

The constructor admits two forms:

* no arguments,
* two mandatory arguments and one optional arguments.

Any other combination of arguments will raise an exception.

Args:
    udp: a user-defined problem (either C++ or Python - note that *udp* will be deep-copied
      and stored inside the :class:`~pygmo.core.unconstrained` instance)
    method (``str``): a string containing the unconstrain method chosen, one of [``'death penalty'``, ``'kuri'``, ``'weighted'``, ``'ignore_c'``, ``'ignore_o'``]
    weights (array-like object): the vector of weights to be used if the method chosen is "weighted"

Raises:
    ValueError: if either:

      * *udp* is unconstrained,
      * *method* is not one of [``'death penalty'``, ``'kuri'``, ``'weighted'``, ``'ignore_c'``, ``'ignore_o'``],
      * *weight* is not of the same size as the problem constraints (if the method ``'weighted'`` is selcted), or not empty otherwise.

    unspecified: any exception thrown by:

      * the constructor of :class:`pygmo.core.problem`,
      * the constructor of the underlying C++ class,
      * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
        signatures, etc.)

)";
}

std::string fast_non_dominated_sorting_docstring()
{
    return R"(fast_non_dominated_sorting(points)

Runs the fast non dominated sorting algorithm on the input *points*

Args:
    points (2d-array like object): the input points

Raises:
    ValueError: if *points* is malformed
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    ``tuple``: (*ndf*, *dl*, *dc*, *ndr*), where:

    * *ndf* (``list`` of 1D NumPy float array): the non dominated fronts
    * *dl* (``list`` of 1D NumPy float array): the domination list
    * *dc* (1D NumPy int array): the domination count
    * *ndr* (1D NumPy int array): the non domination ranks

)";
}

std::string nadir_docstring()
{
    return R"(nadir(points)

Computes the nadir point of a set of points, i.e objective vectors. The nadir is that point that has the maximum
value of the objective function in the points of the non-dominated front.

Complexity is :math:`\mathcal{O}(MN^2)` where :math:`M` is the number of objectives and :math:`N` is the number of points.

Args:
    points (2d-array like object): the input points

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
    points (2d-array like object): the input points

Raises:
    ValueError: if *points* is malformed
    TypeError: if *points* cannot be converted to a vector of vector floats

Returns:
    1D NumPy float array: the ideal point

)";
}

std::string hvwfg_docstring()
{
    return R"(__init__(stop_dimension = 2)

The hypervolume algorithm from the Walking Fish Group (2011 version).

This object can be passed as parameter to the various methods of the 
class :class:`~pygmo.core.hypervolume` as it derives from the hidden base
class :class:`~pygmo.core._hv_algorithm`

Args:
    stop_dimension (```int```): the input population

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
class :class:`~pygmo.core.hypervolume` as it derives from the hidden base
class :class:`~pygmo.core._hv_algorithm`

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
class :class:`~pygmo.core.hypervolume` as it derives from the hidden base
class :class:`~pygmo.core._hv_algorithm`

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
class :class:`~pygmo.core.hypervolume` as it derives from the hidden base
class :class:`~pygmo.core._hv_algorithm`

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
class :class:`~pygmo.core.hypervolume` as it derives from the hidden base
class :class:`~pygmo.core._hv_algorithm`

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
    pop (:class:`~pygmo.core.population`): the input population

Raises:
    ValueError: if *pop* contains a single-objective or a constrained problem

Examples:
    >>> from pygmo import *
    >>> pop = population(prob = zdt(id = 1), size = 20)
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
    hv_algo (deriving from :class:`~pygmo.core._hv_algorithm`): hypervolume algorithm to be used

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
    hv_algo (deriving from :class:`~pygmo.core._hv_algorithm`): hypervolume algorithm to be used

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
    hv_algo (deriving from :class:`~pygmo.core._hv_algorithm`): hypervolume algorithm to be used


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
    hv_algo (deriving from :class:`~pygmo.core._hv_algorithm`): hypervolume algorithm to be used

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
    hv_algo (deriving from :class:`~pygmo.core._hv_algorithm`): hypervolume algorithm to be used

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

**NOTE** This point is different from the one computed by pagmo::nadir as only the non dominated front is considered
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
* an :class:`~pygmo.core.algorithm`,
* a :class:`~pygmo.core.population`.

Through the UDI, the island class manages the asynchronous evolution (or optimisation)
of its :class:`~pygmo.core.population` via the algorithm's :func:`~pygmo.core.algorithm.evolve()`
method. Depending on the UDI, the evolution might take place in a separate thread (e.g., if the UDI is a
:class:`~pygmo.core.thread_island`), in a separate process (e.g., if the UDI is a
:class:`~pygmo.py_islands.mp_island`) or even in a separate machine (e.g., if the UDI is a
:class:`~pygmo.py_islands.ipyparallel_island`). The evolution is always asynchronous (i.e., running in the
"background") and it is initiated by a call to the :func:`~pygmo.core.island.evolve()` method. At any
time the user can query the state of the island and fetch its internal data members. The user can explicitly
wait for pending evolutions to conclude by calling the :func:`~pygmo.core.island.wait()` and
:func:`~pygmo.core.island.get()` methods.

Typically, pagmo users will employ an already-available UDI in conjunction with this class (see :ref:`here <py_islands>`
for a full list), but advanced users can implement their own UDI types. A user-defined island must implement
the following method:

.. code-block:: python

   def run_evolve(self, algo, pop):
     ...

The ``run_evolve()`` method of the UDI will use the input :class:`~pygmo.core.algorithm`'s
:func:`~pygmo.core.algorithm.evolve()` method to evolve the input :class:`~pygmo.core.population` and, once the evolution
is finished, it will return the evolved :class:`~pygmo.core.population`. Note that, since internally the :class:`~pygmo.core.island`
class uses a separate thread of execution to provide asynchronous behaviour, a UDI needs to guarantee a certain degree of
thread-safety: it must be possible to interact with the UDI while evolution is ongoing (e.g., it must be possible to copy
the UDI while evolution is undergoing, or call the ``get_name()``, ``get_extra_info()`` methods, etc.), otherwise the behaviour
will be undefined.

In addition to the mandatory ``run_evolve()`` method, a UDI may implement the following optional methods:

.. code-block:: python

   def get_name(self):
     ...
   def get_extra_info(self):
     ...

See the documentation of the corresponding methods in this class for details on how the optional
methods in the UDI are used by :class:`~pygmo.core.island`. This class is the Python counterpart of the C++ class
:cpp:class:`pagmo::island`.

An island can be initialised in a variety of ways using keyword arguments:

* if the arguments list is empty, a default :class:`~pygmo.core.island` is constructed, containing a
  :class:`~pygmo.core.thread_island` UDI, a :class:`~pygmo.core.null_algorithm` algorithm and an empty
  population with problem type :class:`~pygmo.core.null_problem`;
* if the arguments list contains *algo*, *pop* and, optionally, *udi*, then the constructor will initialise
  an :class:`~pygmo.core.island` containing the specified algorithm, population and UDI. If the *udi* parameter
  is not supplied, the UDI type is chosen according to a heuristic which depends on the platform, the
  Python version and the supplied *algo* and *pop* parameters:

  * if *algo* and *pop*'s problem provide at least the :attr:`~pygmo.thread_safety.basic` thread safety guarantee,
    then :class:`~pygmo.core.thread_island` will be selected as UDI type;
  * otherwise, if the current platform is Windows or the Python version is at least 3.4, then :class:`~pygmo.py_islands.mp_island`
    will be selected as UDI type, else :class:`~pygmo.py_islands.ipyparallel_island` will be chosen;
* if the arguments list contains *algo*, *prob*, *size* and, optionally, *udi* and *seed*, then a :class:`~pygmo.core.population`
  will be constructed from *prob*, *size* and *seed*, and the construction will then proceed in the same way detailed
  above (i.e., *algo* and the newly-created population are used to initialise the island's algorithm and population,
  and the UDI, if not specified, will be chosen according to the heuristic detailed above).

If the keyword arguments list is invalid, a :exc:`KeyError` exception will be raised.

)";
}

std::string island_evolve_docstring()
{
    return R"(evolve(n = 1)

Launch evolution.

This method will evolve the island’s :class:`~pygmo.core.population` using the island’s :class:`~pygmo.core.algorithm`.
The evolution happens asynchronously: a call to :func:`~pygmo.core.island.evolve()` will create an evolution task that
will be pushed to a queue, and then return immediately. The tasks in the queue are consumed by a separate thread of execution
managed by the :class:`~pygmo.core.island` object. Each task will invoke the ``run_evolve()`` method of the UDI *n*
times consecutively to perform the actual evolution. The island's population will be updated at the end of each ``run_evolve()``
invocation. Exceptions raised inside the tasks are stored within the island object, and can be re-raised by calling
:func:`~pygmo.core.island.get()`.

It is possible to call this method multiple times to enqueue multiple evolution tasks, which will be consumed in a FIFO (first-in
first-out) fashion. The user may call :func:`~pygmo.core.island.wait()` or :func:`~pygmo.core.island.get()` to block until all
tasks have been completed, and to fetch exceptions raised during the execution of the tasks.

Args:
     n (``int``): the number of times the ``run_evolve()`` method of the UDI will be called within the evolution task

Raises:
    OverflowError: if *n* is negative or larger than an implementation-defined value
    unspecified: any exception thrown by the underlying C++ method, or by failures at the intersection between C++ and
      Python (e.g., type conversion errors, mismatched function signatures, etc.)

)";
}

std::string island_get_docstring()
{
    return R"(get()

Block until evolution ends and re-raise the first stored exception.

This method will block until all the evolution tasks enqueued via :func:`~pygmo.core.island.evolve()` have been completed.
The method will then raise the first exception raised by any task enqueued since the last time :func:`~pygmo.core.island.wait()`
or :func:`~pygmo.core.island.get()` were called.

Raises:
    unspecified: any exception thrown by evolution tasks or by the underlying C++ method

)";
}

std::string island_wait_docstring()
{
    return R"(wait()

This method will block until all the evolution tasks enqueued via :func:`~pygmo.core.island.evolve()` have been completed.

)";
}

std::string island_busy_docstring()
{
    return R"(busy()

Check island status.

Returns:
    ``bool``: ``True`` if the island is evolving, ``False`` otherwise

)";
}

std::string island_get_algorithm_docstring()
{
    return R"(get_algorithm()

Get the algorithm.

It is safe to call this method while the island is evolving.

Returns:
    :class:`~pygmo.core.algorithm`: a copy of the island's algorithm

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
    algo (:class:`~pygmo.core.algorithm`): the algorithm that will be copied into the island

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
    :class:`~pygmo.core.population`: a copy of the island's population

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
    pop (:class:`~pygmo.core.population`): the population that will be copied into the island

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
the separate thread of execution within :class:`pygmo.core.island`. Evolution tasks running on this
UDI must involve :class:`~pygmo.core.algorithm` and :class:`~pygmo.core.problem` instances
that provide at least the :attr:`~pygmo.thread_safety.basic` thread safety guarantee, otherwise
errors will be raised during the evolution.

See also the documentation of the corresponding C++ class :cpp:class:`pagmo::thread_island`.

)";
}

std::string archipelago_docstring()
{
    return R"(Archipelago.

An archipelago is a collection of :class:`~pygmo.core.island` objects which provides a convenient way to perform
multiple optimisations in parallel.

This class is the Python counterpart of the C++ class :cpp:class:`pagmo::archipelago`.

)";
}

std::string archipelago_evolve_docstring()
{
    return R"(evolve(n = 1)

Evolve archipelago.

This method will call :func:`pygmo.core.island.evolve()` on all the islands of the archipelago.
The input parameter *n* represent the number of times the ``run_evolve()`` method of the island's
UDI is called within the evolution task.

Args:
     n (``int``): the parameter that will be passed to :func:`pygmo.core.island.evolve()`

Raises:
    unspecified: any exception thrown by :func:`pygmo.core.island.evolve()`

)";
}

std::string archipelago_busy_docstring()
{
    return R"(busy()

Check archipelago status.

Returns:
    ``bool``: ``True`` if at least one island is evolving, ``False`` otherwise

)";
}

std::string archipelago_wait_docstring()
{
    return R"(wait()

Block until all evolutions have finished.

This method will call :func:`pygmo.core.island.wait()` on all the islands of the archipelago.

)";
}

std::string archipelago_get_docstring()
{
    return R"(get()

Block until all evolutions have finished and raise the first exception that was encountered.

This method will call :func:`pygmo.core.island.get()` on all the islands of the archipelago.
If an invocation of :func:`pygmo.core.island.get()` raises an exception, then on the remaining
islands :func:`pygmo.core.island.wait()` will be called instead, and the raised exception will be re-raised
by this method.

Raises:
    unspecified: any exception thrown by any evolution task queued in the archipelago's
      islands

)";
}

} // namespace
