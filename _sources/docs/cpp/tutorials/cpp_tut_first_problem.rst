Writing your first optimisation problem
=======================================

A fundamental requirement of any optimisation
library is the ability to represent an optimisation
problem to be solved. In this section, our
goal is to show how this is done in pagmo.

A simple problem
----------------

We will be considering the minimisation of
the multidimensional objective function

.. math::

   f\left(x_1, x_2, x_3, x_4\right) = x_1x_4(x_1+x_2+x_3) + x_3,

subject to the box bounds

.. math::

   1 \le x_{1,2,3,4} \le 5

and to the constraints

.. math::

   \begin{align}
      x_1^2+x_2^2+x_3^2+x_4^2 - 40 &= 0, \\
      25 - x_1 x_2 x_3 x_4 &\le 0.
   \end{align}

In pagmo's taxonomy, this optimisation problem is

* *continuous* (because :math:`x_{1,2,3,4}` are real variables),
* *deterministic* (as neither the objectives nor the constraints depend
  on stochastic variables),
* *single-objective* (because the objective
  function produces a single value), and
* *constrained* (because there are nonlinear
  constraints in addition to the box bounds).

.. note::

   In this chapter, we will focus on the implementation
   of a single-objective
   continuous problem. pagmo can also represent many other
   types of problems, including multi-objective and
   (mixed) integer problems. TODO link to the deep dive.

We will start by implementing the objective function
and the box bounds first, and we will then add
the constraints at a later stage.

The bare necessities
--------------------

In pagmo, an optimisation problem must provide
at the very least an objective function and
the box bounds. Let's see how this is done
for our simple example problem:

.. literalinclude:: ../../../../../tutorials/first_udp_ver0.cpp
   :caption: first_udp_ver0.cpp
   :language: c++
   :linenos:

Let's analyse this code.

After the inclusion of the necessary header files, and a helpful

.. code-block:: c++

   using namespace pagmo;

to reduce typing, we reach the definition
of our optimisation
problem via a class called ``problem_v0``. As explained
in the :ref:`section about type erasure <cpp_tut_type_erasure>`,
this class does not need to derive from any base class.
It is just a "regular" class implementing two specific
member functions:

* ``fitness()``, which is used to compute the value of the
  objective function for the input decision
  vector ``dv``, and
* ``get_bounds()``, which is used to fetch the box bounds of
  the problem.

As you can notice, the signatures of both ``fitness()`` and ``get_bounds()``
use the type :cpp:type:`pagmo::vector_double`. This is the type
used by pagmo to represent decision and fitness vectors, and currently
it is just an alias for ``std::vector<double>``.

The objective function is called ``fitness()`` in pagmo because
it is used to return not only the value of the objective function,
but also of the constraints (thus in some sense it computes the
overall "fitness"
of the input decision vector). In this specific case, however,
our optimisation problem does not have constraints yet,
and thus the ``fitness()`` implementation just returns
a vector of size 1 whose only element is the value
of the (single) objective function:

.. code-block:: c++

   vector_double fitness(const vector_double &dv) const
   {
       return {dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2]};
   }

The other mandatory function, ``get_bounds()``, returns the
box bounds of the problem as a ``std::pair`` of lower/upper
bounds:

.. code-block:: c++

   std::pair<vector_double, vector_double> get_bounds() const
   {
       return {{1., 1., 1., 1.}, {5., 5., 5., 5.}};
   }

In addition to returning the box bounds, the ``get_bounds()``
function plays another important role: it also
(implicitly) establishes the dimension of the problem
via the sizes of the returned lower/upper bounds vectors
(in this specific case, 4).

Meet pagmo::problem
-------------------

After the definition of our optimisation problem, ``problem_v0``,
we encounter the ``main()`` function. In the ``main()``, the first
thing we do is to construct a :cpp:class:`pagmo::problem` from an
instance of ``problem_v0``:

.. code-block:: c++

   problem p{problem_v0{}};

:cpp:class:`pagmo::problem` is pagmo's
:ref:`type-erased <cpp_tut_type_erasure>` interface to optimisation
problems. It is a generic container which can store internally
an instance of any class which "acts like" an optimisation problem,
that is, any class which provides (at least) the two member functions
described earlier (``fitness()`` and ``get_bounds()``). In the pagmo
jargon, we refer to classes which "act like" optimisation problems
as *user-defined problems*, or UDPs.

In addition to storing a UDP (which, by itself, would not be
that useful), :cpp:class:`pagmo::problem` provides various member
functions to access the properties and capabilities of the UDP.
We can, for instance, call the :cpp:func:`pagmo::problem::fitness()`
member function of ``p`` to invoke the fitness function of the UDP:

.. code-block:: c++

   std::cout << "Value of the objfun in (1, 2, 3, 4): " << p.fitness({1, 2, 3, 4})[0] << '\n';

We can also fetch the lower/upper box bounds of the UDP
via the :cpp:func:`pagmo::problem::get_lb()` and
:cpp:func:`pagmo::problem::get_ub()` member functions:

.. code-block:: c++

   // Fetch the lower/upper bounds for the first variable.
   std::cout << "Lower bounds: [" << p.get_lb()[0] << "]\n";
   std::cout << "Upper bounds: [" << p.get_ub()[0] << "]\n\n";

Printing ``p`` to screen via

.. code-block:: c++

   std::cout << p << '\n';

will produce a human-readable
summary that may look like this:

.. code-block:: none

   Problem name: 10problem_v0
         Global dimension:                       4
         Integer dimension:                      0
         Fitness dimension:                      1
         Number of objectives:                   1
         Equality constraints dimension:         0
         Inequality constraints dimension:       0
         Lower bounds: [1, 1, 1, 1]
         Upper bounds: [5, 5, 5, 5]
         Has batch fitness evaluation: false

         Has gradient: false
         User implemented gradient sparsity: false
         Has hessians: false
         User implemented hessians sparsity: false

         Fitness evaluations: 1

         Thread safety: basic

Quite a mouthful! Do not worry about deciphering this
output right now, as we will examine the more intricate
aspects of the definition of an optimisation problem in due time.

For now, let us just point out that, from our simple
UDP definition, pagmo was able to infer
on its own various properties of the optimisation problem
(e.g., the problem dimension, the number of objectives,
the absence of constraints, etc.). pagmo is able to do this
thanks to both introspection capabilities (based on template
metaprogramming) and (hopefully) sensible defaults.

Adding the constraints
----------------------

In order to implement the constraints in our UDP we have to:

* add a couple of member functions which describe the type
  and number of constraints,
* modify the fitness function to return, in addition to the
  objective function, also the value of the constraints for
  an input decision vector.

Let us see the code:

.. literalinclude:: ../../../../../tutorials/first_udp_ver1.cpp
   :language: c++
   :diff: ../../../../../tutorials/first_udp_ver0.cpp

In order to specify the type and number of constraints in our
optimisation problem, we have to implement the two member
functions ``get_nec()``, which returns the number of equality
constraints, and ``get_nic()``, which returns the number of
inequality constraints:

.. code-block:: c++

   vector_double::size_type get_nec() const
   {
      return 1;
   }
   vector_double::size_type get_nic() const
   {
      return 1;
   }

Note that the number of (in)equality constraints is represented
via the size type of :cpp:type:`~pagmo::vector_double`
(which is an unsigned integral type, usually ``std::size_t``).

Next, we need to modify our fitness function to compute,
in addition to the objective function, the (in)equality
constraints for an input decision vector.
pagmo adopts the following conventions:

* the constraints are expressed as equations with zero
  on the right-hand-side,
* the values returned by the fitness function are computed
  from the left-hand-sides of the constraint equations,
* the inequality constraints are expressed via
  a less-than-or-equal relation (:math:`\leq`),
* in the fitness vector, the equality constraints
  follow the value(s) of the objective function and precede
  the inequality constraints.

In our specific example, we have 1 equality constraint and 1
inequality constraint,

.. math::

   \begin{align}
      x_1^2+x_2^2+x_3^2+x_4^2 - 40 &= 0, \\
      25 - x_1 x_2 x_3 x_4 &\le 0,
   \end{align}

and thus the fitness function will have to return a vector
with 3 values, which are, in order, the objective function,
the equality constraint and the inequality constraint:

.. code-block:: c++

   vector_double fitness(const vector_double &dv) const
   {
      return {
         dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2],                     // objfun
         dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3] - 40., // equality con.
         25. - dv[0] * dv[1] * dv[2] * dv[3]                                  // inequality con.
      };
   }

Now we can create a :cpp:class:`pagmo::problem` from our
new UDP, and, if we print it to screen, we can verify how
the :cpp:class:`pagmo::problem` class has correctly
identified the number and type of constraints from the
implementation of our UDP:

.. code-block:: none

   Problem name: 10problem_v1
         Global dimension:                       4
         Integer dimension:                      0
         Fitness dimension:                      3
         Number of objectives:                   1
         Equality constraints dimension:         1
         Inequality constraints dimension:       1
         Tolerances on constraints: [0, 0]
         Lower bounds: [1, 1, 1, 1]
         Upper bounds: [5, 5, 5, 5]
         Has batch fitness evaluation: false

         Has gradient: false
         User implemented gradient sparsity: false
         Has hessians: false
         User implemented hessians sparsity: false

         Fitness evaluations: 3

         Thread safety: basic

We can also verify that the :cpp:func:`pagmo::problem::fitness()`
function now produces a vector with three components:

.. code-block:: c++

   // Compute the value of the objective function, equality and
   // inequality constraints in the point (1, 2, 3, 4).
   const auto fv = p.fitness({1, 2, 3, 4});
   std::cout << "Value of the objfun in (1, 2, 3, 4): " << fv[0] << '\n';
   std::cout << "Value of the eq. constraint in (1, 2, 3, 4): " << fv[1] << '\n';
   std::cout << "Value of the ineq. constraint in (1, 2, 3, 4): " << fv[2] << '\n';
