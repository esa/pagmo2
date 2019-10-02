Writing your first optimisation problem
=======================================

A fundamental requirement of any optimisation
library is the ability to represent an optimisation
problem to be solved. In this section, our
goal is to show how this is done in pagmo.

A simple problem
----------------

We will be considering the minimisation of
the bidimensional objective function

.. math::

   f\left(x_1, x_2\right) = \sqrt{x_2},

subject to the box bounds

.. math::

   \begin{align}
      x_1 & \in \left[ -0.5, 1 \right], \\
      x_2 & \in \left[ 0, 8 \right],
   \end{align}

and to the inequality constraints

.. math::

   \begin{align}
      x_2 & \geq 8x_1^3, \\
      x_2 & \geq \left(-x_1 + 1\right)^3.
   \end{align}

In pagmo's taxonomy, this optimisation problem is

* *continuous* (because :math:`x_1` and :math:`x_2`
  are real variables),
* *single-objective* (because the objective
  function produces a single value), and
* *constrained* (because there are nonlinear
  constraints in addition to the box bounds).

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
       return {std::sqrt(dv[1])};
   }

The other mandatory function, ``get_bounds()``, returns the
box bounds of the problem as a ``std::pair`` of lower/upper
bounds:

.. code-block:: c++

   std::pair<vector_double, vector_double> get_bounds() const
   {
       return {{-0.5, 0}, {1, 8}};
   }

In addition to returning the box bounds, the ``get_bounds()``
function plays another important role: it also
(implicitly) establishes the dimension of the problem
via the sizes of the returned lower/upper bounds vectors
(in this specific case, 2).

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

   std::cout << "Value of the objfun in (1, 2): " << p.fitness({1, 2})[0] << '\n';

We can also fetch the lower/upper box bounds of the UDP
via the :cpp:func:`pagmo::problem::get_lb()` and
:cpp:func:`pagmo::problem::get_ub()` member functions:

.. code-block:: c++

   std::cout << "Lower bounds: [" << p.get_lb()[0] << ", " << p.get_lb()[1] << "]\n";
   std::cout << "Upper bounds: [" << p.get_ub()[0] << ", " << p.get_ub()[1] << "]\n\n";

Printing ``p`` to screen via

.. code-block:: c++

   std::cout << p << '\n';

will produce a human-readable
summary that may look like this:

.. code-block:: none

   Problem name: 10problem_v0
         Global dimension:                       2
         Integer dimension:                      0
         Fitness dimension:                      1
         Number of objectives:                   1
         Equality constraints dimension:         0
         Inequality constraints dimension:       0
         Lower bounds: [-0.5, 0]
         Upper bounds: [1, 8]
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
For now, we would just like to point out that, from our simple
UDP definition, pagmo was able to infer
on its own various properties of the optimisation problem
(e.g., the problem dimension, the number of objectives,
the absence of constraints, etc.). pagmo is able to do this
thanks to both introspection capabilities (based on template
metaprogramming) and (hopefully) sensible defaults.

Adding the constraints
----------------------

.. literalinclude:: ../../../../../tutorials/first_udp_ver1.cpp
   :language: c++
   :diff: ../../../../../tutorials/first_udp_ver0.cpp
