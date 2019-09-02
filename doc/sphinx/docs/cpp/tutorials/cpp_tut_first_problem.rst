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


.. literalinclude:: ../../../../../tutorials/first_udp_ver1.cpp
   :language: c++
   :diff: ../../../../../tutorials/first_udp_ver0.cpp
