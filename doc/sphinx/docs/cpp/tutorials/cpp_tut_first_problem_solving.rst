.. _cpp_tut_first_problem_solve:

Solving your first UDP
======================

Solving an optimization problem using an optimization algorithm is, in pagmo,
described as *evolving* a population. In the scientific literature, an
interesting
discussion has developed over the past decades on whether evolution is or not
a form of
optimization. In pagmo we take the opposite standpoint and we regard
optimization,
of all types, as a form of evolution. Regardless on whether you will
be using an SQP,
an interior point optimizer or an evolutionary strategy solver, in pagmo you
will always have to call a method called ``evolve()`` to improve over your
initial solutions,
i.e., your *population*.

In this section we continue with the simple UDP
from the :ref:`previous section <cpp_tut_first_problem>`
and look at ways to minimize it. The following example shows the use with of an evolutionary algorithm.

As previously defined the problem is
*continuous*, *deterministic*, *single objective* and *constrained*.
Looking at the :ref:`algorithms table <available_algorithms_problems>`
we can find multiple applicable optimization algorithms.

From the :ref:`global optimization heuristics <_heuristic_global_optimization>` we could use
Extended Ant Colony Optimization (:cpp:class:`pagmo::gaco`) or
Improved Harmony Search (:cpp:class:`pagmo::ihs`).

Ant Colony Optimization
-----------------------

Lets see how we can optimize the problem from the :ref:`previous section <cpp_tut_first_problem>`.
We import the necessary packages and define the problem (``problem_v1``), as in the previous section.

.. literalinclude:: ../../../../../tutorials/first_udp_ver1_solve.cpp
   :caption: first_udp_ver1_solve.cpp
   :language: c++
   :linenos:

In the ``main`` function we then specify the pagmo problem as was done before.

.. code-block:: c++

    problem prob{problem_v1{}};

The code after this is the part that we are interested in.
First we define the algorithm that we want to use for the optimization (:cpp:class:`pagmo::gaco`),
where *1000* is the number of generations.

.. code-block:: c++

    algorithm algo{gaco(1000)};

The next step is to initialie the population (i.e. a number of random solutions) for the problem.
This is done by passing the problem *prob* to the ``pop`` function together with the *population size*.

.. code-block:: c++

    population pop{prob, 100};

Third to actually evolve (i.e. optimize) the population
we pass the generated population (pop) to the ``algo.evolve`` function.

.. note::

    The population size for the Ant Colony Optimization has to be at
    least as big as the number of solutions stored.
    The number of solution is defined by the *ker* parameter of the
    ``gaco`` function call and defaults to *63*.


.. code-block:: c++

    pop = algo.evolve(pop);

Lastly, we print the individuals from the last generation.

.. code-block:: c++

    std::cout << pop;

This will also print the optimal solution (*Champion decision vector*)
as well as the results of the objective function and constraints (*Champion fitness*).


Improved Harmony Search
-----------------------

Switching to an alternative optimiser is straightforward.
If we want to use the Improved Harmony Search instead of GACO,
we include the header and change the algorithm definition:

.. code-block:: c++

    algorithm algo{ish(1000)};
