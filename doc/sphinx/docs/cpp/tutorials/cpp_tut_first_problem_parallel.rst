.. _cpp_tut_first_problem_parallel:

Parallelizing your first UDP
============================

In the :ref:`definition section <cpp_tut_first_problem>`
we showed how a User Defined Problem is constructed.
In the :ref:`previous section <cpp_tut_first_problem_solve>`
we showed how to solve this problem with two different optimizers.

If you executed the example from the previous section, you probably noticed that this setup executes almost immediately
as the population and the number of generations is small.
However, finding good solutions to optimization problems can take a much larger population and higher number of generations
and hence significant amount of time.
Thus, parallelization of the optimization can be extremely important.

In pagmo parallelization is achieved via :ref:`islands <cpp_island>` and :ref:`archipelagos <cpp_archipelago>`.
**Islands** are used to asynchronously evolve populations. So one can launch multiple islands to evolve populations in parallel.
An **archipelago** is a collection of islands which we can use to evolve a population in parallel.


Archipelagos for Parallelization
--------------------------------

The difference between the sequential and parallel solutions are as follows.


.. literalinclude:: ../../../../../tutorials/first_udp_ver1_solve_parallel.cpp
   :language: c++
   :diff: ../../../../../tutorials/first_udp_ver1_solve.cpp

Thus, as in the previous sections we create the problem and create the algorithm to solve the problem.
However, instead of constructing a single population we construct a :cpp:class:`pagmo::archipelago`.
An archipelago is a collection of :cpp:class:`pagmo::island` objects, in this case the archipelago consists of *16 islands*.
Additional parameters required for constructing the islands is the problem (*prob*) and algorithm (*algo*)
along with the population size of each island (*1000*).

.. code-block:: c++

    archipelago archi{16, algo, p, 1000};;

Next we call ``archi.evolve(1)``, this function call evolves the population of each island **once** for 1000 generations
(defined when setting up the algorithm).

.. code-block:: c++

    archi.evolve(1);

As islands evolve asychronously we need to wait until all islands evolved. This can be done via ``archi.wait_check()``.

.. code-block:: c++

    archi.wait_check();


After that we can print the performance of the champion solutions of each island.
This will print a total of 16 champion solutions as there are 16 islands.

.. code-block:: c++

    for (const auto &isl : archi) {
        std::cout << isl.get_population().champion_f()[0] << '\n';
    }

Topology
--------

When creating an archipelago the default is that the islands in the archipelago are not connect,
i.e. the population on each island evolves independently.
Creating dependencies between the islands is possible by changing the :ref:`topology <cpp_topology>` of the archipelago.
This allows individuals of the different populations to migrate to other islands and thus for the best individuals
across all populations to propagate their features throughout the global population.

We can create a fully connected archipelago (every island is connected to all other islands) by introducing
the following changes:

.. literalinclude:: ../../../../../tutorials/first_udp_ver1_solve_parallel_topology.cpp
   :language: c++
   :diff: ../../../../../tutorials/first_udp_ver1_solve_parallel.cpp

We construct a :cpp:class:`pagmo::fully_connected` topology.

.. code-block:: c++

    fully_connected topo{};;

And when constructing the archipelago we simply pass the topology as the first parameter.

.. code-block:: c++

    archipelago archi{topo, 16, algo, prob, 1000};


In the simple unconnected example we evolved the populations of each island once for 1000 generations.
However, the migration of individuals between islands only happens after each evolution, i.e. if we only evolve the populations once there is no migration.
Hence, in this case we evolve the populations 10 times for 100 generations each (i.e. migration happens 10 times).

.. code-block:: c++

    archi.evolve(10);




