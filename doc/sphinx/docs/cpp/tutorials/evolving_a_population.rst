.. _cpp_tutorial_evolving_population:

Evolving a population
=====================

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

After following the :ref:`installation guide <install>`,
you will be able to compile and run
C++ pagmo programs. The following example shows the use with no
multithreading of an algoritmic evolution:

.. literalinclude:: ../../../../../tutorials/nsga2_example.cpp
   :language: c++
   :linenos:

Place it into a ``nsga2.cpp`` text file and compile it (for example) with:

.. code-block:: console

   $ g++ -O2 -DNDEBUG -std=c++11 nsga2.cpp -ltbb -lboost_serialization
