.. _py_tutorial_coding_udp_multi_objective:

Coding a User Defined Problem with constraints multiple objectives
------------------------------------------------------------------

In this chapter we show how to code an unconstrained user defined problem (UDP) with multiple objectives.
We assume that the mathematical formulation of problem is the following:

\begin{array}{rl}
\mbox{minimize:} & f_{1}\left(\boldsymbol{x}\right) = 1 - \exp \left[-\sum_{i=1}^{n} \left(x_{i} - \frac{1}{\sqrt{n}} \right)^{2} \right]\\
& f_{2}\left(\boldsymbol{x}\right) = 1 - \exp \left[-\sum_{i=1}^{n} \left(x_{i} + \frac{1}{\sqrt{n}} \right)^{2} \right]
\mbox{subject to:} & -4 <= x_i <= 4\\
& 1 <= i <= n\\
\end{array}

which is a test function for multi-objective optimization being introduced in
*Schaffer, J. David (1984). Some experiments in machine learning using vector
evaluated genetic algorithms (artificial intelligence, optimization, adaptation,
pattern recognition) (PhD). Vanderbilt University.* and illustrated
`here <https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization>`_

The implementation as UDP can be realized as follows:

.. doctest::

    >>> import pygmo as pg
    >>> import numpy as np


    >>> class FonsecaFleming():
    ...     # Define objectives
    ...     def fitness(self, x):
    ...         f1 = x[0]**2
    ...         f2 = (x[0]-2)**2
    ...         return [f1, f2]
    ...
    ...     # Return number of objectives
    ...     def get_nobj(self):
    ...         return 2
    ...
    ...     # Return bounds of decision variables
    ...     def get_bounds(self):
    ...         return ([0]*1, [2]*1)
    ...
    ...     # Return function name
    ...     def get_name(self):
    ...         return "Schaffer function N.1"

Note that the only difference between a mono- and multi-objective problem lies in the number of objectives.

Let's now create an object from our new UDP class and pass it to a pygmo :class:`~pygmo.problem`.

.. doctest::

    >>> # create UDP
    >>> prob = pg.problem(FonsecaFleming(10))

In the next step, the problem can be solved straightforward using the NSGA2 algorithm from the :class:`~pygmo.algorithm` class:

.. doctest::

    >>> # create population
    >>> pop = pg.population(prob, size=200)
    >>> # select algorithm
    >>> algo = pg.algorithm(pg.nsga2(gen=40))
    >>> # run optimization
    >>> pop = algo.evolve(pop)
    >>> # print results
    >>> fits, vectors = pop.get_f(), pop.get_x()
    >>> print(fits[:1]) #doctest: +SKIP
    [[1.44097647e-08 3.99951985e+00]]

And we are already done!
