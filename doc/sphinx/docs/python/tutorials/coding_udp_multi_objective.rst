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
*C. M. Fonseca and P. J. Fleming,
An Overview of Evolutionary Algorithms in Multiobjective Optimization
in Evolutionary Computation, vol. 3, no. 1, pp. 1-16, March 1995*.

The implementation as UDP can be realized as follows:

.. doctest::

    >>> import pygmo as pg
    >>> import numpy as np


    >>> class FonsecaFleming():
    ...     # Pass dimensions to constructor
    ...     def __init__(self, n):
    ...         self.n = n
    ...
    ...     # Define objectives
    ...     def fitness(self, x):
    ...     f1 = 1-np.exp(-sum([(x[i]-1/np.sqrt(self.n))**2
    ...                         for i in range(1, self.n)]))
    ...     f2 = 1-np.exp(-sum([(x[i]+1/np.sqrt(self.n))**2
    ...                         for i in range(1, self.n)]))
    ...         return [f1, f2]
    ...
    ...     # Return number of objectives
    ...     def get_nobj(self):
    ...         return 2
    ...
    ...     # Return bounds of decision variables
    ...     def get_bounds(self):
    ...         return ([-4]*self.n, [4]*self.n)
    ...
    ...     # Return function name
    ...     def get_name(self):
    ...         return "Fonseca and Fleming function"

Note that the only difference between a mono- and multi-objective problem lies in the number of objectives.
Moreover, parameters such as the number of summands in the exponent `n` can be passed flexibly to the constructor
and used within the class.

Let's now create an object with `n=10` from our new UDP class and pass it to a pygmo :class:`~pygmo.problem`.

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
    [[0.93880491 0.08521805]]

And we are already done!
