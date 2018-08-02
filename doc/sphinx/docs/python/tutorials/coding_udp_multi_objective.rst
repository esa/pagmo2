.. _py_tutorial_coding_udp_multi_objective:

Coding a User Defined Problem with multiple objectives
------------------------------------------------------

In this chapter we show how to code an unconstrained user defined problem (UDP) with multiple objectives.
We assume that the mathematical formulation of problem is the following:

.. math::
    \begin{array}{ll}
      \mbox{minimize: } & f_{1}(x) = x^{2} \\
      & f_{2}(x) = (x-2)^{2} \\
      \mbox{subject to:} & 0 \le x \le 2
    \end{array}

which is a test function for multi-objective optimization being introduced in
*Schaffer, J. David (1984). Some experiments in machine learning using vector
evaluated genetic algorithms (artificial intelligence, optimization, adaptation,
pattern recognition) (PhD). Vanderbilt University.* and illustrated
`here <https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization>`_.

The implementation as UDP can be realized as follows:

.. doctest::

    >>> class Schaffer:
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

    >>> import pygmo as pg
    >>> # create UDP
    >>> prob = pg.problem(Schaffer())

In the next step, the problem can be solved straightforward using the NSGA2 algorithm from the :class:`~pygmo.algorithm` class:

.. doctest::

    >>> # create population
    >>> pop = pg.population(prob, size=20)
    >>> # select algorithm
    >>> algo = pg.algorithm(pg.nsga2(gen=40))
    >>> # run optimization
    >>> pop = algo.evolve(pop)
    >>> # extract results
    >>> fits, vectors = pop.get_f(), pop.get_x()
    >>> # extract and print non-dominated fronts
    >>> ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    >>> print(ndf) #doctest: +SKIP
    [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=uint64)]

And we are already done!
