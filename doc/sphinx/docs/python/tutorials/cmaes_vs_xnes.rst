.. _py_tutorial_cmaes_vs_xnes:

A comparison between CMA-ES and xNES
===============================================

In this tutorial we will show how to use pygmo to comapre the performances of the two UDAs: :class:`~pygmo.cmaes` and :class:`~pygmo.xnes` 

Both algorithms are based on the idea of updating the gaussian distribution defining the generation of new samples and are the
subject of active research.

.. doctest::

    >>> import pygmo as pg
    >>> # The user-defined problem
    >>> udp = pg.rosenbrock(dim = 10)
    >>> # The pygmo problem
    >>> prob = pg.problem(udp)
    >>> xnes = pg.algorithm(pg.xnes(gen = 40000, ftol=1e-8, xtol=1e-14))
    >>> cmaes = pg.algorithm(pg.cmaes(gen = 40000, ftol=1e-8, xtol=1e-14))

