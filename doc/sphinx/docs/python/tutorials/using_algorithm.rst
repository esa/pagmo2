.. _py_tutorial_using_algorithm:

Use of the class :class:`~pygmo.core.algorithm`
===============================================

The :class:`~pygmo.core.algorithm` class represents a generic optimization
algorithm. The user codes the details of such an algorithm in a separate class (the
user-defined algorithm, or UDA) which is then passed to :class:`~pygmo.core.algorithm`
that provides a single unified interface.

.. note::  The User Defined Algorithms (UDAs) are optimization algorithms (coded by the user) that can
           be used to build a pygmo object of type :class:`~pygmo.core.algorithm`

Some UDAs (optimization algorithms) are already provided with pygmo and we refer to them as pygmo UDAs.

For the purpose of this tutorial we are going to use a pygmo UDA called :class:`~pygmo.core.cmaes`
to show the basic construction of a :class:`~pygmo.core.algorithm`, but the same logic would also
apply to a custom UDAs, that is a UDA that is actually coded by the user.

Let us start:

.. doctest::

    >>> import pygmo as pg
    >>> algo = pg.algorithm(pg.cmaes(gen = 100, sigma0=0.3))
    >>> print(algo) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Algorithm name: CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy [stochastic]
    Extra info:
    	Generations: 100
    	cc: auto
    	cs: auto
    	c1: auto
    	cmu: auto
    	sigma0: 0.3
    	Stopping xtol: 1e-06
    	Stopping ftol: 1e-06
    	Memory: false
    	Verbosity: 0
    	Seed: ...

In the code above, after the trivial import of the pygmo package, we define a variable algo
by constructing a :class:`~pygmo.core.algorithm` from :class:`~pygmo.core.cmaes`, our implementation
of the Covariance Matrix Adaptation Evolutionary Strategy. To construct the pygmo UDA we also pass
some parameters (``gen`` and ``sigma0``) whose meaning is documented in :class:`~pygmo.core.cmaes`.
In the following line we inspect the :class:`~pygmo.core.algorithm`. We can see, at a glance, the
name of the :class:`~pygmo.core.algorithm` and some extra info that indicate the user (in this case us),
has implemented, in the UDA (in this case :class:`~pygmo.core.cmaes`), the optional method
``get_extra_info()`` that prints to screen some fundamental parameters defining the UDA.

We may also get access to the UDA, and thus to its methods not exposed in the
:class:`~pygmo.core.algorithm` interface, at any time via the :class:`~pygmo.core.algorithm.extract` method:

.. doctest::

    >>> uda = algo.extract(pg.cmaes)
    >>> type(uda)
    <class 'pygmo.core.cmaes'>
    >>> uda = algo.extract(pg.de)
    >>> type(uda)
    <class 'NoneType'>

Such an extraction will only work if the correct UDA type is passed as argument.
