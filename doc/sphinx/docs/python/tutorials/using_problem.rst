.. _py_tutorial_using_problem:

Use of the class :class:`~pygmo.problem`
=============================================

.. image:: ../../images/prob_no_text.png

The :class:`~pygmo.problem` class represents a generic optimization
problem. The user codes the details of such a problem in a separate class (the
user-defined problem, or UDP) which is then passed to :class:`~pygmo.problem`
that provides a single unified interface.

.. note:: The User Defined Problems (UDPs) are optimization problems (coded by the user) that can
          be used to build a pygmo object of type :class:`~pygmo.problem`

Some UDPs (optimization problems) are already provided with pygmo and we refer to them as pygmo UDPs.

For the purpose of this tutorial we are going to use a pygmo UDP called :class:`~pygmo.rosenbrock`
to show the basic construction of a :class:`~pygmo.problem`, but the same logic would also
apply to a custom UDPs, that is a UDP that is actually coded by the user.

Let us start:

.. doctest::

    >>> import pygmo as pg
    >>> prob = pg.problem(pg.rosenbrock(dim = 5))
    >>> print(prob) #doctest: +NORMALIZE_WHITESPACE
    Problem name: Multidimensional Rosenbrock Function
    	Global dimension:			5
    	Integer dimension:			0
    	Fitness dimension:			1
    	Number of objectives:			1
    	Equality constraints dimension:		0
    	Inequality constraints dimension:	0
    	Lower bounds: [-5, -5, -5, -5, -5]
    	Upper bounds: [10, 10, 10, 10, 10]
    <BLANKLINE>
        Has gradient: true
    	User implemented gradient sparsity: false
        Expected gradients: 5
    	Has hessians: false
    	User implemented hessians sparsity: false
    <BLANKLINE>
    	Fitness evaluations: 0
        Gradient evaluations: 0
    <BLANKLINE>
    	Thread safety: basic
    <BLANKLINE>


In the code above, after the trivial import of the pygmo package, we define a variable prob
by constructing a :class:`~pygmo.problem` from :class:`~pygmo.rosenbrock`, a multidimensional problem
constructed from its global dimensions. In the following line we inspect the :class:`~pygmo.problem`
We can see, at a glance, that the UDP :class:`~pygmo.rosenbrock` is a five dimensional, single objective, box constrained
problem for which neither gradients nor hessians nor sparsity information is provided in the UDP.

We also see that its fitness function has never been called hence the counter for fitness evaluations is
zero.

All of the information contained in the :class:`~pygmo.problem` print out can be retrieved using
the appropriate methods, for example:

.. doctest::

    >>> prob.get_fevals() == 0
    True

Lets check how a fitness computation increases the counter:

.. doctest::

    >>> prob.fitness([1,2,3,4,5])
    array([14814.])
    >>> prob.get_fevals() == 1
    True

We may also get back the UDP, and thus access all the methods not exposed in the
:class:`~pygmo.problem` interface at any time via the :func:`~pygmo.problem.extract()` method:

.. doctest::

    >>> udp = prob.extract(pg.rosenbrock)
    >>> type(udp)
    <class 'pygmo.core.rosenbrock'>
    >>> udp = prob.extract(pg.rastrigin)
    >>> udp is None
    True

Such an *extraction* will only work if the correct UDP type is passed as argument.
