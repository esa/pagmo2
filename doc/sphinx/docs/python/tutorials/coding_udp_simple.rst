.. _py_tutorial_coding_udp_simple:

Coding a simple User Defined Problem
====================================

While pagmo provides a number of UDPs to help you test your own optimization strategy or user defined algorithm, the possibility
to write your own UDP is fundamental. In this tutorial we will show how to code a UDP. Remember that UDPs are classes that can be used 
to construct a :class:`~pygmo.core.problem` which, in turn, is what a :class:`~pygmo.core.algorithm` can solve.

We encourage the user to read the documentation of the class :class:`~pygmo.core.problem` to have a detailed list of methods that can be, or have to be,
implemented in a UDP. To start simple we consider the simple problem of minimizing the two dimensional sphere function.

In pagmo minimization is always assumed and should you need to maximize some objective function, just put a minus sign in front of that objective.

.. doctest::

    >>> class sphere_function:
    ...     def fitness(self, dv):
    ...         return [sum(dv*dv)]
    ...         
    ...     def get_bounds(self):
    ...         return ([-1,-1],[1,1])

The two mandatory methods you must implement in your class are ``fitness(self, dv)`` and ``def get_bounds(self)``. The problem dimension
will be inferred by the return value of the second, while the actual fitness of decision vectors will be computed calling the first.
It is important to remember that ``dv`` is assumed to be a NumPy array, so that the array arithmetic applies in the body of ``fitness``.
Note also how to define a UDP we do not need to inherit from some other pygmo related class. Since we do not define any other method pygmo 
will assume a single objective, no constraints, no gradients etc...

Lets now build a :class:`~pygmo.core.problem` from our new UDP.

.. doctest::

    >>> import pygmo as pg
    >>> sf = pg.problem(sphere_function())

That easy! To inspect what type of problem pygmo has detected from our UDP we may print on screen:

.. doctest::

    >>> print(sf) #doctest: +NORMALIZE_WHITESPACE
    Problem name: <class 'sphere_function'>
    	Global dimension:			2
    	Fitness dimension:			1
    	Number of objectives:			1
    	Equality constraints dimension:		0
    	Inequality constraints dimension:	0
    	Lower bounds: [-1, -1]
    	Upper bounds: [1, 1]
    <BLANKLINE>
    	Has gradient: false
    	User implemented gradient sparsity: false
    	Has hessians: false
    	User implemented hessians sparsity: false
    <BLANKLINE>
    	Function evaluations: 0
    <BLANKLINE>
    	Thread safety: none
    <BLANKLINE>

Lets now add some complexity. We want our UDP to be scalable and to have some decent name.

    >>> class sphere_function:
    ...     def __init__(self, dim):
    ...         self.dim = dim
    ...
    ...     def fitness(self, dv):
    ...         return [sum(dv*dv)]
    ...         
    ...     def get_bounds(self):
    ...         return ([-1] * self.dim, [1] * self.dim)
    ...
    ...     def get_name(self):
    ...         return "Sphere Function"
    ...
    ...     def get_extra_info(self):
    ...         return "\tDimensions: " + str(self.dim)
    >>> sf = pg.problem(sphere_function(3))
    >>> print(sf) #doctest: +NORMALIZE_WHITESPACE
    Problem name: Sphere Function
    	Global dimension:			3
    	Fitness dimension:			1
    	Number of objectives:			1
    	Equality constraints dimension:		0
    	Inequality constraints dimension:	0
    	Lower bounds: [-1, -1, -1]
    	Upper bounds: [1, 1, 1]
    <BLANKLINE>
    	Has gradient: false
    	User implemented gradient sparsity: false
    	Has hessians: false
    	User implemented hessians sparsity: false
    <BLANKLINE>
    	Function evaluations: 0
    <BLANKLINE>
    	Thread safety: none
    <BLANKLINE>
    Extra info:
    	Dimensions: 3

