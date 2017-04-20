.. _py_tutorial_coding_udp_constrained:

Coding a User Defined Problem with constraints
----------------------------------------------

We here show how to code a non-trivial user defined problem (UDP) with a single objective, equality and inequality constraints.
We assume that the mathematical formulation of problem is the following:

.. math::

   \begin{array}{rl}
   \mbox{minimize:} & \sum_{i=1}^3 \left[(x_{2i-1}-3)^2 / 1000 - (x_{2i-1}-x_{2i}) + \exp(20(x_{2i-1}-x_{2i}))\right]\\
   \mbox{subject to:} & -5, <= x_i, <= 5\\
   & 4(x_1-x_2)^2+x_2-x_3^2+x_3-x_4^2  = 0 \\
   & 8x_2(x_2^2-x_1)-2(1-x_2)+4(x_2-x_3)^2+x_1^2+x_3-x_4^2+x_4-x_5^2 = 0 \\
   & 8x_3(x_3^2-x_2)-2(1-x_3)+4(x_3-x_4)^2+x_2^2-x_1+x_4-x_5^2+x_1^2+x_5-x_6^2 = 0 \\
   & 8x_4(x_4^2-x_3)-2(1-x_4)+4(x_4-x_5)^2+x_3^2-x_2+x_5-x_6^2+x_2^2+x_6-x_1 = 0 \\
   & 8x_5(x_5^2-x_4)-2(1-x_5)+4(x_5-x_6)^2+x_4^2-x_3+x_6+x_3^2-x_2 <= 0 \\
   & 8x_6(x_6^2-x_5)-2(1-x_6)             +x_5^2-x_4+x_4^2 >= x_5 \\
   \end{array}

which is a modified instance of the problem 5.9 in Luksan, L., and Jan Vlcek. "Sparse and partially separable test problems
for unconstrained and equality constrained optimization." (1999). The modification is in the last two constraints that are,
for the purpose of this tutorial, considered as inequalities rather than equality constraints.

The problem at hand has box bounds, 4 equality constraints, two inequalities (note the different form of these) and one objective. Neglecting
for the time being the fitness, the basic structure for the UDP to have pygmo understand the problem type will be:

.. doctest::

    >>> class my_constrained_udp:
    ...     def get_bounds(self):
    ...         return ([-5]*6,[5]*6)
    ...     def get_nic(self):
    ...         return 2 
    ...     def get_nec(self):
    ...         return 4

Note how we need to specify both the number of equality constraints and the number of inequality constraints (as pygmo by default assumes 0 for both).
There is no need to specify the number of objectives as by default pygmo assumes single objective optimization. The full documenation on the UDP specification can 
be found in the :class:`pygmo.problem` docs.

We still have to write the fitness function as that is a mandatory method (together with ``get_bounds()``) for all UDPs. Constructing a :class:`~pygmo.problem` with
an incomplete UDP will fail. In pygmo the fitness includes both the objectives and the constraints according to the described order [obj,ec,ic]. All equality constraints
are in the form :math:`g(x) = 0`, while inequalities :math:`g(x) <= 0` as documented in :func:`pygmo.problem.fitness()`.

.. doctest::

    >>> import math
    >>> class my_constrained_udp:
    ...     def fitness(self, x):
    ...         obj = 0
    ...         for i in range(3):
    ...             obj += (x[2*i-2]-3)**2 / 1000. - (x[2*i-2]-x[2*i-1]) + math.exp(20.*(x[2*i - 2]-x[2*i-1]))
    ...         ce1 = 4*(x[0]-x[1])**2+x[1]-x[2]**2+x[2]-x[3]**2
    ...         ce2 = 8*x[1]*(x[1]**2-x[0])-2*(1-x[1])+4*(x[1]-x[2])**2+x[0]**2+x[2]-x[3]**2+x[3]-x[4]**2
    ...         ce3 = 8*x[2]*(x[2]**2-x[1])-2*(1-x[2])+4*(x[2]-x[3])**2+x[1]**2-x[0]+x[3]-x[4]**2+x[0]**2+x[4]-x[5]**2
    ...         ce4 = 8*x[3]*(x[3]**2-x[2])-2*(1-x[3])+4*(x[3]-x[4])**2+x[2]**2-x[1]+x[4]-x[5]**2+x[1]**2+x[5]-x[0]
    ...         ci1 = 8*x[4]*(x[4]**2-x[3])-2*(1-x[4])+4*(x[4]-x[5])**2+x[3]**2-x[2]+x[5]+x[2]**2-x[1]
    ...         ci2 = -(8*x[5] * (x[5]**2-x[4])-2*(1-x[5]) +x[4]**2-x[3]+x[3]**2 - x[4])
    ...         return [obj, ce1,ce2,ce3,ce4,ci1,ci2]
    ...     def get_bounds(self):
    ...         return ([-5]*6,[5]*6)
    ...     def get_nic(self):
    ...         return 2 
    ...     def get_nec(self):
    ...         return 4
    ...     def gradient(self, x):
    ...         return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

In order to check that the UDP above is wll formed for pygmo we try to construct a :class:`pygmo.problem` from it and inspect it:

.. doctest::

    >>> import pygmo as pg
    >>> prob = pg.problem(my_constrained_udp())
    >>> print(prob) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Problem name: ...
    	Global dimension:			6
    	Fitness dimension:			7
    	Number of objectives:			1
    	Equality constraints dimension:		4
    	Inequality constraints dimension:	2
    	Tolerances on constraints: [0, 0, 0, 0, 0, ... ]
    	Lower bounds: [-5, -5, -5, -5, -5, ... ]
    	Upper bounds: [5, 5, 5, 5, 5, ... ]
    <BLANKLINE>
    	Has gradient: true
    	User implemented gradient sparsity: false
    	Expected gradients: 42
    	Has hessians: false
    	User implemented hessians sparsity: false
    <BLANKLINE>
    	Function evaluations: 0
    	Gradient evaluations: 0
    <BLANKLINE>
    	Thread safety: none
    <BLANKLINE>

All seems in order. The dimensions are corresponding to what we wanted, the gradient is detected etc.

Solving a constrained User Defined Problem
----------------------------------------------

    >>> algo = pg.algorithm(pg.nlopt('auglag'))
    >>> algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
    >>> algo.set_verbosity(100)
    >>> pop = pg.population(prob = my_constrained_udp(), size = 1)
    >>> pop.problem.c_tol = [1E-6] * 6
    >>> algo.evolve(pop) # doctest: +SKIP

    >>> class add_gradient:
    ...     def __init__(self, prob):
    ...         if type(prob) is not pg.core.problem:
    ...             self.prob = pg.problem(prob)
    ...         else:
    ...             self.prob = prob
    ...     def fitness(self, x):
    ...         return self.prob.fitness(x)
    ...     def get_bounds(self):
    ...         return self.prob.get_bounds()
    ...     def get_nec(self):
    ...         return self.prob.get_nec()
    ...     def get_nic(self):
    ...         return self.prob.get_nic()
    ...     def get_nobj(self):
    ...         return self.prob.get_nobj()
    ...     def gradient(self, x):
    ...         return pg.estimate_gradient_h(lambda x: self.fitness(x), x)