.. _py_tutorial_coding_udp_constrained:

Coding a User Defined Problem with constraints (NLP)
----------------------------------------------------

We here show how to code a non-trivial user defined problem (UDP) with a single objective, equality and inequality constraints.
We assume that the mathematical formulation of problem is the following:

.. math::

   \begin{array}{rl}
   \mbox{minimize:} & \sum_{i=1}^3 \left[(x_{2i-1}-3)^2 / 1000 - (x_{2i-1}-x_{2i}) + \exp(20(x_{2i-1}-x_{2i}))\right]\\
   \mbox{subject to:} & -5 <= x_i <= 5\\
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

The problem at hand has box bounds, 4 equality constraints, two inequalities (note the different form of these) and one objective and,
in the taxonomy of optimization problems, can be categorized as a non linear programming (NLP) problem.


Neglecting for the time being the fitness, the basic structure for the UDP to have pygmo understand the problem type will be:

.. doctest::

    >>> class my_constrained_udp:
    ...     def get_bounds(self):
    ...         return ([-5]*6,[5]*6)
    ...     # Inequality Constraints
    ...     def get_nic(self):
    ...         return 2
    ...     # Equality Constraints
    ...     def get_nec(self):
    ...         return 4

Note how we need to specify both the number of equality constraints and the number of inequality constraints (as pygmo by default assumes
0 for both). There is no need to specify the number of objectives as by default pygmo assumes single objective optimization.
The full documenation on the UDP specification can be found in the :class:`pygmo.problem` docs.

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

In order to check that the UDP above is well formed for pygmo we try to construct a :class:`pygmo.problem` from it and inspect it:

.. doctest::

    >>> import pygmo as pg
    >>> prob = pg.problem(my_constrained_udp())
    >>> print(prob) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Problem name: ...
    	Global dimension:			6
    	Integer dimension:			0
    	Fitness dimension:			7
    	Number of objectives:			1
    	Equality constraints dimension:		4
    	Inequality constraints dimension:	2
    	Tolerances on constraints: [0, 0, 0, 0, 0, ... ]
    	Lower bounds: [-5, -5, -5, -5, -5, ... ]
    	Upper bounds: [5, 5, 5, 5, 5, ... ]
        Has batch fitness evaluation: false
    <BLANKLINE>
    	Has gradient: true
    	User implemented gradient sparsity: false
    	Expected gradients: 42
    	Has hessians: false
    	User implemented hessians sparsity: false
    <BLANKLINE>
    	Fitness evaluations: 0
    	Gradient evaluations: 0
    <BLANKLINE>
    	Thread safety: none
    <BLANKLINE>

All seems in order. The dimensions are corresponding to what we wanted, the gradient is detected etc.

Solving your constrained UDP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So we now have a UDP with constraints and a numerical gradient. Let's solve it. Many different startegies can be deployed
and we here will just try two a) using the augmented lagrangian method b) using monotonic basin hopping.
Consider the following script:

.. doctest::

    >>> #METHOD A
    >>> algo = pg.algorithm(uda = pg.nlopt('auglag'))
    >>> algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
    >>> algo.set_verbosity(200) # in this case this correspond to logs each 200 objevals
    >>> pop = pg.population(prob = my_constrained_udp(), size = 1)
    >>> pop.problem.c_tol = [1E-6] * 6
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    objevals:        objval:      violated:    viol. norm:
            1      5.148e+30              6        203.761 i
          201        1.27621              5       0.179321 i
          401        1.71251              5      0.0550095 i
          601        1.96643              5      0.0269182 i
          801        2.21529              5     0.00340511 i
         1001        2.25337              5    0.000478665 i
         1201        2.25875              4    6.60584e-05 i
    >>> print(pop.get_fevals()) # doctest: +SKIP
    3740
    >>> print(pop.get_gevals()) # doctest: +SKIP
    3696

The solution is here found after calling 3740 times the fitness function (~1200 objective evaluations and ~2500 constraints evaluations) and
3696 times the gradient (each corresponding to 6 fitness evaluations in our case, since :func:`pygmo.estimate_gradient_h()`) is used
to estimate the gradient numerically.

    >>> #METHOD B
    >>> algo = pg.algorithm(uda = pg.mbh(pg.nlopt("slsqp"), stop = 20, perturb = .2))
    >>> algo.set_verbosity(1) # in this case this correspond to logs each 1 call to slsqp
    >>> pop = pg.population(prob = my_constrained_udp(), size = 1)
    >>> pop.problem.c_tol = [1E-6] * 6
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    Fevals:          Best:      Violated:    Viol. Norm:         Trial:
        14    3.59547e+36              6        501.393              0 i
        19    1.89716e+38              5        432.423              0 i
        39    1.89716e+38              5        432.423              1 i
        44    1.89716e+38              5        432.423              2 i
        49    1.18971e+28              5        231.367              0 i
        171        2.25966             0              0              0
        176        2.25966             0              0              1
        379        2.25966             0              0              2
        384        2.25966             0              0              3
        389        2.25966             0              0              4
        682        2.25966             0              0              5
        780        2.25966             0              0              6
       1040        2.25966             0              0              7
       1273        2.25966             0              0              8
       1278        2.25966             0              0              9
       1415        2.25966             0              0             10
       1558        2.25966             0              0             11
       1563        2.25966             0              0             12
       1577        2.25966             0              0             13
       1645        2.25966             0              0             14
       1878        2.25966             0              0             15
       2051        2.25966             0              0             16
       2179        2.25966             0              0             17
       2184        2.25966             0              0             18
       2189        2.25966             0              0             19
       2194        2.25966             0              0             20
    >>> pop.problem.get_fevals() # doctest: +SKIP
    2195
    >>> pop.problem.get_gevals() # doctest: +SKIP
    1320

Both strategies in these runs converge to a local feasible minima of value 2.25966. Repeating the above solution
strategies from different initial populations, the value of 1.60799 is sometimes also be obtained.

Do not use a black-box solver if you do not have to
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We conclude this tutorial arguing how, contrary to common (bad) practices of part of the scientific community,
the use of appropriate local optimization algorithms is always to be preferred and heuristic approaches should only
be used in situations where they are needed as they otherwise are just a bad idea. Let's consider here the problem
suite in constrained optimization that was used during the CEC2006 competition. In pygmo, we have implemented such an UDP
in the class :class:`pygmo.cec2006`. Such a class does not implement a gradient since the competition was intended for
heuristic optimization. So we quickly code a meta-UDP that adds numerical gradients to a UDP without gradients:

    >>> class add_gradient:
    ...     def __init__(self, prob):
    ...             self.prob = pg.problem(prob)
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
    ...         return pg.estimate_gradient(lambda x: self.fitness(x), x) # we here use the low precision gradient

Super cool, I know. Such a meta-UDP is useful as it now allows calling, for example, SQP methods on the CEC2006 problem instances.
Consider here only one case: the problem ``g07``. You can complete this tutorial studying what happens in the remaining ones.

    >>> # Start adding the numerical gradient (low-precision) to the UDP
    >>> prob = pg.problem(add_gradient(pg.cec2006(prob_id = 7)))
    >>> # Define a solution strategy (SQP method)
    >>> algo = pg.algorithm(uda = pg.mbh(pg.nlopt("slsqp"), 20, .2))
    >>> pop = pg.population(prob, 1)
    >>> # Solve the problem
    >>> pop = algo.evolve(pop) # doctest: +SKIP
    >>> # Collect information
    >>> print(pop.champion_f[0]) # doctest: +SKIP
    24.306209068189599
    >>> pop.problem.get_fevals() # doctest: +SKIP
    1155
    >>> pop.problem.get_gevals() # doctest: +SKIP
    1022

The total number of evaluations made to solve the problem (at a precision of 1e-8) is thus 1155 + 1022 * 2 = 3599.
To compare, as an example, with what an heuristic method could deliver we check the table that appears in:

Huang, Vicky Ling, A. Kai Qin, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm
for constrained real-parameter optimization." Evolutionary Computation, 2006. CEC 2006. IEEE Congress on. IEEE, 2006.

to find that after 5000 fitness evaluations this particular problem is still not solved by the heuristic approach introduced in the paper.
