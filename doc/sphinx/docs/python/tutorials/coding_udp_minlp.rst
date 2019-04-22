.. _py_tutorial_coding_udp_minlp:

Coding a User Defined Problem with an integer part (MINLP)
-----------------------------------------------------------

In this pygmo tutorial we show how to code a non-trivial user defined problem (UDP) with a single objective,
inequality constraints (no equality constraints in this case) and an integer part. We assume that
the mathematical formulation of problem is the following:

.. math::

   \begin{array}{rl}
   \mbox{minimize:} & \sum_{i=1}^3 \left[(x_{2i-1}-3)^2 / 1000 - (x_{2i-1}-x_{2i}) + \exp(20(x_{2i-1}-x_{2i}))\right]\\
   \mbox{subject to:} & -5, <= x_i, <= 5\\
   & 4(x_1-x_2)^2+x_2-x_3^2+x_3-x_4^2  <= 0 \\
   & 8x_2(x_2^2-x_1)-2(1-x_2)+4(x_2-x_3)^2+x_1^2+x_3-x_4^2+x_4-x_5^2 <= 0 \\
   & 8x_3(x_3^2-x_2)-2(1-x_3)+4(x_3-x_4)^2+x_2^2-x_1+x_4-x_5^2+x_1^2+x_5-x_6^2 <= 0 \\
   & 8x_4(x_4^2-x_3)-2(1-x_4)+4(x_4-x_5)^2+x_3^2-x_2+x_5-x_6^2+x_2^2+x_6-x_1 <= 0 \\
   & 8x_5(x_5^2-x_4)-2(1-x_5)+4(x_5-x_6)^2+x_4^2-x_3+x_6+x_3^2-x_2 <= 0 \\
   & 8x_6(x_6^2-x_5)-2(1-x_6)             +x_5^2-x_4+x_4^2 >= x_5 \\
   & x_5, x_6 \in \mathbb Z
   \end{array}

which is a modified instance of the problem 5.9 in Luksan, L., and Jan Vlcek. "Sparse and partially separable test problems
for unconstrained and equality constrained optimization." (1999). The modification is in the constraints that are,
for the purpose of this tutorial, considered as inequalities rather than equalities and in constraining the last two 
variables of the decision vector to be integers. The final problem, in the taxonomy of optimization problems, is categorized 
as a mixed integer non linear programming (MINLP) problem.

Neglecting, for the time being the fitness, the basic structure for the UDP to have pygmo understand the problem type will be:

.. doctest::

    >>> class my_minlp:
    ...     def get_bounds(self):
    ...         return ([-5]*6,[5]*6)
    ...     # Inequality Constraints
    ...     def get_nic(self):
    ...         return 6
    ...     # Integer Dimension
    ...     def get_nix(self):
    ...         return 2

Note how we need to specify explicitly the number of inequality constraints and the integer problem
dimension (as pygmo otherwise by default assumes 0 for both). Note also that the bounds (for the integer part)
must be integers, otherwise pygmo will complain. There is no need, for this case, to also specify explicitly the number of objectives
as by default pygmo assumes single objective optimization. The full documenation on the UDP optional methods can be
found in the :class:`pygmo.problem` docs.

We still have to write the fitness function, as that is a mandatory method (together with ``get_bounds()``) for all UDPs. Constructing a :class:`~pygmo.problem` with
an incomplete UDP will fail. In pygmo the fitness encapsulates both the objectives and the constraints in the mandatory order
[obj,ec,ic]. All inequality constraints are in the form :math:`g(x) <= 0` as documented in :func:`pygmo.problem.fitness()`. 

.. doctest::

    >>> import math
    >>> class my_minlp:
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
    ...         return 6
    ...     def get_nix(self):
    ...         return 2

In order to check that the UDP above is well formed we try to construct a :class:`pygmo.problem` from it and inspect it:

.. doctest::

    >>> import pygmo as pg
    >>> prob = pg.problem(my_minlp())
    >>> print(prob) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Problem name: ...
    	Global dimension:			6
    	Integer dimension:			2
    	Fitness dimension:			7
    	Number of objectives:			1
    	Equality constraints dimension:		0
    	Inequality constraints dimension:	6
    	Tolerances on constraints: [0, 0, 0, 0, 0, ... ]
    	Lower bounds: [-5, -5, -5, -5, -5, ... ]
    	Upper bounds: [5, 5, 5, 5, 5, ... ]
        Has batch fitness evaluation: false
    <BLANKLINE>
    	Has gradient: false
    	User implemented gradient sparsity: false
    	Has hessians: false
    	User implemented hessians sparsity: false
    <BLANKLINE>
    	Fitness evaluations: 0
    <BLANKLINE>
    	Thread safety: none
    <BLANKLINE>

All seems in order. The dimensions are corresponding to what we wanted, no gradient is detected etc.

Solving your MINLP by relaxation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MINLP problems are among the most difficult problems in optimization and not many generic approaches exist that
are able to effectively tackle these problems. For the purpose of this tutorial we show a possible solution approach for
the MINLP at hand based on a relaxation technique. In essence, we remove the integer constraints and solve the problem
in :math:`\mathbb R^6`. We then take the solution, fix the last two components to the nearest feasible integers, and
solve again the resulting, reduced problem in :math:`\mathbb R^4`.

To actuate the above strategy (which is here just as an example and is indeed not guaranteed to find the best solution)
we need a good NLP solver for the relaxed version of our problem. Thus we need the gradients of our objective function
and constraints. So we add them:

    >>> def _gradient(self, x):
    ...     return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
    >>> my_minlp.gradient = _gradient
    >>> # We need to reconstruct the problem as we changed its definition (adding the gradient)
    >>> prob = pg.problem(my_minlp())
    >>> prob.c_tol = [1e-8]*6

Note that, in this UDP, taking the gradient with respect to the integer part of the decision vector makes sense as it contains
relevant information, but that is not always the case. Whenever the gradient of your UDP does not contain any information,
relaxation techniques are not really an option and some global heuristic approach (e.g. evolutionary) may be the only way to go.

Pygmo's support for MINLP problems is built around the idea of making integer relaxation very easy. So we can
call an NLP solver (or any other suitable algorithm) on our MINLP and the relaxed version of the problem will be solved
returning a population with decision vectors that violate the integer constraints.

    >>> # We run 20 instances of the optimization in parallel via a default archipelago setup
    >>> archi = pg.archipelago(n = 20, algo = pg.ipopt(), prob = my_minlp(), pop_size=1)
    >>> archi.evolve(2); archi.wait()
    >>> # We get the best of the parallel runs
    >>> a = archi.get_champions_f()
    >>> a2 = sorted(archi.get_champions_f(), key = lambda x: x[0])[0]
    >>> best_isl_idx = [(el == a2).all() for el in a].index(True)
    >>> x_best = archi.get_champions_x()[best_isl_idx]
    >>> f_best = archi.get_champions_f()[best_isl_idx]
    >>> print("Best relaxed solution, x: {}".format(x_best)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Best relaxed solution, x:  [...  
    >>> print("Best relaxed solution, f: {}".format(f_best)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Best relaxed solution, f:  [...  

The relaxed version of the problem has a global optimal solution with :math:`x_5 = 0.75822315`, :math:`x_6 = 0.91463117`, which
suggests to look for solutions considering the values :math:`x_5 \in [0,1]`, :math:`x_6 \in [0,1]`. For each of the four 
possible cases we thus fix the box bounds on the last two variables. In case :math:`x_5 = 0`, :math:`x_6 = 0` we get:

    >>> # We fix the box bounds for x5 and x6
    >>> def get_bounds_(self):
    ...     return ([-5]*4+[0,0],[5]*4+[0,0])
    >>> my_minlp.get_bounds = get_bounds_
    >>> # We need to reconstruct the problem as we changed its definition (modified the bounds)
    >>> prob = pg.problem(my_minlp())
    >>> prob.c_tol = [1e-14]*4 + [0] * 2
    >>> # We solve the problem, this time using one only process
    >>> pop = pg.population(prob)
    >>> x_best[-1] = 0; x_best[-2] = 0
    >>> pop.push_back(x_best)
    >>> algo = pg.algorithm(pg.ipopt())
    >>> pop = algo.evolve(pop)
    >>> print("Best objective: ", pop.champion_f[0]) # doctest: +SKIP
    Best objective:  134.065695174
    >>> print("Best decision vector: ", pop.champion_x) # doctest: +SKIP
    Best decision vector:  [ 0.4378605   0.33368365 -0.75844494 -1.          0.          0.        ]

We found a feasible solution! Note that in the other 3 cases no feasible solution exists.

.. note::
   The solution strategy above is, in general, flawed in assuming the best solution of the relaxed problem is close to the
   the full MINLP problem solution. More sophisticated techniques would instead search the combinatorial part more exhaustvely.
   We used here this approach only to show how simple is, in pygmo, to define and solve the relaxed problem and
   to then feedback the optimal decision vector into a MINLP solution strategy. 
