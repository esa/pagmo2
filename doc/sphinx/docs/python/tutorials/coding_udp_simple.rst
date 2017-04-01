.. _py_tutorial_coding_udp_simple:

Coding a simple User Defined Problem
------------------------------------

While pagmo provides a number of UDPs to help you test your own optimization strategy or user defined algorithm, the possibility
to write your own UDP is at the core of pygmo's use. In this tutorial we will show how to code a UDP. Remember that UDPs are classes that can be used 
to construct a :class:`~pygmo.core.problem` which, in turn, is what a :class:`~pygmo.core.algorithm` can solve.

We encourage the user to read the documentation of the class :class:`~pygmo.core.problem` to have a detailed list of methods that can be, or have to be,
implemented in a UDP. To start simple we consider the simple problem of minimizing the two dimensional sphere function.

.. math::
   \begin{array}{ll}
     \mbox{minimize: } & x_1^2+x_2^2 \\
     \mbox{subject to:} & -1 \le x_i \le 1, i = 1..2
   \end{array}

In pagmo minimization is always assumed and should you need to maximize some objective function, just put a minus sign in front of that objective.

.. doctest::

    >>> class sphere_function:
    ...     def fitness(self, x):
    ...         return [sum(x*x)]
    ...         
    ...     def get_bounds(self):
    ...         return ([-1,-1],[1,1])

The two mandatory methods you must implement in your class are ``fitness(self, x)`` and ``def get_bounds(self)``. The problem dimension
will be inferred by the return value of the second, while the actual fitness of decision vectors will be computed calling the first method 
and passing as argument ``x`` as a NumPy array. It is important to remember that ``x`` is a NumPy array, so that the NumPy
array arithmetic applies in the body of ``fitness``. Note also how to define a UDP we do not need to inherit from some other
pygmo related class.  Since we do not define, in this case, any other method pygmo will assume a single objective, no constraints,
no gradients etc...

Lets now build a :class:`~pygmo.core.problem` from our new UDP.

.. doctest::

    >>> import pygmo as pg
    >>> prob = pg.problem(sphere_function())

That easy! To inspect what type of problem pygmo has detected from our UDP we may print on screen:

.. doctest::

    >>> print(prob) #doctest: +NORMALIZE_WHITESPACE
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

Lets now add some (mild) complexity. We want our UDP to be scalable:

.. math::
   \begin{array}{ll}
     \mbox{minimize: } & \sum_i x_i^2 \\
     \mbox{subject to:} & -1 \le x_i \le 1, i = 1..n
   \end{array}

and to have a human readable name.

    >>> class sphere_function:
    ...     def __init__(self, dim):
    ...         self.dim = dim
    ...
    ...     def fitness(self, x):
    ...         return [sum(x*x)]
    ...         
    ...     def get_bounds(self):
    ...         return ([-1] * self.dim, [1] * self.dim)
    ...
    ...     def get_name(self):
    ...         return "Sphere Function"
    ...
    ...     def get_extra_info(self):
    ...         return "\tDimensions: " + str(self.dim)
    >>> prob = pg.problem(sphere_function(3))
    >>> print(prob) #doctest: +NORMALIZE_WHITESPACE
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

Well that was easy, but now have a :class:`~pygmo.core.problem` to solve ... 

    >>> algo = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))
    >>> pop = pg.population(prob,10)
    >>> pop = algo.evolve(pop)
    >>> print(pop.champion_f) #doctest: +SKIP
    [  3.75822114e-06]

Wow those bees!! 

Possible pitfalls
^^^^^^^^^^^^^^^^^

Well that was nice as it worked like a charm. But the UDP can also be a rather complex class and the chances
that it is some how malformed are high. Lets see some common mistakes.

    >>> class sphere_function:
    ...     def fitness(self, x):
    ...         return [sum(x*x)]
    ...         
    >>> pg.problem(sphere_function()) #doctest: +SKIP
    NotImplementedError                       Traceback (most recent call last)
    ...
    NotImplementedError: the mandatory 'get_bounds()' method has not been detected in the user-defined Python problem
    '<sphere_function object at 0x1108cad68>' of type '<class 'sphere_function'>': the method is either not present or not callable


oops, I forgot to implement one of the two mandatory methods. In this case its not possible to construct a :class:`~pygmo.core.problem`
and, when we try, we then get a rather helpful error message. 

In other cases while the UDP is still malformed, the construction of :class:`~pygmo.core.problem` will succeed and the issue will
be revealed only when calling the malformed method:

    >>> class sphere_function:
    ...     def fitness(self, x):
    ...         return sum(x*x)
    ...         
    ...     def get_bounds(self):
    ...         return ([-1,-1],[1,1])
    >>> prob = pg.problem(sphere_function())
    >>> prob.fitness([1,2]) #doctest: +SKIP
    AttributeError                            Traceback (most recent call last)
    ...
    AttributeError: 'numpy.float64' object has no attribute '__iter__'

Notes on computational speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most performant way to write a UDP is to code it in C++ and expose it to python. Most UDPs that
are included in pygmo (see :ref:`py_problems`) are like that. When writing your own UDP, though, it is often quicker and less
painful to code, as shown in this tutorial, directly in python. What effect does this have w.r.t. the ideal
situation? Well, lets see, on an old test machine, a simple example: the scalable Rosenbrock function:

.. math::
   \begin{array}{ll}
     \mbox{minimize: } & \sum_{i=1}^{N-1} 100 (x_{i+1} - x_i^2 )^2 + (1-x_i)^2 \\
     \mbox{subject to:} & -5 \le x_i \le 10, i = 1..N
   \end{array}

which in pygmo can be quickly written as:

    >>> import numpy as np
    >>> class py_rosenbrock:
    ...     def __init__(self,dim):
    ...         self.dim = dim
    ...     def fitness(self,x):
    ...         retval = np.zeros((1,))
    ...         for i in range(len(x) - 1):
    ...             retval[0] += 100.*(x[i + 1]-x[i]**2)**2+(1.-x[i])**2
    ...         return retval
    ...     def get_bounds(self):
    ...         return (np.full((self.dim,),-5.),np.full((self.dim,),10.))

We now make a quick and dirty profiling instantiating a high dimensional instance of Rosenbrock: 2000 variables!!

    >>> prob_python = pg.problem(py_rosenbrock(2000))
    >>> prob_cpp = pg.problem(pg.rosenbrock(2000))
    >>> dummy_x = np.full((2000,), 1)
    >>> import time
    >>> start_time = time.time(); [prob.fitness(arr) for i in range(100)]; print(time.time() - start_time) #doctest: +SKIP
    0.4034...
    >>> start_time = time.time(); [prob2.fitness(arr) for i in range(100)]; print(time.time() - start_time) #doctest: +SKIP
    0.001353...

wait a minute ... really? two orders of magnitude? Do not panic. This is a very large problem and that for loop is not going to be
super optimized in python. Lets see if we can do better in these cases .... Let us use the jit decorator from numba

    >>> from numba import jit
    >>> class jit_rosenbrock:
    ...     def __init__(self,dim):
    ...         self.dim = dim
    ...     @jit
    ...     def fitness(self,x):
    ...         retval = np.zeros((1,))
    ...         for i in range(len(x) - 1):
    ...             retval[0] += 100.*(x[i + 1]-x[i]**2)**2+(1.-x[i])**2
    ...         return retval
    ...     def get_bounds(self):
    ...         return (np.full((self.dim,),-5.),np.full((self.dim,),10.))
    >>> prob_jit = pg.problem(jit_rosenbrock(2000))
    >>> start_time = time.time(); [prob_jit.fitness(arr) for i in range(100)]; print(time.time() - start_time) #doctest: +SKIP
    0.0059030...

much better right?


