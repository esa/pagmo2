.. _py_tutorial_udp_meta_decorator:

Using the decorator meta-problem
--------------------------------

The :class:`decorator <pygmo.decorator_problem>` class is a Python meta-problem
that can be used to modify and customise on-the-fly any method belonging to the public API of a user-defined
problem (UDP).

Recall that, in PyGMO's jargon, *meta-problems* are UDPs that take another UDP as input, modifying
its behaviour in a variety of (hopefully) useful ways. The :class:`~pygmo.decompose` meta-problem, for instance,
transforms a multi-objective optimisation problem into a single-objective problem. This effect is achieved
in the fitness function of :class:`~pygmo.decompose`, which first invokes the fitness function
of the UDP used during construction (the *inner problem* of the meta-problem), and then transforms
the resulting multi-dimensional fitness vector into a scalar fitness. From a pythonic point of view,
we could say that :class:`~pygmo.decompose` `decorates <https://en.wikipedia.org/wiki/Decorator_pattern>`__
the fitness function of the inner problem according to a specific prescription that turns a
multi-objective fitness into a scalar one. The :class:`decorator <pygmo.decorator_problem>` problem
generalises this idea by allowing the user to arbitrarily decorate not only the fitness function,
but also any other method from the public API of a UDP.

The :class:`decorator <pygmo.decorator_problem>` problem is meant to be used in those situations
where it is desirable to quickly, temporarily and *non-intrusively* alter the behaviour of a UDP, and it should not
be used as a replacement for other meta-problems (e.g., if you need to decompose a multi-objective
UDP you should definitely use the :class:`~pygmo.decompose` problem, which will perform better
and provide more features without having to write any additional code). Users should also keep in mind
that Python provides many other ways of modifying the behaviour of a class, which, depending
on the situation, may be more appropriate than the :class:`decorator <pygmo.decorator_problem>` problem
(e.g., subclassing, monkey patching, etc.).

Hello, decorated world!
^^^^^^^^^^^^^^^^^^^^^^^

Like all meta-problems, :class:`~pygmo.decorator_problem` accepts a UDP as a construction parameter.
In this example, we will use a :class:`Rosenbrock <pygmo.rosenbrock>` problem for illustration
purposes (and to show that :class:`~pygmo.decorator_problem` works also with exposed C++ problems):

.. doctest::

   >>> import pygmo as pg
   >>> rb = pg.rosenbrock()

We can now proceed to the construction of a decorated Rosenbrock problem:

.. doctest::

   >>> drb = pg.problem(pg.decorator_problem(rb))

In this case, we did not pass any decorator to the constructor of :class:`~pygmo.decorator_problem`, so
*drb* will be functionally equivalent to an undecorated Rosenbrock problem. The only difference will
be that printing *drb* to screen will tell us that *rb* has been wrapped by a :class:`~pygmo.decorator_problem`:

.. doctest::

   >>> drb #doctest: +NORMALIZE_WHITESPACE
   Problem name: Multidimensional Rosenbrock Function [decorated]
           Global dimension:                       2
           Integer dimension:                      0
           Fitness dimension:                      1
           Number of objectives:                   1
           Equality constraints dimension:         0
           Inequality constraints dimension:       0
           Lower bounds: [-5, -5]
           Upper bounds: [10, 10]
   <BLANKLINE>
           Has gradient: true
           User implemented gradient sparsity: false
           Expected gradients: 2
           Has hessians: false
           User implemented hessians sparsity: false
   <BLANKLINE>
           Fitness evaluations: 0
           Gradient evaluations: 0
   <BLANKLINE>
           Thread safety: none
   <BLANKLINE>
   Extra info:
           No registered decorators.

So far so good, although not terribly exciting :)

Let us now write our first decorator. This decorator is meant to be applied to the fitness
function of the Rosenbrock problem. In addition to returning the original fitness,
it will also print on screen the time needed to compute it. The code is as follows:

.. doctest::

   >>> def f_decor(orig_fitness_function):
   ...     def new_fitness_function(self, dv):
   ...         import time
   ...         start = time.monotonic()
   ...         fitness = orig_fitness_function(self, dv)
   ...         print("Elapsed time: {} seconds".format(time.monotonic() - start))
   ...         return fitness
   ...     return new_fitness_function

The decorator ``f_decor()`` takes as input the original fitness function, and internally defines
a new fitness function. ``new_fitness_function()`` has exactly the same prototype as prescribed
by the UDP interface: it takes as input parameters the calling :class:`~pygmo.decorator_problem` object (``self``) and the
decision vector (``dv``), and returns a fitness vector computed via ``orig_fitness_function()``.
The call to the original fitness function is bracketed between a couple of lines of code that measure
the elapsed runtime via Python's :func:`time.monotonic()` function.

We can now construct a decorated Rosenbrock problem:

.. doctest::

   >>> drb = pg.problem(pg.decorator_problem(rb, fitness_decorator=f_decor))

As you can see, we have passed our decorator, ``f_decor``, as a keyword argument named ``fitness_decorator``
to the constructor of :class:`~pygmo.decorator_problem`. All decorators must be passed as keyword arguments
whose name ends in ``_decorator`` and starts with the UDP method to be decorated (in this case, ``fitness``).
The string representation of *drb* will now reflect that the fitness function has been decorated:

.. doctest::

   >>> drb #doctest: +NORMALIZE_WHITESPACE
   Problem name: Multidimensional Rosenbrock Function [decorated]
           Global dimension:                       2
           Integer dimension:                      0
           Fitness dimension:                      1
           Number of objectives:                   1
           Equality constraints dimension:         0
           Inequality constraints dimension:       0
           Lower bounds: [-5, -5]
           Upper bounds: [10, 10]
   <BLANKLINE>
           Has gradient: true
           User implemented gradient sparsity: false
           Expected gradients: 2
           Has hessians: false
           User implemented hessians sparsity: false
   <BLANKLINE>
           Fitness evaluations: 0
           Gradient evaluations: 0
   <BLANKLINE>
           Thread safety: none
   <BLANKLINE>
   Extra info:
           Registered decorators:
                   fitness

Let's now verify that the fitness function has been decorated as expected:

.. doctest::

   >>> fv = drb.fitness([1, 2]) # doctest: +ELLIPSIS
   Elapsed time: ... seconds
   >>> print(fv)
   [100.]

Yay!

Logging fitness evaluations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous section we saw an example of a simple stateless decorator.
Decorators, however, need not to be stateless: since all the functions of the UDP
API take as first input parameter the calling problem, we can implement decorators
that alter the state of the problem itself. As a concrete example, we will now
write a fitness function decorator that logs in the calling problem all the
decision vectors passed to the fitness function.

The fitness logging decorator is rather simple:

.. doctest::

   >>> def f_log_decor(orig_fitness_function):
   ...     def new_fitness_function(self, dv):
   ...         if hasattr(self, "dv_log"):
   ...             self.dv_log.append(dv)
   ...         else:
   ...             self.dv_log = [dv]
   ...         return orig_fitness_function(self, dv)
   ...     return new_fitness_function

The logic is straightforward:

* the first time the fitness function of the decorated problem is called, the
  condition ``hasattr(self, "dv_log")`` will be ``False`` because, initially,
  the decorated problem does not contain any logging structure. The decorated
  fitness function will then proceed to add to the problem a 1-element
  :class:`list` called ``dv_log`` containing the current decision vector ``dv``;
* on subsequent calls of the decorated fitness function, the current decision vector
  ``dv`` will be appended to the ``dv_log`` list.

Let's see the logging decorator in action. First, we create a decorated problem:

.. doctest::

   >>> drb = pg.problem(pg.decorator_problem(rb, fitness_decorator=f_log_decor))

Second, we verify that the UDP inside *drb* does not yet contain a ``dv_log`` logging structure:

.. doctest::

   >>> hasattr(drb.extract(pg.decorator_problem), "dv_log")
   False

Next, we call the fitness function a few times:

.. doctest::

   >>> drb.fitness([1, 2])
   array([100.])
   >>> drb.fitness([3, 4])
   array([2504.])
   >>> drb.fitness([5, 6])
   array([36116.])

We can now verify that all the decision vectors passed so far to the fitness function
have been logged in the internal :class:`~pygmo.decorator_problem` object:

   >>> drb.extract(pg.decorator_problem).dv_log
   [array([1., 2.]), array([3., 4.]), array([5., 6.])]

All according to plan!

Of course, the logging presented here is rather simplistic. In a real application, one may want to rely
on Python's :mod:`logging` module rather than use an ad-hoc logging structure, and perhaps one may want
to log other information as well (e.g., the fitness vector).

What else can be decorated?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the examples above, we have focused on the decoration of the fitness function, which, arguably, is
the most important function in a UDP. :class:`~pygmo.decorator_problem` however can be used to decorate
any method belonging to the public API of a UDP, including gradient and hessians computations,
sparsity-related methods, the stochastic seed setter and a getter, etc. The exhaustive list of methods
that can be implemented (and decorated) in a UDP is reported in the documentation of :class:`pygmo.problem`.
