.. _py_tutorial_coding_udi_simple:

Coding a User Defined Island
------------------------------------

While pagmo provides a number of UDIs (see :ref:`islands`) to provide access to a number of parallelization technologies, the expert user
can code his own expanding pygmo functionalities. In this tutorial we will show how to code a UDI. Remember that UDIs are classes that can be used 
to construct a :class:`~pygmo.island` which, in turn, is what a :class:`~pygmo.archipelago` can use to distribute / parallelize optimization
tasks and have solutions migrate improving the overall optimization quality via the generalized island model.

We encourage the user to read the documentation of the class :class:`~pygmo.island` to have a detailed list of methods that can be, or have to be,
implemented in a UDI. Also the tutorial :ref:`py_tutorial_using_island` is a good starting point to understand the overall use. 

A UDI is a python class which, in its simplest form, contains the method ``run_evolve(self, algo, pop)`` which uses the :class:`~pygmo.algorithm` ``algo`` to evolve 
the :class:`~pygmo.population` ``pop``.

.. doctest::

    >>> import pygmo as pg
    >>> class my_isl:
    ...     def run_evolve(self, algo, pop):
    ...         return algo.evolve(pop)
    ...     def get_name(self):
    ...         return "It's my island!"

We have also included above the optional method ``get_name(self)`` that will be used by various ``__repr__(self)`` to provide humar readable information
on some pygmo classes. The above UDI can then be used to construct a :class:`~pygmo.island` (similarly to how UDP can be used to construct :class:`~pygmo.problem`, etc..).

.. doctest::

    >>> isl = pg.island(algo = pg.de(100), prob = pg.ackley(5), udi = my_isl(), size = 20)
    >>> print(isl) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Island name: It's my island!
        Status: idle
    <BLANKLINE>
    Algorithm: DE: Differential Evolution
    <BLANKLINE>
    Problem: Ackley Function
    <BLANKLINE>
    Population size: 20
        Champion decision vector: [...
        Champion fitness: [...

That was easy! Lets now understand what we actually did. The object ``isl`` now contains our UDI and, upon construction will open a thread and delegate the execution of
``run_evolve(self, algo, pop)`` to it upon call to :func:`~pygmo.island.evolve()`. 

But there is a catch. We are in python! So it is not possible, in general, to have the same interpreter execute instructions in parallel as,
in most of the popular python language implementations, memory management is not thread-safe. So, while the code above is perfectly fine and will
work with pygmo, a set of ``my_isl`` running evolutions will not run in parallel as each :class:`~pygmo.island`, when executing its :func:`~pygmo.island.evolve()` 
method, acquires the GIL (Global Interpreter Lock) and holds it during the :func:`~pygmo.island.evolve()` execution. 

As a consequence, the following code:

    >>> archi = pg.archipelago(n = 5, algo = pg.de(100), prob = pg.rosenbrock(10), pop_size = 20, udi = my_isl())
    >>> archi.evolve()

will not run evolution in parallel (only using different threads).

To code properly an UDI one need to code the ``def run_evolve(self, algo, pop)`` so that the GIL is released during the offload of the evolution task to a separate process.
An example on how this can be achieved using, for example the multiprocessing module of python. Let us have a look at some code snippets from the  :class:`~pygmo.mp_island`

.. doctest::

   >>> def _evolve_func(algo, pop): # doctest : +SKIP
   ...     return algo.evolve(pop)
   >>> class mp_island(object): # doctest : +SKIP
   ...     def __init__(self):
   ...         # Init the process pool, if necessary.
   ...         mp_island.init_pool()
   ...
   ...     def run_evolve(self, algo, pop):
   ...         with mp_island._pool_lock:
   ...             res = mp_island._pool.apply_async(_evolve_func, (algo, pop))
   ...         return res.get()

The full details are here not reported and can be read in the :class:`~pygmo.mp_island` code. In a nutshell, what happens is that the ``algo.evolve(pop)`` gets offloaded to
a process (in a shared pool inited upon construction calling the :func:`~pygmo.mp_island.init_pool()` static method). The instruction ``res.get()``, makes the thread where ``run_evolve``
remain waiting for the process execution and while doing so it releases the GIL, making parallelization effective. 

.. warning::
   When coding a UDI the user has to take care, according to the parallelization technology chosen, that the GIL is managed properly.