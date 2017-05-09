.. _py_tutorial_using_island:

Use of the class :class:`~pygmo.island`
===============================================

.. image:: ../../images/island_no_text.png

The :class:`~pygmo.island` class is the unit parallelization block of pagmo. The idea is that, in pygmo, an :class:`~pygmo.island` is a computational
unit that can be physically located anywhere and that can thus be used to outsource calls to :func:`~pygmo.algorithm.evolve()`
unloading the main computational unit from the task. As such it is able to run evolutions ona separate thread, process or remote machine,
according to the implementation details of the UDI (user Defined Island) whose type it erases. Similarly to how an :class:`~pygmo.algorithm`
evolves a :class:`~pygmo.population` according to the UDA, and a problem computes its fitness according to the UDP.

The non-advanced user does not have to implement his own UDI as we provide the most popular parallel task execution paradigms already
coded in UDIs provided with pygmo, a list of which can be found at :ref:`islands`.

.. note::
   A collection of :class:`pygmo.island` form an :class:`~pygmo.archipelago`, you can skip this tutorial and follow directly the tutorial ":ref:`py_tutorial_using_archipelago`"
   in case you are happy with the defualt choices pygmo will do for you to parallelize your tasks via the :class:`~pygmo.archipelago`.

WeLet us start by instantiating an island.

.. doctest::

    >>> import pygmo as pg
    >>> isl = pg.island(algo = pg.de(10), prob = pg.ackley(5), size=20, udi=pg.thread_island())

Regardless of the *udi* kwarg, should the construction be successfull, a new thread will be opened and delegated
to run the :func:`~pygmo.island.evolve()` method. Since, in this case, we constructed the island specifying an 
*udi*, and in particular we choose the :class:`~pygmo.thread_island`, the population evolution will happen directly
on the opened thread. 

.. note::
   Since an evolution (optimization task) requires both evaluations of the :class:`~pygmo.problem` fitness function and
   the execution of the :class:`~pygmo.algorithm` logic, both these must be marked as thread safe in order for a 
   :class:`~pygmo.thread_island` to be successfull.