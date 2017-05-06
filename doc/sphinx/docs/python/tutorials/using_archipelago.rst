.. _py_tutorial_using_archipelago:

Use of the class :class:`~pygmo.archipelago`
===============================================

The :class:`~pygmo.archipelago` class is the main parallelization engine of pagmo. It essentially is
a container of :class:`~pygmo.island` able to initiate evolution (optimization tasks) in each :class:`~pygmo.island`
asynchronously while keeping track of the results and of the information exchange (migration) between the tasks (via the
generalized island model). The various :class:`~pygmo.island` in an :class:`~pygmo.archipelago` can be heterogeneous
and refer to different UDAs, UDPs and UDIs.

In this tutorial we will show how to use an :class:`~pygmo.archipelago` to run parallel evolutions of some
optimization task defined in a UDP. So let us start defining our UDP. We will here define a single objective
problem with one equality and one inequality constraint. The problem in itself is not important for the
purpose of this tutorial.

Mathematically:

.. math::
   \begin{array}{ll}
     \mbox{minimize: } & \sum_i x_i \\
     \mbox{subject to:} & -1 \le x_i \le 1, i = 1..n
                        & \sum_i x_i^2 = 1
                        % \sum_i x_1 \ge 0
   \end{array}

.. doctest::

    >>> class toy_problem:
    ...     def __init__(self, dim):
    ...         self.dim = dim
    ...
    ...     def fitness(self, x):
    ...         return [sum(x), 1 - sum(x*x), - sum(x)]
    ...
    ...     def gradient(self, x):
    ...         return pg.estimate_gradient(lambda x: self.fitness(x), x) # numerical gradient
    ...
    ...     def get_nec(self):
    ...         return 1
    ...
    ...     def get_nic(self):
    ...         return 1
    ...
    ...     def get_bounds(self):
    ...         return ([-1] * self.dim, [1] * self.dim)
    ...
    ...     def get_name(self):
    ...         return "A toy problem"
    ...
    ...     def get_extra_info(self):
    ...         return "\tDimensions: " + str(self.dim)

