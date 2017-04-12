.. _py_tutorial_nlopt_basics:

A first tutorial on the use of NLopt solvers
--------------------------------------------

In this tutorial we show the basic usage pattern of :class:`pygmo.nlopt`. This user defined
algorithm (UDA) wraps the NLopt library making it easily accessible via the pygmo common
:class:`pygmo.algorithm` interface. Let see how this miracle occur.

I have the gradient
^^^^^^^^^^^^^^^^^^^

.. doctest::
   
    >>> import pygmo as pg
    >>> uda = pg.nlopt("slsqp")
    >>> algo = pg.algorithm(uda)
    >>> print(algo) # doctest: +NORMALIZE_WHITESPACE
    Algorithm name: NLopt - slsqp [deterministic]
        Thread safety: basic
    <BLANKLINE>
    Extra info:
        NLopt version: 2.4.2
        Last optimisation return code: NLOPT_SUCCESS (value = 1, Generic success return value)
        Verbosity: 0
        Individual selection policy: best
        Individual replacement policy: best
        Stopping criteria:
            stopval:  disabled
            ftol_rel: disabled
            ftol_abs: disabled
            xtol_rel: 1e-08
            xtol_abs: disabled
            maxeval:  disabled
            maxtime:  disabled
    <BLANKLINE>


In a few lines we have constructed a :class:`pygmo.algorithm` containing the slsqp solver from
NLopt. For a list of solvers availbale via the NLopt library check the docs of :class:`~pygmo.nlopt`.
In this tutorial we will make use of slsqp, a Sequential Quadratic Programming algorithm suited for 
generic Non Linear Programming problems (i.e. non linearly constrained single objective problems).

All the stopping criteria used by the NLopt library are available via dedicated attributes, so that we may, for
example, set the ``ftol_rel`` by writing:

.. doctest::
   
    >>> algo.extract(pg.nlopt).ftol_rel = 1e-8

Let us algo activate some verbosity as to store a log and have a screen output:

.. doctest::
   
    >>> algo.set_verbosity(1)

We now need a problem to solve. Let us start with one of the UDPs provided in pygmo. The
:class:`pygmo.luksan_vlcek1` provides a classic, scalable, equally constrained problem. It 
also has its gradient implemented so that we do not have to worry about that for the moment.

.. doctest::
   
    >>> udp = pg.luksan_vlcek1(dim = 20)
    >>> prob = pg.problem(udp)
    >>> pop = pg.population(prob, size = 1)
    >>> pop.problem.c_tol = [1e-8] * prob.get_nc()

The lines above can be shortened and are equivalent to:

.. doctest::
   
    >>> pop = pg.population(pg.luksan_vlcek1(dim = 20), size = 1)
    >>> pop.problem.c_tol = [1e-8] * pop.problem.get_nc()

.. image:: ../../images/nlopt_basic_lv1.png
   :scale: 80 %
   :alt: slsqp performance
   :align: right

We now solve this problem by writing:

.. doctest::
   
   >>> pop = algo.evolve(pop) # doctest: +SKIP
   fevals:       fitness:      violated:    viol. norm:
         1         250153             18        2619.51 i
         2         132280             18        931.767 i
         3        26355.2             18        357.548 i
         4          14509             18        140.055 i
         5          77119             18        378.603 i
         6        9104.25             18         116.19 i
         7        9104.25             18         116.19 i
         8        2219.94             18        42.8747 i
         9        947.637             18        16.7015 i
        10        423.519             18        7.73746 i
        11        82.8658             18        1.39111 i
        12        34.2733             15       0.227267 i
        13        11.9797             11      0.0309227 i
        14        42.7256              7        0.27342 i
        15        1.66949             11       0.042859 i
        16        1.66949             11       0.042859 i
        17       0.171358              7     0.00425765 i
        18     0.00186583              5    0.000560166 i
        19    1.89265e-06              3    4.14711e-06 i
        20    1.28773e-09              0              0
        21    7.45125e-14              0              0
        22    3.61388e-18              0              0
        23    1.16145e-23              0              0
   <BLANKLINE>
   Optimisation return status: NLOPT_XTOL_REACHED (value = 4, Optimization stopped because xtol_rel or xtol_abs was reached)

As usual we can access the values in the log to analyze the algorithm performance and, for example, produce a plot such as that
shown here.

.. doctest::

   >>> log = algo.extract(pg.nlopt).get_log()
   >>> from matplotlib import pyplot as plt # doctest: +SKIP
   >>> plt.semilogy([line[0] for line in log], [line[1] for line in log], label = "obj") # doctest: +SKIP
   >>> plt.semilogy([line[0] for line in log], [line[3] for line in log], label = "con") # doctest: +SKIP
   >>> plt.xlabel("fevals") # doctest: +SKIP
   >>> plt.ylabel("value") # doctest: +SKIP
   >>> plt.show() # doctest: +SKIP

I do not have the gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^