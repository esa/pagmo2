.. _py_tutorial_hypervolume_approx:

================================================================
Approximating the hypervolume
================================================================

Determining the hypervolume indicator is a computationally expensive task.
Even in case of a reasonably small dimension and low number of points (e.g. 100 points in 10 dimensions),
there are currently no known algorithms that can yield the results fast enough for use in 
most multiple-objective optimizers.

In this tutorial we will show a way to compute the hypervolume indicator faster, but at the cost of accuracy.
Two algorithms are included in pygmo and are capable of computing the hypervolume approximately:

#. `pygmo.bf_fpras()` - capable of approximating the hypervolume indicator
#. `pygmo.bf_approx()` - capable of approximating the least and the greatest contributor

.. note::
   :class:`~pygmo.core.population` object will never delegate the computation to any of the approximated algorithms.
   The only way to use the approximated algorithms is through the explicit request 
   (see the beginning of the tutorial :ref:`py_tutorial_hypervolume_advanced` for
   more information on how to do that).

FPRAS
================

The class `pygmo.bf_fpras()` found in PyGMO is a Fully Polynomial-Time Randomized Approximation Scheme accustomed
for the computation of the hypervolume indicator. You can invoke the FPRAS as follows:

.. doctest::

    >>> import pygmo as pg
    >>> prob = pg.problem(pg.dtlz(id = 3, fdim=10, dim=11))
    >>> pop = pg.population(prob, 100)
    >>> fpras = pg.bf_fpras(eps=0.1, delta=0.1)
    >>> hv = pg.hypervolume(pop)
    >>> offset = 5
    >>> ref_point = [max(pop.get_f(), key = lambda x: x[it])[it] + offset for it in [0,1,2,3]]
    >>> hv.compute(ref, hv_algo=fpras) # doctest: +SKIP

To influence the accuracy of the FPRAS, it is possible to provide the following keyword arguments to its constructor:

#. *eps* - relative accuracy of the approximation
#. *delta* - probability of error

For given parameters **eps=eps0** and **delta=delta0**, the obtained solution is (with probability **1 - delta0**)
within a factor of **1 +/- eps0** from the exact hypervolume.

.. note::
 The smaller the **eps** and **delta**, the longer it will take for the algorithm to evaluate.

By the *relative* error, we mean the scenario in which the approximation is accurate within given order of
magnitude, e.g. 312.32 and 313.41, are accurate within **eps = 0.1**, because they are accurate within two
orders of magnitude. At the same time, these are NOT accurate within **eps = 0.01**.

Running time
------------------

.. image:: ../../images/hv_fpras.png
    :align: center

The plots presents the measured running time (average and MAX out of 10) of FPRAS for varying ``Front size`` and ``Dimension``.
The algorithm is instantiated with **eps=0.1** and **delta=0.1**.
Notice the lack of any exponential increase in time as the dimension increases.

Since FPRAS scales so well with the dimension size, we can also present data for larger front sizes and number of objectives.

Now, that is quite a feat! A front of 1000 points in 100 dimensions would be beyond the reach of any of the algorithms
that rely on the exact computation.

Approximation of the least contributor
==========================================

Additionally to FPRAS, pygmo provides an approximated algorithm dedicated for the computation of the least/greatest contributor.
This is useful when we want to utilize evolutionary algorithms which rely on that feature, especially when the
problems has many objectives.

.. doctest::

  >>> # Problem with 30 objectives and 300 individuals
  >>> prob = pg.problem(pg.dtlz(id = 3, fdim=30, dim=35))
  >>> pop = pg.population(prob, size = 300)
  >>> hv_algo = pg.bf_approx(eps=0.1, delta=0.1)
  >>> hv = pg.hypervolume(pop)
  >>> offset = 5
  >>> ref_point = [max(pop.get_f(), key = lambda x: x[it])[it] + offset for it in [0,1,2,3]]
  >>> hv.least_contributor(ref_point, hv_algo=hv_algo) # doctest: +SKIP

.. note::
 Algorithm bf_approx provides only two features - computation of the least and the greatest contributor.
 Request for the computation of any other measure will raise and exception.