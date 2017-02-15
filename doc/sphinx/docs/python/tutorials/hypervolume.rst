.. _py_tutorial_hypervolume:

The hypervolume indicator
=========================

.. image:: ../../images/hypervolume.png
    :scale: 50 %
    :align: right

In multi-objective problems, the notion of solution is that of non-dominated fronts. The
quality of such fronts, i.e. the quality of a solution, can be measured by several indicators.

One of such measures is the hypervolume indicator, which is the hypervolume between a non-dominated front (P) and
a reference point (R). However, to rigorously calcuate the indicator can be time-consuming, hence efficiency
and approximate methods can be very important. 

In pygmo the main functionalities allowing to compute the hypervolume indicator and related quantities
are provided by the class :class:`~pygmo.core.hypervolume`. Instantiating this class from a 
:class:`~pygmo.core.population` or simply from a NumPy array will allow to compute the hypervolume indicator or
the exclusive contributions of single points using exact or approximated algorithms.