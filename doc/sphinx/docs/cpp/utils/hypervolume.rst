.. _cpp_hypervolume_utils:

Hypervolumes
=============

.. image:: ../../../../doxygen/images/hypervolume.png
    :scale: 80 %
    :alt: hypervolume
    :align: right

The computation of hypervolumes plays an important role in solving multi-objective optimization
problems. In pagmo we include a number of different methods and algorithms that allow
hypervolumes to be computed efficiently at all dimensions. More information on the details
of the algorithms implemented and their performance can be found in the publication:

:Title: "Empirical performance of the approximation of the least hypervolume contributor."
:Authors: Krzysztof Nowak, Marcus Märtens, and Dario Izzo.
:Published in: International Conference on Parallel Problem Solving from Nature. Springer International Publishing, 2014.

--------------------------------------------------------------------------

.. doxygenclass:: pagmo::hypervolume
   :members:
