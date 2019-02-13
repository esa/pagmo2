Lennard Jones Cluster
======================================================

.. versionadded:: 2.11

.. image:: ../../images/lennard_jones.jpg

*#include <pagmo/problems/lennard_jones.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: lennard_jones

   This user-defined problem (UDP) is a box-constrained continuous single-objecive problem. 
   It represents the minimization of the energy of a cluster of atoms assuming a Lennard-Jones potential between each pair.
   The complexity for computing the objective function scales with the square of the number of atoms.

   The decision vector contains :math:`[z_2, y_3, z_3, x_4, y_4, z_4, ....]` as the cartesian coordinates :math:`x_1, y_1, z_1, x_2, y_2` and :math:`x_3`
   are fixed to zero.

   See: http://doye.chem.ox.ac.uk/jon/structures/LJ.html

This UDP is typically used to construct a :cpp:class:`~pagmo::problem`. 



