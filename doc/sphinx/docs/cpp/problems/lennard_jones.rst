Lennard Jones Cluster (box-bounded)
======================================================

.. versionadded:: 2.10

.. image:: ../../images/lennard_jones.jpg

*#include <pagmo/problems/lennard_jones.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: lennard_jones

   This user-defined problem (UDP) is a box-constrained continuous single-objecive problem. 
   It represents the minimization of the energy of a cluster of atoms assuming a Lennard-Jones potential between each pair.
   The complexity for computing the objective function scales with the square of the number of atoms.

   The decision vector contains [z2, y3, z3, x4, y4, z4, ....] as the cartesian coordinates x1, y1, z1, x2, y2 and x3
   are fixed to zero.

   See: http://doye.chem.ox.ac.uk/jon/structures/LJ.html

