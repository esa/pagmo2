.. _py_algorithms:

List of algorithms available in pygmo
=====================================

A Quick Look
------------

User defined algorithms (UDAs) are classes that allow the construction of a :class:`pygmo.algorithm` offering a unified access to
all of them. The user can implement his own UDA, or use any ot the ones we provide in pygmo and that are listed here for convenience.

Each algorithm can be associated only to problems of certain types S = Single, M = Multi-objective, C = Constrained, U = Unconstrained, sto = stochastic

Heuristic Global Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
========================================================== ========================================= =============== ===================================================================
Common Name                                                Name in PyGMO                             Type            Comments
========================================================== ========================================= =============== ===================================================================
The null algorithm                                         :class:`pygmo.null_algorithm`             SM-CU           Exposed from C++, used for initialization purposes, does nothing
Differential Evolution (DE)                                :class:`pygmo.de`                         S-U             Exposed from C++
Self-adaptive DE (jDE and iDE)                             :class:`pygmo.sade`                       S-U             Exposed from C++
Self-adaptive DE (de_1220 aka pDE)                         :class:`pygmo.de1220`                     S-U             Exposed from C++
Particle Swarm Optimization (PSO)                          :class:`pygmo.pso`                        S-U             Exposed from C++
(N+1)-ES Simple Evolutionary Algorithm                     :class:`pygmo.sea`                        S-U (sto)       Exposed from C++
Simple Genetic Algorithm                                   :class:`pygmo.sga`                        S-U (sto)       Exposed from C++
Corana's Simulated Annealing (SA)                          :class:`pygmo.simulated_annealing`        S-U             Exposed from C++
Artificial Bee Colony (ABC)                                :class:`pygmo.bee_colony`                 S-U             Exposed from C++
Covariance Matrix Adaptation Evo. Strategy (CMA-ES)        :class:`pygmo.cmaes`                      S-U             Exposed from C++
Non-dominated Sorting GA (NSGA2)                           :class:`pygmo.nsga2`                      M-U             Exposed from C++
Multi-objective EA vith Decomposition (MOEA/D)             :class:`pygmo.moead`                      M-U             Exposed from C++
Improved Harmony Serach                                    :class:`pygmo.ihs`                        SM-CU-I         Exposed from C++
Exponential Evolution Strategies (xNES)                    :class:`pygmo.xnes`                       S-U (sto)       Exposed from C++
========================================================== ========================================= =============== ===================================================================

Meta-algorithms 
^^^^^^^^^^^^^^^
================================== ============================================ =============== ===========================================
Common Name                        Name in PyGMO                                Type            Comments
================================== ============================================ =============== ===========================================
Monotonic Basin Hopping (MBH)      :class:`pygmo.mbh`                           S-CU            Exposed from C++
Cstrs Self-Adaptive                :class:`pygmo.cstrs_self_adaptive`           S-C             Exposed from C++
Augmented Lagrangian algorithm     :class:`pygmo.nlopt`                         S-CU            Exposed from C++
================================== ============================================ =============== ===========================================

Local optimization 
^^^^^^^^^^^^^^^^^^
====================================================== ========================================= =============== =====================================================================
Common Name                                             Name in PyGMO                             Type           Comments
====================================================== ========================================= =============== =====================================================================
Compass Search (CS)                                    :class:`pygmo.compass_search`             S-CU            Exposed from C++
COBYLA (from NLopt)                                    :class:`pygmo.nlopt`                      S-CU            Exposed from C++
BOBYQA (from NLopt)                                    :class:`pygmo.nlopt`                      S-U             Exposed from C++
NEWUOA + bound constraints (from NLopt)                :class:`pygmo.nlopt`                      S-U             Exposed from C++
PRAXIS (from NLopt)                                    :class:`pygmo.nlopt`                      S-U             Exposed from C++
Nelder-Mead simplex (from NLopt)                       :class:`pygmo.nlopt`                      S-U             Exposed from C++
sbplx (from NLopt)                                     :class:`pygmo.nlopt`                      S-U             Exposed from C++
MMA (Method of Moving Asymptotes) (from NLopt)         :class:`pygmo.nlopt`                      S-CU            Exposed from C++
CCSA (from NLopt)                                      :class:`pygmo.nlopt`                      S-CU            Exposed from C++
SLSQP (from NLopt)                                     :class:`pygmo.nlopt`                      S-CU            Exposed from C++
low-storage BFGS (from NLopt)                          :class:`pygmo.nlopt`                      S-U             Exposed from C++
preconditioned truncated Newton (from NLopt)           :class:`pygmo.nlopt`                      S-U             Exposed from C++
Shifted limited-memory variable-metric (from NLopt)    :class:`pygmo.nlopt`                      S-U             Exposed from C++
Ipopt                                                  :class:`pygmo.ipopt`                      S-CU            Exposed from C++
====================================================== ========================================= =============== =====================================================================

----------------------------------------------------------------------------------------------------------------------


Algorithms exposed from C++
---------------------------

.. autoclass:: pygmo.null_algorithm
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.bee_colony
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.de
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.sea
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.sga
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.sade
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.de1220
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.cmaes
  :members:

-------------------------------------------------------------

.. autoclass:: pygmo.moead
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.compass_search
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.simulated_annealing
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.pso
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.nsga2
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.mbh
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.cstrs_self_adaptive
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.nlopt
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.ipopt
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.ihs
   :members:

-------------------------------------------------------------

.. autoclass:: pygmo.xnes
   :members:
