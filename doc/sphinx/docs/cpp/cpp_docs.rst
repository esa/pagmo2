C++ Documentation
=================
The full, stand-alone, detailed, documentation of the c++ code.

Core classes
^^^^^^^^^^^^
These are the core PaGMO classes. In order to learn how to use them
we suggest to follow the tutorials / examples.

.. toctree::
  :maxdepth: 1

  types
  problem
  algorithm
  population
  island
  archipelago
  bfe
  topology
  r_policy
  s_policy

Implemented algorithms
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  algorithms/null
  algorithms/bee_colony
  algorithms/cmaes
  algorithms/compass_search
  algorithms/de
  algorithms/de1220
  algorithms/gaco
  algorithms/gwo
  algorithms/ihs
  algorithms/ipopt
  algorithms/moead
  algorithms/mbh
  algorithms/cstrs_self_adaptive
  algorithms/nlopt
  algorithms/nsga2
  algorithms/pso
  algorithms/pso_gen
  algorithms/sade
  algorithms/sea
  algorithms/sga
  algorithms/simulated_annealing
  algorithms/xnes


Implemented problems
^^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  problems/null
  problems/rosenbrock
  problems/rastrigin
  problems/schwefel
  problems/ackley
  problems/golomb_ruler
  problems/griewank
  problems/lennard_jones
  problems/zdt
  problems/dtlz
  problems/hock_schittkowsky_71
  problems/inventory
  problems/luksan_vlcek1
  problems/minlp_rastrigin
  problems/translate
  problems/decompose
  problems/cec2006
  problems/cec2009
  problems/cec2013
  problems/cec2014
  problems/unconstrain
  problems/wfg

Implemented islands
^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  islands/thread_island
  islands/fork_island

Implemented batch evaluators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  batch_evaluators/default_bfe
  batch_evaluators/thread_bfe
  batch_evaluators/member_bfe

Implemented topologies
^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
  :maxdepth: 1

  topologies/unconnected
  topologies/fully_connected
  topologies/base_bgl_topology
  topologies/ring

Implemented replacement policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
  :maxdepth: 1

  r_policies/fair_replace

Implemented selection policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
  :maxdepth: 1

  s_policies/select_best

Utilities
^^^^^^^^^
Various optimization utilities.

.. toctree::
  :maxdepth: 1

  utils/multi_objective
  utils/constrained
  utils/discrepancy
  utils/hypervolume
  utils/gradient_and_hessians

Miscellanea
^^^^^^^^^^^
Various coding utilities.

.. toctree::
  :maxdepth: 1

  miscellanea/generic
  miscellanea/type_traits
  miscellanea/exceptions
  miscellanea/utility_classes
