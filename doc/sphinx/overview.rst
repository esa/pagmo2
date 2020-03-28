Capabilities
============

Overview
--------

* Support for a wide array of types of
  optimisation problems (continuous, integer, single
  and multi-objective, constrained and unconstrained,
  with or without derivatives, stochastic, etc.).
* A comprehensive library of algorithms,
  including global and local solvers, meta-heuristics,
  single and multi-objective algorithms,
  wrappers for third-party solvers (e.g.,
  `NLopt <https://nlopt.readthedocs.io/en/latest/>`__,
  `Ipopt <https://projects.coin-or.org/Ipopt>`__, etc.).
* Comprehensive support for coarse-grained
  parallelisation via the
  `generalised island model <https://link.springer.com/chapter/10.1007/978-3-642-28789-3_7>`__.
  In the island model, multiple optimisation instances
  run in parallel (possibly on different machines) and
  exchange information as the optimisation proceeds,
  improving the overall time-to-solution and allowing
  to harness the computational power of modern computer
  architectures (including massively-parallel
  high-performance clusters).
* Support for fine-grained parallelisation
  (i.e., at the level of single objective function
  evaluations) in selected algorithms via the batch
  fitness evaluation framework. This allows to
  speed-up single optimisations via parallel
  processing (e.g., multithreading, high-performance
  clusters, GPUs, SIMD vectorization, etc.).
* A library of ready-to-use optimisation problems
  for algorithmic testing and performance evaluation
  (Rosenbrock, Rastrigin, Lennard-Jones, etc.).
* A library of optimisation-oriented utilities
  (e.g., hypervolume computation, non-dominated
  sorting, plotting, etc.).

List of algorithms
------------------

This is the list of user defined algorithms (UDAs) currently
provided with pagmo. These are classes that
can be used to construct a :cpp:class:`pagmo::algorithm`, which will
then provide a unified interface to access the algorithm's functionalities.

Generally speaking, algorithms can solve only specific problem classes.
In the tables below, we use the following
flags to signal which problem types an algorithm can solve:

* S = Single-objective
* M = Multi-objective
* C = Constrained
* U = Unconstrained
* I = Integer programming
* sto = Stochastic

Note that algorithms that do not directly support integer programming
will still work on integer problems
(i.e., they will optimise the relaxed problem). Note also that it is possible
to use :ref:`meta-problems <available_meta_problems>`
to turn constrained problems into unconstrained ones, and multi-objective
problems into single-objective ones.

Heuristic Global Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
========================================================== ========================================= =========================
Common Name                                                Docs of the C++ class                     Capabilities
========================================================== ========================================= =========================
Extended Ant Colony Optimization (GACO)                    :cpp:class:`pagmo::gaco`                  S-CU-I
Differential Evolution (DE)                                :cpp:class:`pagmo::de`                    S-U
Self-adaptive DE (jDE and iDE)                             :cpp:class:`pagmo::sade`                  S-U
Self-adaptive DE (de_1220 aka pDE)                         :cpp:class:`pagmo::de1220`                S-U
Grey wolf optimizer (GWO)                                  :cpp:class:`pagmo::gwo`                   S-U
Improved Harmony Search                                    :cpp:class:`pagmo::ihs`                   SM-CU-I
Particle Swarm Optimization (PSO)                          :cpp:class:`pagmo::pso`                   S-U
Particle Swarm Optimization Generational (GPSO)            :cpp:class:`pagmo::pso_gen`               S-U-sto
(N+1)-ES Simple Evolutionary Algorithm                     :cpp:class:`pagmo::sea`                   S-U-sto
Simple Genetic Algorithm                                   :cpp:class:`pagmo::sga`                   S-U-I-sto
Corana's Simulated Annealing (SA)                          :cpp:class:`pagmo::simulated_annealing`   S-U
Artificial Bee Colony (ABC)                                :cpp:class:`pagmo::bee_colony`            S-U
Covariance Matrix Adaptation Evo. Strategy (CMA-ES)        :cpp:class:`pagmo::cmaes`                 S-U-sto
Exponential Evolution Strategies (xNES)                    :cpp:class:`pagmo::xnes`                  S-U-sto
Non-dominated Sorting GA (NSGA2)                           :cpp:class:`pagmo::nsga2`                 M-U-I
Multi-objective EA vith Decomposition (MOEA/D)             :cpp:class:`pagmo::moead`                 M-U
Multi-objective Hypervolume-based ACO (MHACO)              :cpp:class:`pagmo::maco`                  M-U-I
Non-dominated Sorting PSO (NSPSO)                          :cpp:class:`pagmo::nspso`                 M-U
========================================================== ========================================= =========================

Local optimization 
^^^^^^^^^^^^^^^^^^
====================================================== ============================================================================================= ===============
Common Name                                            Docs of the C++ class                                                                         Capabilities
====================================================== ============================================================================================= ===============
Compass Search (CS)                                    :cpp:class:`pagmo::compass_search`                                                            S-CU
COBYLA (from NLopt)                                    :cpp:class:`pagmo::nlopt`                                                                     S-CU
BOBYQA (from NLopt)                                    :cpp:class:`pagmo::nlopt`                                                                     S-U
NEWUOA + bound constraints (from NLopt)                :cpp:class:`pagmo::nlopt`                                                                     S-U
PRAXIS (from NLopt)                                    :cpp:class:`pagmo::nlopt`                                                                     S-U
Nelder-Mead simplex (from NLopt)                       :cpp:class:`pagmo::nlopt`                                                                     S-U
Subplex (from NLopt)                                   :cpp:class:`pagmo::nlopt`                                                                     S-U
MMA (Method of Moving Asymptotes) (from NLopt)         :cpp:class:`pagmo::nlopt`                                                                     S-CU
CCSA (from NLopt)                                      :cpp:class:`pagmo::nlopt`                                                                     S-CU
SLSQP (from NLopt)                                     :cpp:class:`pagmo::nlopt`                                                                     S-CU
Low-storage BFGS (from NLopt)                          :cpp:class:`pagmo::nlopt`                                                                     S-U
Preconditioned truncated Newton (from NLopt)           :cpp:class:`pagmo::nlopt`                                                                     S-U
Shifted limited-memory variable-metric (from NLopt)    :cpp:class:`pagmo::nlopt`                                                                     S-U
Ipopt                                                  :cpp:class:`pagmo::ipopt`                                                                     S-CU
SNOPT (in pagmo_plugins_non_free affiliated package)   `pagmo::snopt7 <https://esa.github.io/pagmo_plugins_nonfree/cpp_snopt7.html>`__               S-CU
WORHP (in pagmo_plugins_non_free affiliated package)   `pagmo::wohrp <https://esa.github.io/pagmo_plugins_nonfree/cpp_worhp.html>`__                 S-CU
====================================================== ============================================================================================= ===============

Meta-algorithms
^^^^^^^^^^^^^^^

====================================================== ============================================ ==========================
Common Name                                            Docs of the C++ class                        Capabilities [#meta_capa]_
====================================================== ============================================ ==========================
Monotonic Basin Hopping (MBH)                          :cpp:class:`pagmo::mbh`                      S-CU
Cstrs Self-Adaptive                                    :cpp:class:`pagmo::cstrs_self_adaptive`      S-C
Augmented Lagrangian algorithm (from NLopt) [#auglag]_ :cpp:class:`pagmo::nlopt`                    S-CU
====================================================== ============================================ ==========================

.. rubric:: Footnotes

.. [#meta_capa] The capabilities of the meta-algorithms depend also on the capabilities of the algorithms they wrap. If, for instance,
   a meta-algorithm supporting constrained problems is constructed from an algorithm which does *not* support constrained problems, the
   resulting meta-algorithms will *not* be able to solve constrained problems.

.. [#auglag] The Augmented Lagrangian algorithm can be used only in conjunction with other NLopt algorithms.

List of problems
----------------

This is the list of user defined problems (UDPs) currently provided with pagmo.
These are classes that can be used to construct a :cpp:class:`pagmo::problem`,
which will
then provide a unified interface to access the problem's functionalities.

In the tables below, we classify optimisation problems
according to the following flags:

* S = Single-objective
* M = Multi-objective
* C = Constrained
* U = Unconstrained
* I = Integer programming
* sto = Stochastic

Scalable problems
^^^^^^^^^^^^^^^^^
========================================================== ========================================= ===============
Common Name                                                Docs of the C++ class                     Type
========================================================== ========================================= ===============
Ackley                                                     :cpp:class:`pagmo::ackley`                S-U
Golomb Ruler                                               :cpp:class:`pagmo::golomb_ruler`          S-C-I
Griewank                                                   :cpp:class:`pagmo::griewank`              S-U
Hock Schittkowsky 71                                       :cpp:class:`pagmo::hock_schittkowsky_71`  S-C
Inventory                                                  :cpp:class:`pagmo::inventory`             S-U-sto
Lennard Jones                                              :cpp:class:`pagmo::lennard_jones`         S-U
Luksan Vlcek 1                                             :cpp:class:`pagmo::luksan_vlcek1`         S-C
Rastrigin                                                  :cpp:class:`pagmo::rastrigin`             S-U
MINLP Rastrigin                                            :cpp:class:`pagmo::minlp_rastrigin`       S-U-I
Rosenbrock                                                 :cpp:class:`pagmo::rosenbrock`            S-U
Schwefel                                                   :cpp:class:`pagmo::schwefel`              S-U
========================================================== ========================================= ===============

Problem suites 
^^^^^^^^^^^^^^^
================================== ============================================ ===============
Common Name                        Docs of the C++ class                        Type
================================== ============================================ ===============
CEC2006                            :cpp:class:`pagmo::cec2006`                  S-C
CEC2009                            :cpp:class:`pagmo::cec2009`                  S-C
CEC2013                            :cpp:class:`pagmo::cec2013`                  S-U
CEC2014                            :cpp:class:`pagmo::cec2014`                  S-U
ZDT                                :cpp:class:`pagmo::zdt`                      M-U
DTLZ                               :cpp:class:`pagmo::dtlz`                     M-U
WFG                                :cpp:class:`pagmo::wfg`                      M-U
================================== ============================================ =============== 

.. _available_meta_problems:

Meta-problems
^^^^^^^^^^^^^

Meta-problems are UDPs that take another UDP as input, yielding a new UDP which modifies the behaviour and/or the properties of the original
problem in a variety of ways.

========================================================== =========================================
Common Name                                                Docs of the C++ class                    
========================================================== =========================================
Decompose                                                  :cpp:class:`pagmo::decompose`            
Translate                                                  :cpp:class:`pagmo::translate`            
Unconstrain                                                :cpp:class:`pagmo::unconstrain`          
========================================================== =========================================


List of islands
---------------

This is the list of user defined islands (UDIs)
currently provided with pagmo. These are classes that
can be used to construct a :cpp:class:`pagmo::island`,
which will then
provide a unified interface to access the island's functionalities.

In the pagmo jargon, an island is an entity tasked with
managing the asynchronous evolution of a population via
an algorithm in the generalised island model.
Different UDIs enable different parallelisation
strategies (e.g., multithreading, multiprocessing,
cluster architectures, etc.).

========================================================== =========================================
Common Name                                                Docs of the C++ class                    
========================================================== =========================================
Thread island                                              :cpp:class:`pagmo::thread_island`        
Fork island                                                :cpp:class:`pagmo::fork_island`          
========================================================== =========================================

List of batch fitness evaluators
--------------------------------

This is the list of user defined batch fitness
evaluators (UDBFEs)
currently provided with pagmo. These are classes that
can be used to construct a :cpp:class:`pagmo::bfe`,
which will then
provide a unified interface to access the evaluator's
functionalities.

In the pagmo jargon, a batch fitness evaluator
implements the capability of evaluating a group
of decision vectors in a parallel and/or vectorised
fashion. Batch fitness evaluators are used to implement
fine-grained parallelisation in pagmo (e.g., parallel
initialisation of populations, or parallel
fitness evaluations within the inner loop of an algorithm).

========================================================== =========================================
Common Name                                                Docs of the C++ class                 
========================================================== =========================================
Default BFE                                                :cpp:class:`pagmo::default_bfe`
Thread BFE                                                 :cpp:class:`pagmo::thread_bfe`
========================================================== =========================================
