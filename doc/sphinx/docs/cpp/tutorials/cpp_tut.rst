.. _tutorial:

Introduction
============
pagmo is a C++ scientific library to *perform parallel optimizations*. 
pagmo is **not** a framework for forecasting, clustering, predictions etc. 

Applicable optimization problems arise in most quantitative disciplines, such as:

- computer science (e.g. mapping and scheduling for hard-real-time systems, scheduling in networking)
- engineering (e.g. )
- finance (e.g. asset pricing, portfolio optimization)
- economics (e.g. utility maximization problem, expenditure minimization problem)
- operations research (e.g. Cutting stock problem, floorplanning, traveling salesman problem, staffing, job shop problem)
- physics (e.g. )


Some problems are applicable to several of these domains, for example the more general bin packing problem is 
applicable to computer science (scheduling in real-time systems and networks) and operations research 
(staffing, job shop problems).
The general aim in all these fields is to minimize or maximize a given function, i.e. we need to find the inputs to a 
function that result in the best output of the function. 

Ideally one always wants to find the optimal solution for a given problem. However, this is often not feasible as 
the time required to solve optimization problems can grow exponentially with the size of the problem. 
In order to tackle these challenges we use heuristics and metaheuristics that find approximate solutions. 
Additionally, by using parallelism we can solve problems even faster or find better solutions.

This is where pagmo comes in. It provides you with among others a collection of solvers and a framework to 
solve optimization problems in parallel. This way you can for example test a set of solvers on your specific problem. 
An overview of solvers and other capabilities can be found :ref:`here <capabilities>`.

In this tutorial we explain the basics of pagmo and detail more advanced usecases. 

C++ tutorial
============

.. toctree::
  :maxdepth: 1

  cpp_tut_preliminaries


First Problem
^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  cpp_tut_first_problem
  cpp_tut_first_problem_solving
  cpp_tut_first_problem_parallel
  cpp_tut_next_steps


Additional Problems
^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  cpp_tut_bin_packing

Advanced Topics
^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  cpp_tut_implement_optimizer