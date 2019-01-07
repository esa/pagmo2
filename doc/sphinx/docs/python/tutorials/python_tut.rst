.. _python_tutorials:

Python tutorials
================

.. note::

   Although pygmo is available both on Python 2.7 and Python 3, the tutorials
   (and pygmo's documentation in general) use the syntax of Python 3. Although many code snippets,
   examples, etc. will also run on Python 2.7 without modifications, in some cases they will have
   to be adapted in order to be run successfully on Python 2.7.

   A prominent example is the definition of new user-defined classes (such as optimisation problems, algorithms,
   etc.). Whereas, in Python 3, when defining of a user-defined problem (UDP) we can simply write

   .. code::

      class my_problem:
          ...

   in Python 2.7 we must write instead

   .. code::

      class my_problem(object):
          ...

   (that is, the UDP must derive from :class:`object`). Failure to do so will result in runtime errors when using
   the API of :class:`~pygmo.problem`. In this specific case, the problem lies in the dichotomy between
   *old* and *new* style classes in Python 2.x (see `here <https://wiki.python.org/moin/NewClassVsClassicClass>`__
   for an in-depth explanation). pygmo will assume that, like in Python 3, all classes are new-style classes.

   Our general recommendation is, if possible, to use Python 3 (or, at least, to use Python 2 styles and idioms
   that are forward-compatible with Python 3).

Basics
^^^^^^

.. toctree::
  :maxdepth: 2

  using_problem
  using_algorithm
  using_population
  using_island
  using_archipelago
  evolving_a_population

Coding your own problem (UDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 2

  coding_udp_simple
  coding_udp_constrained
  coding_udp_minlp
  coding_udp_multi_objective

Meta-problems
^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 2

  udp_meta_decorator

Coding your own island (UDI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 2

  coding_udi

Local optimization
^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 2

  nlopt_basics

Multi-objective optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  moo
  moo_moead

Hypervolumes
^^^^^^^^^^^^

.. toctree::
  :maxdepth: 1

  hypervolume
  hypervolume_advanced
  hypervolume_approx

Advanced Examples
^^^^^^^^^^^^^^^^^

.. toctree::
  :maxdepth: 2

  cec2013_comp
  solving_schwefel_20
  cmaes_vs_xnes
