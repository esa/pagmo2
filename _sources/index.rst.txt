.. PaGMO documentation master file, created by
   sphinx-quickstart on Mon Apr 18 12:49:02 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pagmo & Pygmo
=============

.. image:: ../sphinx/docs/images/prob.png
   :target: docs/problem_list.html
   :width: 15%

.. image:: ../sphinx/docs/images/algo.png
   :target: docs/algorithm_list.html
   :width: 15%

.. image:: ../sphinx/docs/images/pop.png
   :width: 15%

.. image:: ../sphinx/docs/images/island.png
   :target: docs/island_list.html
   :width: 15%

.. image:: ../sphinx/docs/images/archi.png
   :width: 15%

.. image:: ../sphinx/docs/images/migration.png
   :width: 15%


Pagmo (C++) or pygmo (Python) is a scientific library for massively parallel optimization. It is built
around the idea of providing a unified interface to optimization algorithms and to optimization problems and to
make their deployment in massively parallel environments easy.

Efficient implementantions of bio-inspired and evolutionary algorithms are sided to state-of the art optimization algorithms
(Simplex Methods, SQP methods,  interior points methods ....) and can be easily mixed (also with your newly invented algorithms)
to build a super-algorithm exploiting algoritmic cooperation via the asynchronous, generalized island model.

Pagmo and pygmo can be used to solve constrained, unconstrained, single objective, multiple objective, continuous and integer optimization
problems, stochastic and deterministic problems, as well as to perform research on novel algorithms and paradigms and easily compare them to state of the
art implementations of established ones.

If you are using pagmo/pygmo as part of your research, teaching, or other activities, we would be grateful if you could star
the repository and/or cite our work. The DOI of the latest version and other citation resources are available
at `this link <https://doi.org/10.5281/zenodo.1045336>`__.

Contents:

.. toctree::
   :maxdepth: 1

   install
   quickstart

.. toctree::
   :maxdepth: 1

   docs/algorithm_list
   docs/problem_list
   docs/island_list

.. toctree::
   :maxdepth: 1

   docs/cpp/cpp_docs
   docs/python/python_docs
   docs/python/tutorials/python_tut

.. toctree::
   :maxdepth: 1
   
   credits
   changelog
