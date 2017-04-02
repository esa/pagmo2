.. PaGMO documentation master file, created by
   sphinx-quickstart on Mon Apr 18 12:49:02 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

.. image:: ../sphinx/docs/images/pagmo_logo.png
   :height: 300px
   :alt: pagmo2.0 logo
   :align: right

Pagmo (C++) or pygmo (Python) is a scientific library for massively parallel optimization. It is built
around the idea of providing a unified interface to optimization algorithms and to optimization problems and to
make their deployment in massively parallel environments easy.

Efficient implementantions of bio-inspired and evolutionary algorithms are sided to state-of the art optimization algorithms
(Simplex Methods, SQP methods,  interior points methods ....) and can be easily mixed (also with your newly invented algorithms)
to build a super-algorithm exploiting algoritmic cooperation via the asynchronous, generalized island model.

Pagmo and pygmo can be used to solve constrained, unconstrained, single objective, multiple objective, continuous and int optimization
problems, as well as to perform research on novel algorithms and paradigms and easily compare them to state of the
art implementations of established ones.


Contents:

.. toctree::
   :maxdepth: 1

   docs/cpp/cpp_docs
   docs/python/python_docs
   docs/python/tutorials/python_tut
