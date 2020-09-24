.. PaGMO documentation master file, created by
   sphinx-quickstart on Mon Apr 18 12:49:02 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pagmo
=====

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


pagmo is a C++ scientific library for massively parallel optimization.
It is built
around the idea of providing a unified interface to optimization
algorithms and problems, and to
make their deployment in massively parallel environments easy.

Efficient implementantions of bio-inspired and evolutionary algorithms
are sided to state-of-the-art optimization algorithms
(Simplex Methods, SQP methods,  interior points methods, ...)
and can be easily mixed (also with your newly-invented algorithms)
to build a super-algorithm exploiting algorithmic cooperation
via the asynchronous, generalized island model.

pagmo can be used to solve constrained, unconstrained, single objective,
multiple objective, continuous and integer optimization
problems, stochastic and deterministic problems, as well as to perform
research on novel algorithms and paradigms and easily compare
them to state-of-the-art implementations of established ones.

If you are using pagmo as part of your research, teaching, or other activities, we would be grateful if you could star
the repository and/or cite our work. For citation purposes, you can use the following BibTex entry, which refers
to the `pagmo paper <https://doi.org/10.21105/joss.02338>`__ in the Journal of Open Source Software:

.. code-block:: bibtex

   @article{Biscani2020,
     doi = {10.21105/joss.02338},
     url = {https://doi.org/10.21105/joss.02338},
     year = {2020},
     publisher = {The Open Journal},
     volume = {5},
     number = {53},
     pages = {2338},
     author = {Francesco Biscani and Dario Izzo},
     title = {A parallel global multiobjective framework for optimization: pagmo},
     journal = {Journal of Open Source Software}
   }

The DOI of the latest version of the software is available at
`this link <https://doi.org/10.5281/zenodo.1045336>`__.

If you prefer using Python rather than C++, pagmo
can be used from Python via `pygmo <https://github.com/esa/pygmo2>`__,
its Python bindings.

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quickstart
   overview
   docs/cpp/tutorials/cpp_tut
   docs/cpp/cpp_docs
   credits
   changelog
