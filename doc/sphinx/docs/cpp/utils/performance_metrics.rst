.. _cpp_performance_metrics_utils:

Multi-objective performance metrics
===================================

A number of utilities to assess the quality of a set of objective vectors (e.g., the non dominated front
returned by a multi-objective optimization algorithm) with respect to a reference front, such as the Pareto
front of the problem or the best known approximation of it. All objectives are assumed to be minimized.

The modified inverted generational distance (IGD+) is described in the publication:

:Title: "Modified distance calculation in generational distance and inverted generational distance."
:Authors: Hisao Ishibuchi, Hiroyuki Masuda, Yuki Tanigaki and Yusuke Nojima.
:Published in: International Conference on Evolutionary Multi-Criterion Optimization (EMO 2015), Part II, pp. 110-125, 2015.
:DOI: `10.1007/978-3-319-15892-1_8 <https://doi.org/10.1007/978-3-319-15892-1_8>`__

--------------------------------------------------------------------------

.. doxygenfunction:: pagmo::hypervolume_ratio

--------------------------------------------------------------------------

.. doxygenfunction:: pagmo::igd_plus

--------------------------------------------------------------------------

.. doxygenfunction:: pagmo::normalized_igd_plus
