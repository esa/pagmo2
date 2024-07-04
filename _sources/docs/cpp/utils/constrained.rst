.. _cpp_constrained_utils:

Constrained optimization utilities
======================================

A number of utilities to compute quantities that are of relevance to
constrained optimization tasks.

--------------------------------------------------------------------------

.. doxygenfunction:: pagmo::sort_population_con(const std::vector<vector_double>&, vector_double::size_type, const vector_double&)

--------------------------------------------------------------------------

.. doxygenfunction:: pagmo::sort_population_con(const std::vector<vector_double> &input_f, vector_double::size_type neq, double tol)

--------------------------------------------------------------------------

.. doxygenfunction:: pagmo::compare_fc(const vector_double&, const vector_double&, vector_double::size_type, const vector_double&)

--------------------------------------------------------------------------

.. doxygenfunction:: pagmo::compare_fc(const vector_double&, const vector_double&, vector_double::size_type, double)
