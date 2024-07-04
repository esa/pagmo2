Types
=====

*#include <pagmo/types.hpp>*

.. doxygentypedef:: pagmo::vector_double

.. doxygentypedef:: pagmo::sparsity_pattern

.. doxygentypedef:: pagmo::pop_size_t

.. cpp:namespace-push:: pagmo

.. cpp:type:: individuals_group_t = std::tuple<std::vector<unsigned long long>, std::vector<vector_double>, std::vector<vector_double>>

   .. versionadded:: 2.11

   Group of individuals.

   This tuple represents a group of individuals via:

   * a vector of ``unsigned long long`` representing the IDs of the individuals,
   * a vector of :cpp:type:`~pagmo::vector_double` representing the decision vectors
     (or chromosomes) of the individuals,
   * another vector of :cpp:type:`~pagmo::vector_double` representing the fitness
     vectors of the individuals.

   In other words, :cpp:type:`~pagmo::individuals_group_t` is a stripped-down version of
   :cpp:class:`~pagmo::population` without the :cpp:class:`~pagmo::problem`. :cpp:type:`~pagmo::individuals_group_t`
   is used to exchange individuals between the islands of an :cpp:class:`~pagmo::archipelago` during migration.

.. cpp:namespace-pop::
