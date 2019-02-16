Problem
=======

*#include <pagmo/problem.hpp>*

.. doxygenclass:: pagmo::problem
   :members:

.. cpp:namespace-push:: pagmo

Associated type traits
----------------------

.. cpp:class:: template <typename T> has_batch_fitness

   This type trait detects if ``T`` provides a member function whose signature
   is compatible with

   .. code-block:: c++

      vector_double batch_fitness(const vector_double &) const;

   The ``batch_fitness()`` member function is part of the interface for the definition of a
   user-defined problem (see the :cpp:class:`~pagmo::problem` documentation for details).

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:class:: template <typename T> override_has_batch_fitness

   This type trait detects if ``T`` provides a member function whose signature
   is compatible with

   .. code-block:: c++

      bool has_batch_fitness() const;

   The ``has_batch_fitness()`` member function is part of the interface for the definition of a
   user-defined problem (see the :cpp:class:`~pagmo::problem` documentation for details).

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:namespace-pop::
