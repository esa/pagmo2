Member function BFE
===================

.. versionadded:: 2.11

*#include <pagmo/batch_evaluators/member_bfe.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: member_bfe

   This class is a user-defined batch fitness evaluator (UDBFE) that can be used to
   construct a :cpp:class:`~pagmo::bfe`.
   :cpp:class:`~pagmo::member_bfe` is a simple wrapper which delegates batch fitness evaluations
   to the input problem's :cpp:func:`pagmo::problem::batch_fitness()` member function.

   .. cpp:function:: vector_double operator()(const problem &p, const vector_double &dvs) const

      Call operator.

      The call operator will pass *dvs* to the input problem *p*'s :cpp:func:`pagmo::problem::batch_fitness()` member function,
      and return its output. If the UDP stored within *p* does not implement :cpp:func:`pagmo::problem::batch_fitness()`,
      an error will be raised.

      :param p: the input :cpp:class:`~pagmo::problem`.
      :param dvs: the input decision vectors that will be evaluated.

      :return: the fitness vectors corresponding to the input decision vectors in *dvs*.

      :exception unspecified: any exception raised by the invocation of *p*'s :cpp:func:`pagmo::problem::batch_fitness()` member function.

   .. cpp:function:: std::string get_name() const

      :return: a human-readable name for this :cpp:class:`~pagmo::member_bfe`.

   .. cpp:function:: template <typename Archive> void serialize(Archive &, unsigned)

      Serialisation support.

      Note that :cpp:class:`~pagmo::member_bfe` is stateless, and thus this (de)serialisation function is empty and performs no work.

.. cpp:namespace-pop::
