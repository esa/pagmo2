Default BFE
===========

.. versionadded:: 2.11

*#include <pagmo/batch_evaluators/default_bfe.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: default_bfe

   This class is a user-defined batch fitness evaluator (UDBFE) that can be used to
   construct a :cpp:class:`~pagmo::bfe`.
   :cpp:class:`~pagmo::default_bfe` is the default UDBFE used by :cpp:class:`~pagmo::bfe`, and,
   depending on the properties of the input :cpp:class:`~pagmo::problem`, it will delegate the implementation
   of its call operator to :cpp:class:`~pagmo::member_bfe` or :cpp:class:`~pagmo::thread_bfe`.

   .. cpp:function:: vector_double operator()(const problem &p, const vector_double &dvs) const

      Call operator.

      The call operator will internally employ :cpp:class:`~pagmo::member_bfe` or :cpp:class:`~pagmo::thread_bfe`
      to perform the evaluation of the input batch of decision vectors *dvs*. The choice between :cpp:class:`~pagmo::member_bfe`
      and :cpp:class:`~pagmo::thread_bfe` is made according to the following heuristic:

      * if *p* provides a batch fitness member function (as established by :cpp:func:`pagmo::problem::has_batch_fitness()`),
        then a :cpp:class:`~pagmo::member_bfe` will be constructed and invoked to produce the return value; otherwise,
      * if *p* provides at least the :cpp:enumerator:`~pagmo::thread_safety::basic` thread safety level (as established
        by :cpp:func:`pagmo::problem::get_thread_safety()`), then  a :cpp:class:`~pagmo::thread_bfe` will be constructed
        and invoked to produce the return value.

      If *p* does not provide a batch fitness member function and if it does not provide at least the :cpp:enumerator:`~pagmo::thread_safety::basic`
      thread safety level, an error will be raised.

      :param p: the input :cpp:class:`~pagmo::problem`.
      :param dvs: the input decision vectors that will be evaluated.

      :return: the fitness vectors corresponding to the input decision vectors in *dvs*.

      :exception std\:\:invalid_argument: if the input problem *p* does not provide a batch fitness member function and it is does not provide at least the :cpp:enumerator:`~pagmo::thread_safety::basic` thread safety level.
      :exception unspecified: any exception raised by the call operator of :cpp:class:`~pagmo::member_bfe` or :cpp:class:`~pagmo::thread_bfe`.

   .. cpp:function:: std::string get_name() const

      :return: a human-readable name for this :cpp:class:`~pagmo::default_bfe`.

   .. cpp:function:: template <typename Archive> void serialize(Archive &, unsigned)

      Serialisation support.

      Note that :cpp:class:`~pagmo::default_bfe` is stateless, and thus this (de)serialisation function is empty and performs no work.

.. cpp:namespace-pop::
