Multithreaded BFE
=================

.. versionadded:: 2.11

*#include <pagmo/batch_evaluators/thread_bfe.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: thread_bfe

   This class is a user-defined batch fitness evaluator (UDBFE) that can be used to
   construct a :cpp:class:`~pagmo::bfe`.
   :cpp:class:`~pagmo::thread_bfe` will use multiple threads of execution to parallelise
   the evaluation of the fitnesses of a batch of input decision vectors.

   .. cpp:function:: vector_double operator()(const problem &p, const vector_double &dvs) const

      Call operator.

      The call operator will use the input problem *p* to evaluate
      the fitnesses of the decision vectors stored contiguously in *dvs*. The fitness evaluation
      will be run in parallel using multiple threads of execution. Because of this, the input
      problem *p* must provide at least the :cpp:enumerator:`~pagmo::thread_safety::basic`
      thread safety level, otherwise an exception will be raised (see :cpp:func:`pagmo::problem::get_thread_safety()`).

      If *p* provides at least the :cpp:enumerator:`~pagmo::thread_safety::constant` thread safety level,
      then *p* will be shared across multiple threads and its :cpp:func:`~problem::fitness()` function
      will be called simultaneously from different threads. Otherwise, copies of *p* will be created and
      the :cpp:func:`~problem::fitness()` function will be called on these copies.

      :param p: the input :cpp:class:`~pagmo::problem`.
      :param dvs: the input decision vectors that will be evaluated.

      :return: the fitness vectors corresponding to the input decision vectors in *dvs*.

      :exception std\:\:invalid_argument: if *p* does not provide at least the :cpp:enumerator:`~pagmo::thread_safety::basic` thread safety level.
      :exception std\:\:overflow_error: in case of (unlikely) internal overflow conditions.
      :exception unspecified: any exception raised by memory allocation failures or by the public API of :cpp:class:`~pagmo::problem`.

   .. cpp:function:: std::string get_name() const

      :return: a human-readable name for this :cpp:class:`~pagmo::thread_bfe`.

   .. cpp:function:: template <typename Archive> void serialize(Archive &, unsigned)

      Serialisation support.

      Note that :cpp:class:`~pagmo::thread_bfe` is stateless, and thus this (de)serialisation function is empty and performs no work.

.. cpp:namespace-pop::
