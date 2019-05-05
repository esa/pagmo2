.. _cpp_type_traits:

Type traits and enums
=====================

.. cpp:namespace-push:: pagmo

.. cpp:enum-class:: thread_safety

   This enum defines a set of values that can be used to specify
   the thread safety level of an object.

   Most of the user-defined classes
   that can be implemented in pagmo (i.e., problems, algorithms, etc.) provide
   mechanisms to specify which thread safety level is provided by objects of
   these classes. The information on the thread safety level is used by pagmo
   to establish at runtime whether or not it is safe to use an object in a
   multithreaded context.

   For instance, :cpp:class:`~pagmo::thread_island` will refuse to run
   parallel optimisations in multiple threads if the involved
   :cpp:class:`~pagmo::problem` or :cpp:class:`~pagmo::algorithm`
   do not provide at least the :cpp:enumerator:`~pagmo::thread_safety::basic`
   thread safety level.

   The thread safety levels are ordered in the following way:
   :cpp:enumerator:`~pagmo::thread_safety::none`
   \< :cpp:enumerator:`~pagmo::thread_safety::basic`
   \< :cpp:enumerator:`~pagmo::thread_safety::constant`.

   .. cpp:enumerator:: none

      No thread safety: concurrent operations on distinct objects are unsafe.

   .. cpp:enumerator:: basic

      Basic thread safety: concurrent operations on distinct objects are safe.

   .. cpp:enumerator:: constant

      Constant thread safety: constant (i.e., read-only) concurrent operations
      on the same object are safe.

.. cpp:function:: std::ostream &operator<<(std::ostream &os, thread_safety ts)

   Stream operator for :cpp:enum:`~pagmo::thread_safety`. It will direct to the stream *os*
   a human-readable representation of *ts*.

   :param os: an output stream.
   :param ts: the :cpp:enum:`~pagmo::thread_safety` to be directed to the output stream.

   :return: a reference to *os*.

   :exception unspecified: any exception raised by the public interface of ``std::ostream``.

.. cpp:namespace-pop::

.. doxygenclass:: pagmo::is_udp
   :members:

.. doxygenclass:: pagmo::is_uda
   :members:

.. doxygenclass:: pagmo::has_fitness
   :members:

.. doxygenclass:: pagmo::has_bounds
   :members:

.. doxygenclass:: pagmo::has_e_constraints
   :members:

.. doxygenclass:: pagmo::has_i_constraints
   :members:

.. doxygenclass:: pagmo::has_integer_part
   :members:

.. doxygenclass:: pagmo::has_name
   :members:

.. doxygenclass:: pagmo::has_extra_info
   :members:

.. doxygenclass:: pagmo::has_get_thread_safety
   :members:

.. doxygenclass:: pagmo::has_gradient
   :members:

.. doxygenclass:: pagmo::override_has_gradient
   :members:

.. doxygenclass:: pagmo::has_gradient_sparsity
   :members:

.. doxygenclass:: pagmo::override_has_gradient_sparsity
   :members:

.. doxygenclass:: pagmo::has_hessians
   :members:

.. doxygenclass:: pagmo::override_has_hessians
   :members:

.. doxygenclass:: pagmo::has_hessians_sparsity
   :members:

.. doxygenclass:: pagmo::override_has_hessians_sparsity
   :members:

.. doxygenclass:: pagmo::has_set_verbosity
   :members:

.. doxygenclass:: pagmo::override_has_set_verbosity
   :members:

.. doxygenclass:: pagmo::has_evolve
   :members:

.. doxygenclass:: pagmo::has_get_nobj
   :members:

.. doxygenclass:: pagmo::has_set_seed
   :members:

.. doxygenclass:: pagmo::override_has_set_seed
   :members:

.. doxygenclass:: pagmo::has_run_evolve
   :members:

.. doxygenclass:: pagmo::is_udi
   :members:
