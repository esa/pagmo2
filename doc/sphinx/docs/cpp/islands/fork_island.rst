Fork island
===========

.. versionadded:: 2.9

*#include <pagmo/islands/fork_island.hpp>*

.. note::

   The :cpp:class:`~pagmo::fork_island` class is available only on POSIX platforms (e.g., Linux, OSX,
   the BSDs, but *not* Microsoft Windows, unless some POSIX compatibility layer has been installed
   on the system).

.. cpp:namespace-push:: pagmo

.. cpp:class:: fork_island

   This user-defined island (UDI) will use the POSIX ``fork()`` system call to offload the evolution
   of a population to a child process.
   
   Generally speaking, users are encouraged to use :cpp:class:`~pagmo::thread_island` rather than
   :cpp:class:`~pagmo::fork_island`: :cpp:class:`~pagmo::thread_island` performs better,
   it works also with problems and algorithms which are not serialisable, and it is available on all
   platforms.

   :cpp:class:`~pagmo::fork_island` should however be preferred when dealing with problems and algorithms
   which do not offer the basic :cpp:type:`~pagmo::thread_safety` guarantee. By offloading the optimisation
   to a separate process (rather than a separate thread), :cpp:class:`~pagmo::fork_island` can ensure
   that thread-unsafe problems and algorithms are always run in only one thread at a time.
   This capability is particularly useful when wrapping in pagmo third-party code which does not support execution
   in multithreaded contexts (a notable example is the :cpp:class:`~pagmo::ipopt` algorithm,
   which uses the thread-unsafe IPOPT optimiser).

   :cpp:class:`~pagmo::fork_island` is the UDI type automatically selected by the constructors of :cpp:class:`~pagmo::island`
   on POSIX platforms when the island's problem and/or algorithm do not provide the basic :cpp:type:`~pagmo::thread_safety`
   guarantee.

   .. note::

      :cpp:class:`~pagmo::fork_island` is not exposed in the Python bindings. Instead, pygmo provides a
      :py:class:`process-based island <pygmo.mp_island>` via Python's :py:mod:`multiprocessing` module.

   .. note::

      When using memory checking tools such as valgrind, or the address/memory sanitizers from GCC/clang, be aware
      that memory leaks in the child process may be flagged by such tools. These are spurious warnings due to the
      fact that the child process is exited via ``std::exit()`` (which does not invoke the destructors
      of objects with automatic storage duration). Thus, such warnings can be safely ignored.

   .. cpp:function:: fork_island()
   .. cpp:function:: fork_island(const fork_island &)
   .. cpp:function:: fork_island(fork_island &&) noexcept

      :cpp:class:`~pagmo::fork_island` is default, copy and move-constructible.

   .. cpp:function:: void run_evolve(island &isl) const

      This method will fork the calling process, and, in the child process, the :cpp:class:`~pagmo::population` of *isl* will be
      evolved using the :cpp:class:`~pagmo::algorithm` of *isl*. At the end of the evolution, the evolved population and the
      algorithm used for the evolution will be sent back to the parent process, where they will replace, in *isl*, the original
      population and algorithm. The child process will then terminate via ``std::exit(0)``.

      If any exception is raised during the evolution, the error message from the exception will be transferred back to the parent
      process, where a ``std::runtime_error`` containing the error message from the child will be raised.

      :param isl: the :cpp:class:`~pagmo::island` that will be evolved.

      :exception std\:\:runtime_error: if any error arises from the use of POSIX primitives (``fork()``, pipes, etc.), or if any
         error is generated in the child process.
      :exception unspecified: any exception raised by:

         - the serialisation of :cpp:class:`~pagmo::population` or :cpp:class:`~pagmo::algorithm`,
         - :cpp:func:`pagmo::island::set_population()` or :cpp:func:`pagmo::island::set_algorithm()`.

   .. cpp:function:: std::string get_name() const

      :return: the string ``"Fork island"``.

   .. cpp:function:: std::string get_extra_info() const

      :return: if an evolution is ongoing, this method will return a string
         representation of the ID of the child process. Otherwise, the ``"No active child."`` string will be returned.

   .. cpp:function:: pid_t get_child_pid() const

      :return: a signed integral value representing the process ID of the child process, if an evolution is ongoing. Otherwise,
         ``0`` will be returned.

   .. cpp:function:: template <typename Archive> void serialize(Archive &)

      Serialisation support.

      Note that :cpp:class:`~pagmo::fork_island` is stateless, and thus this (de)serialisation function is empty and performs no work.

.. cpp:namespace-pop::
