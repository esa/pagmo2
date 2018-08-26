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

   :cpp:class:`~pagmo::fork_island` should however be preferred (if available) when dealing with problems and algorithms
   which do not offer the basic :cpp:type:`~pagmo::thread_safety` guarantee. By offloading the optimisation
   to a separate process (rather than a separate thread), :cpp:class:`~pagmo::fork_island` can ensure
   that thread-unsafe problems and algorithms are always run in only one thread at a time.

   Users are encouraged to code problems and algorithms that provide at least the basic :cpp:type:`~pagmo::thread_safety` guarantee.
   Sometimes, however, the need arises to wrap in pagmo third-party code which does not support execution
   in multithreaded contexts (a notable example is the :cpp:class:`~pagmo::ipopt` algorithm,
   which uses the thread-unsafe IPOPT optimiser). In such cases, :cpp:class:`~pagmo::fork_island`
   provides a way of circumventing the limitations of third-party code to fully exploit pagmo's parallel
   capabilities.

   Note that :cpp:class:`~pagmo::fork_island` is not exposed in the Python bindings. Instead, pygmo provides a
   :py:class:`process-based island <pygmo.mp_island>` via Python's :py:mod:`multiprocessing` module.

.. cpp:namespace-pop::
