Thread island
=============

*#include <pagmo/islands/thread_island.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: thread_island

   Thread island.

   This class is a user-defined island (UDI) that will run evolutions in a separate
   thread of execution.

   :cpp:class:`~pagmo::thread_island` is the UDI type automatically selected by
   the constructors of :cpp:class:`~pagmo::island`
   on non-POSIX platforms or when both the island's problem and algorithm provide at least the
   :cpp:enumerator:`~pagmo::thread_safety::basic` thread safety guarantee.

   .. cpp:function:: explicit thread_island(bool use_pool)

   .. versionadded:: 2.16

   Constructor with *use_pool* flag.

   The *use_pool* flag signals whether or not this island should use a common thread pool
   shared by all islands.

   Using a thread pool is more computationally-efficient, for at least
   two reasons:

   * it avoids runtime overhead when
     the number of islands evolving simultaneously is larger than the CPU
     count (e.g., in a large :cpp:class:`~pagmo::archipelago`);
   * because the implementation uses the Intel TBB libraries, it integrates
     better with other pagmo facilities built on top of TBB (e.g., the
     :cpp:class:`~pagmo::thread_bfe` batch fitness evaluator).

   A thread pool
   however also introduces a serializing behaviour because the number
   of evolutions actually running at the same time is limited by the CPU
   count (whereas without the thread pool an unlimited number of evolutions
   can be active at the same time, albeit with a performance penalty).

   :param use_pool: a boolean flag signalling whether or not a thread pool should be
     used by the island.

   .. cpp:function:: thread_island()

   Default constructor, equivalent to the previous constructor
   with *use_pool* set to ``True``.

   .. cpp:function:: std::string get_name() const

   Island's name.

   :return: ``"Thread island"``.

   .. cpp:function:: std::string get_extra_info() const

   .. versionadded:: 2.16

   Island's extra info.

   :return: a string containing extra info about this island instance.

   .. cpp:function:: void run_evolve(island &isl) const

   Run an evolution.

   This method will use copies of *isl*'s
   algorithm and population, obtained via :cpp:func:`pagmo::island::get_algorithm()`
   and :cpp:func:`pagmo::island::get_population()`,
   to evolve the input island's population. The evolved population will be assigned to *isl*
   using :cpp:func:`pagmo::island::set_population()`, and the algorithm used for the
   evolution will be assigned to *isl* using :cpp:func:`pagmo::island::set_algorithm()`.

   :param isl: the :cpp:class:`~pagmo::island` that will undergo evolution.

   :exception std\:\:invalid_argument: if *isl*'s algorithm or problem do not provide
     at least the :cpp:enumerator:`~pagmo::thread_safety::basic` thread safety guarantee.
   :exception unspecified: any exception thrown by:

     * :cpp:func:`pagmo::island::get_algorithm()`, :cpp:func:`pagmo::island::get_population()`,
     * :cpp:func:`pagmo::island::set_algorithm()`, :cpp:func:`pagmo::island::set_population()`,
     * :cpp:func:`pagmo::algorithm::evolve()`.

   .. cpp:function:: template <typename Archive> void save(Archive &ar, unsigned) const
   .. cpp:function:: template <typename Archive> void load(Archive &ar, unsigned version)

      Serialisation support.

      These member functions are used to implement the (de)serialisation of an island to/from an archive.

      :param ar: the input/output archive.
      :param version: the archive version.

      :exception unspecified: any exception raised by the (de)serialisation of primitive types.

.. cpp:namespace-pop::
