Island
======

*#include <pagmo/island.hpp>*

.. doxygenclass:: pagmo::island
   :members:

Functions
---------

.. cpp:namespace-push:: pagmo

.. cpp:function:: std::ostream &operator<<(std::ostream &os, const island &isl)

   Stream operator for :cpp:class:`pagmo::island`.

   This operator will stream to *os* a human-readable representation of *isl*.

   It is safe to call this method while the island is evolving.

   :param os: the target stream.
   :param isl: the input island.

   :return: a reference to *os*.

   :exception unspecified: any exception trown by the stream operators of fundamental types or by
      the public interface of :cpp:class:`pagmo::island` and of all its members.

.. cpp:namespace-pop::

Types
-----

.. doxygenenum:: pagmo::evolve_status
