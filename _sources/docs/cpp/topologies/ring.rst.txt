Ring
====

.. versionadded:: 2.11

*#include <pagmo/topologies/ring.hpp>*

.. image:: ../../images/ring.png

.. cpp:namespace-push:: pagmo

.. cpp:class:: ring: public base_bgl_topology

   This user-defined topology (UDT) represents a bidirectional ring (that is, a ring
   in which each node connects to both the previous and the following nodes).

   .. cpp:function:: ring()

      Default constructor.

      Equivalent to the constructor from edge weight with *w* = 1.

   .. cpp:function:: explicit ring(double w)

      Constructor from edge weight.

      New edges created via :cpp:func:`~pagmo::ring::push_back()` will have
      a weight of *w*.

      :param w: the weight of the edges.

      :except std\:\:invalid_argument: if *w* is not in the :math:`\left[0, 1\right]` range.

   .. cpp:function:: explicit ring(std::size_t n, double w)

      Constructor from number of vertices and edge weight.

      This constructor will initialise a ring topology with *n* vertices and whose
      edges will have a weight of *w*.

      New edges created via subsequent :cpp:func:`~pagmo::ring::push_back()` calls
      will also have a weight of *w*.

      :param n: the desired number of vertices.
      :param w: the weight of the edges.

      :except std\:\:invalid_argument: if *w* is not in the :math:`\left[0, 1\right]` range.
      :except unspecified: any exception thrown by :cpp:func:`~pagmo::ring::push_back()`.

   .. cpp:function:: void push_back()

      Add the next vertex.

      :except unspecified: any exception thrown by the public API of :cpp:class:`~pagmo::base_bgl_topology`.

   .. cpp:function:: double get_weight() const

      :return: the weight *w* used when constructing this topology.

   .. cpp:function:: std::string get_name() const

      Get the name of the topology.

      :return: ``"Ring"``.

   .. cpp:function:: template <typename Archive> void serialize(Archive &ar, unsigned)

      This function implements the serialisation of a :cpp:class:`~pagmo::ring`.

      :param ar: the input/output archive.

      :exception unspecified: any exception thrown by the serialisation of a :cpp:class:`~pagmo::base_bgl_topology`
         or of primitive types.

.. cpp:namespace-pop::
