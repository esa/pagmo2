Ring
====

.. versionadded:: 2.11

*#include <pagmo/topologies/ring.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: ring: public base_bgl_topology

   This user-defined topology (UDT) represent a bidirectional ring (that is, a ring
   in which each node connects to both the previous and the following nodes).

   .. cpp:function:: void push_back()

      Add the next vertex.

      This method is a no-op.

   .. cpp:function:: std::string get_name() const

      Get the name of the topology.

      :return: ``"Ring"``.

   .. cpp:function:: template <typename Archive> void serialize(Archive &, unsigned)

      Serialisation support.

.. cpp:namespace-pop::
