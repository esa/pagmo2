Base BGL topology
=================

.. versionadded:: 2.11

*#include <pagmo/topologies/base_bgl_topology.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: base_bgl_topology

   This class provides the basic building blocks to implement
   user-defined topologies (UDTs) based on the Boost Graph Library (BGL).

   Note that, by itself, this class does **not** satisfy all the requirements
   of a UDT. Specifically, this class is missing the mandatory ``push_back()`` member function,
   which has to be implemented in a derived class (see :cpp:class:`~pagmo::is_udt` for the
   full list of requirements a UDT must satisfy).

   This class provides a strong thread safety guarantee: any member function can be invoked
   concurrently with any other member function.

   .. seealso::

      https://www.boost.org/doc/libs/1_70_0/libs/graph/doc/index.html

   .. cpp:function:: base_bgl_topology()

      Default constructor.

      The default constructor will initialize an empty graph with no vertices and no edges.

   .. cpp:function:: base_bgl_topology(const base_bgl_topology &)
   .. cpp:function:: base_bgl_topology(base_bgl_topology &&) noexcept
   .. cpp:function:: base_bgl_topology &operator=(const base_bgl_topology &)
   .. cpp:function:: base_bgl_topology &operator=(base_bgl_topology &&) noexcept

      :cpp:class:`~pagmo::base_bgl_topology` is copy/move constructible, and copy/move assignable.
      Copy construction/assignment will perform deep copies, move operations will leave the moved-from object in
      an unspecified but valid state.

      :exception unspecified: when performing copy operations, any exception raised by the copy of the underlying graph object.

   .. cpp:function:: std::size_t num_vertices() const

      :return: the number of vertices in the topology.

   .. cpp:function:: bool are_adjacent(std::size_t i, std::size_t j) const

      Check if two vertices are adjacent.

      Two vertices *i* and *j* are adjacent if there is a directed edge connecting *i* to *j*.

      :param i: the first vertex index.
      :param j: the second vertex index.

      :return: ``true`` if *i* and *j* are adjacent, ``false`` otherwise.

      :exception std\:\:invalid_argument: if *i* or *j* are not smaller than the number of vertices.
      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t i) const

      Fetch the edges connecting to *i*.

      This function will return a pair of vectors of equal size, containing:

      * the list of all vertices connecting to *i*,
      * the weights of the edges.

      :param i: the vertex index.

      :return: the list of connections to *i*.

      :exception std\:\:invalid_argument: if *i* is not smaller than the number of vertices.
      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: void add_vertex()

      Add a vertex.

      This function will add a new vertex to the topology. The newly-added vertex
      will be disjoint from any other vertex in the topology (i.e., there are no
      connections to/from the new vertex).

      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: void add_edge(std::size_t i, std::size_t j, double w = 1)

      Add a new edge.

      This function will add a new edge of weight *w* connecting *i* to *j*.

      :param i: the first vertex index.
      :param j: the second vertex index.
      :param w: the edge's weight.

      :exception std\:\:invalid_argument: if either:

         * *i* or *j* are not smaller than the number of vertices,
         * *i* and *j* are already adjacent,
         * *w* is not in the :math:`\left[0, 1\right]` range.

      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: void remove_edge(std::size_t i, std::size_t j)

      Remove an existing edge.

      This function will remove the edge connecting *i* to *j*.

      :param i: the first vertex index.
      :param j: the second vertex index.

      :exception std\:\:invalid_argument: if either:

         * *i* or *j* are not smaller than the number of vertices,
         * *i* and *j* are not adjacent.

      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: void set_weight(std::size_t i, std::size_t j, double w)

      Set the weight of an edge.

      This function will set to *w* the weight of the edge connecting *i* to *j*.

      :param i: the first vertex index.
      :param j: the second vertex index.
      :param w: the desired weight.

      :exception std\:\:invalid_argument: if either:

         * *i* or *j* are not smaller than the number of vertices,
         * *i* and *j* are not adjacent,
         * *w* is not in the :math:`\left[0, 1\right]` range.

      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: void set_all_weights(double w)

      This function will set the weights of all edges in the topology to *w*.

      :param w: the edges' weight.

      :exception std\:\:invalid_argument: if *w* is not in the :math:`\left[0, 1\right]` range.
      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: std::string get_extra_info() const

      :return: a string containing human-readable information about the topology.

      :exception unspecified: any exception thrown by the public BGL API.

   .. cpp:function:: template <typename Archive> void load(Archive &ar, unsigned)
   .. cpp:function:: template <typename Archive> void save(Archive &ar, unsigned) const

      These functions implement the serialisation of a :cpp:class:`~pagmo::base_bgl_topology`.

      :param ar: the input/output archive.

      :exception unspecified: any exception thrown by the public BGL API.

.. cpp:namespace-pop::
