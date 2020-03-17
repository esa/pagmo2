Free-form topology
==================

.. versionadded:: 2.15

*#include <pagmo/topologies/free_form.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: free_form: public base_bgl_topology

   This user-defined topology (UDT) represents a graph in which
   vertices and edges can be manipulated freely. It is implemented
   as a simple extension of :cpp:class:`~pagmo::base_bgl_topology`
   in which the :cpp:func:`~free_form::push_back()` function
   adds a vertex without connections.

   .. cpp:function:: free_form()

      Default constructor.

      The default constructor initialises an empty topology.

   .. cpp:function:: free_form(const free_form &)
   .. cpp:function:: free_form(free_form &&) noexcept

      Copy and move constructors.

   .. cpp:function:: explicit free_form(bgl_graph_t g)

      Constructor from a :cpp:type:`~pagmo::bgl_graph_t`.

      The internal graph of the topology will be set to *g*.

      :param g: the graph that will be used to initialise ``this``.

      :exception std\:\:invalid_argument: if any edge in the graph has
        a weight outside the :math:`\left[ 0, 1 \right]` range.

   .. cpp:function:: explicit free_form(const topology &t)
   .. cpp:function:: template <typename T> explicit free_form(const T &t)

      .. note::

         The constructor from ``T`` does not participate in overload resolution
         if ``T`` is :cpp:class:`~pagmo::free_form` or if ``T`` does not
         satisfy :cpp:class:`~pagmo::is_udt`.

      Constructors from a :cpp:class:`~pagmo::topology` or a UDT.

      These constructors will first invoke the :cpp:func:`pagmo::topology::to_bgl()`
      function to extract a graph representation of the input :cpp:class:`~pagmo::topology`
      or UDT, and will then use that graph object to initialise this :cpp:class:`~pagmo::free_form`
      object via the constructor from :cpp:type:`~pagmo::bgl_graph_t`.

      In other words, these constructors allow to copy the graph
      of a :cpp:class:`~pagmo::topology` or a UDT into ``this``.

      :param t: the input :cpp:class:`~pagmo::topology` or UDT.

      :exception unspecified: any exception raised by the construction of a :cpp:class:`~pagmo::topology`
        object, the invocation of the :cpp:func:`pagmo::topology::to_bgl()` function or by
        the constructor from :cpp:type:`~pagmo::bgl_graph_t`.

   .. cpp:function:: void push_back()

      Add a new vertex.

      The newly-added vertex will not be connected to any other vertex.

      :exception unspecified: any exception thrown by the public API of :cpp:class:`~pagmo::base_bgl_topology`.

   .. cpp:function:: std::string get_name() const

      Get the name of the topology.

      :return: ``"Free form"``.

   .. cpp:function:: template <typename Archive> void serialize(Archive &ar, unsigned)

      This function implements the serialisation of a :cpp:class:`~pagmo::free_form`.

      :param ar: the input/output archive.

      :exception unspecified: any exception thrown by the serialisation of a :cpp:class:`~pagmo::base_bgl_topology`.

.. cpp:namespace-pop::
