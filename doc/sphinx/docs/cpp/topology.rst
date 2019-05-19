Topology
========

.. versionadded:: 2.11

*#include <pagmo/topology.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: topology

   In the jargon of pagmo, a topology is an object that represents connections among
   :cpp:class:`islands <pagmo::island>` in an :cpp:class:`~pagmo::archipelago`.
   In essence, a topology is a *weighted directed graph* in which

   * the *vertices* (or *nodes*) are islands,
   * the *edges* (or *arcs*) are directed connections between islands across which information flows during the
     optimisation process (via the migration of individuals),
   * the *weights* of the edges (whose numerical values are the :math:`[0.,1.]` range) represent the migration
     probability.

   Following the same schema adopted for :cpp:class:`~pagmo::problem`, :cpp:class:`~pagmo::algorithm`, etc.,
   :cpp:class:`~pagmo::topology` exposes a type-erased generic
   interface to *user-defined topologies* (or UDT for short). UDTs are classes implementing a certain set
   of member functions that describe the properties of (and allow to interact with) a topology. Once
   defined and instantiated, a UDT can then be used to construct an instance of this class,
   :cpp:class:`~pagmo::topology`, which
   provides a generic interface to topologies for use by :cpp:class:`~pagmo::archipelago`.

   In a :cpp:class:`~pagmo::topology`, vertices in the graph are identified by a zero-based unique
   integral index (represented by a ``std::size_t``). This integral index corresponds to the index of an
   :cpp:class:`~pagmo::island` in an :cpp:class:`~pagmo::archipelago`.

   Every UDT must implement at least the following member functions:

   .. code-block:: c++

      std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;
      void push_back();

   The ``get_connections()`` function takes as input a vertex index ``n``, and it is expected to return
   a pair of vectors containing respectively:

   * the indices of the vertices which are connecting to ``n`` (that is, the list of vertices for which a directed edge
     towards ``n`` exists),
   * the weights (i.e., the migration probabilities) of the edges linking the connecting vertices to ``n``.

   The ``push_back()`` method is expected to add a new vertex to the topology, assigning it the next
   available index and establishing connections to other vertices. The ``push_back()`` method is invoked
   by :cpp:func:`pagmo::archipelago::push_back()` upon the insertion of a new island into an archipelago,
   and it is meant
   to allow the incremental construction of a topology. That is, after ``N`` calls to ``push_back()``
   on an initially-empty topology, the topology should contain ``N`` vertices and any number of edges (depending
   on the specifics of the topology).

   In addition to providing the above methods, a UDT must also be default, copy and move constructible.

   Additional optional methods can be implemented in a UDT:

   .. code-block:: c++

      std::string get_name() const;
      std::string get_extra_info() const;

   See the documentation of the corresponding methods in this class for details on how the optional
   methods in the UDT are used by :cpp:class:`~pagmo::topology`.

   TODO FIXME Topologies are often used in asynchronous operations involving migration in archipelagos. pagmo
   guarantees that only a single thread at a time is interacting with any topology, so there is no
   need to protect UDTs against concurrent access. Topologies however are **required** to offer at
   least the basic thread safety guarantee, in order to make it possible to use different
   topologies from different threads.

   .. warning::

      The only operations allowed on a moved-from :cpp:class:`pagmo::topology` are destruction,
      assignment, and the invocation of the :cpp:func:`~pagmo::topology::is_valid()` member function.
      Any other operation will result in undefined behaviour.

   .. cpp:function:: topology()

      Default constructor.

      The default constructor will initialize a :cpp:class:`~pagmo::topology` containing an
      :cpp:class:`~pagmo::unconnected` topology.

      :exception unspecified: any exception raised by the constructor from a generic UDT.

   .. cpp:function:: template <typename T> explicit topology(T &&x)

      Generic constructor from a UDT.

      This constructor participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::topology` and if it satisfies :cpp:class:`pagmo::is_udt`.

      This constructor will construct a :cpp:class:`~pagmo::topology` from the UDT (user-defined topology)
      *x* of type ``T``. The input parameter *x* will be perfectly forwarded to construct the internal UDT instance.

      :param x: the input UDT.

      :exception unspecified: any exception thrown by the public API of the UDT, or by memory allocation failures.

Associated type traits
----------------------

.. cpp:class:: template <typename T> has_get_connections

   The :cpp:any:`value` of this type trait will be ``true`` if
   ``T`` provides a member function with signature:

   .. code-block:: c++

      std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const;

   The ``get_connections()`` member function is part of the interface for the definition of a
   :cpp:class:`~pagmo::topology`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:class:: template <typename T> has_push_back

   The :cpp:any:`value` of this type trait will be ``true`` if
   ``T`` provides a member function with signature:

   .. code-block:: c++

      void push_back();

   The ``push_back()`` member function is part of the interface for the definition of a
   :cpp:class:`~pagmo::topology`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:class:: template <typename T> is_udt

   This type trait detects if ``T`` is a user-defined topology (or UDT).

   Specifically, the :cpp:any:`value` of this type trait will be ``true`` if:

   * ``T`` is not a reference or cv qualified,
   * ``T`` is destructible, default, copy and move constructible, and
   * ``T`` satisfies :cpp:class:`pagmo::has_get_connections` and
     :cpp:class:`pagmo::has_push_back`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:namespace-pop::
