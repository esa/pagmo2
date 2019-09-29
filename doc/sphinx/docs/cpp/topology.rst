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
   interface to *user-defined topologies* (or UDT for short). UDTs are classes providing a certain set
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

   The ``push_back()`` member function is expected to add a new vertex to the topology, assigning it the next
   available index and establishing connections to other vertices. The ``push_back()`` member function is invoked
   by :cpp:func:`pagmo::archipelago::push_back()` upon the insertion of a new island into an archipelago,
   and it is meant
   to allow the incremental construction of a topology. That is, after ``N`` calls to ``push_back()``
   on an initially-empty topology, the topology should contain ``N`` vertices and any number of edges (depending
   on the specifics of the topology).

   In addition to providing the above member functions, a UDT must also be default, copy and move constructible.

   Additional optional member functions can be implemented in a UDT:

   .. code-block:: c++

      std::string get_name() const;
      std::string get_extra_info() const;

   See the documentation of the corresponding member functions in this class for details on how the optional
   member functions in the UDT are used by :cpp:class:`~pagmo::topology`.

   Topologies are used in asynchronous operations involving migration in archipelagos,
   and thus they need to provide a certain degree of thread safety. Specifically, the
   ``get_connections()`` member function of the UDT might be invoked concurrently with
   any other member function of the UDT interface (except for the destructor, the move
   constructor, and, if implemented, the deserialisation function). It is up to the
   authors of user-defined topologies to ensure that this safety requirement is satisfied.

   .. warning::

      The only operations allowed on a moved-from :cpp:class:`pagmo::topology` are destruction,
      assignment, and the invocation of the :cpp:func:`~pagmo::topology::is_valid()` member function.
      Any other operation will result in undefined behaviour.

   .. cpp:function:: topology()

      Default constructor.

      The default constructor will initialize a :cpp:class:`~pagmo::topology` containing an
      :cpp:class:`~pagmo::unconnected` topology.

      :exception unspecified: any exception raised by the constructor from a generic UDT.

   .. cpp:function:: topology(const topology &)
   .. cpp:function:: topology(topology &&) noexcept
   .. cpp:function:: topology &operator=(const topology &)
   .. cpp:function:: topology &operator=(topology &&) noexcept

      :cpp:class:`~pagmo::topology` is copy/move constructible, and copy/move assignable.
      Copy construction/assignment will perform deep copies, move operations will leave the moved-from object in
      a state which is destructible and assignable.

      :exception unspecified: when performing copy operations, any exception raised by the UDT upon copying, or by memory allocation failures.

   .. cpp:function:: template <typename T> explicit topology(T &&x)

      Generic constructor from a UDT.

      This constructor participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::topology` and if it satisfies :cpp:class:`pagmo::is_udt`.

      This constructor will construct a :cpp:class:`~pagmo::topology` from the UDT (user-defined topology)
      *x* of type ``T``. The input parameter *x* will be perfectly forwarded to construct the internal UDT instance.

      :param x: the input UDT.

      :exception unspecified: any exception thrown by the public API of the UDT, or by memory allocation failures.

   .. cpp:function:: template <typename T> topology &operator=(T &&x)

      Generic assignment operator from a UDT.

      This operator participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::topology` and if it satisfies :cpp:class:`pagmo::is_udt`.

      This operator will set the internal UDT to *x* by constructing a :cpp:class:`~pagmo::topology` from *x*,
      and then move-assigning the result to *this*.

      :param x: the input UDT.

      :return: a reference to *this*.

      :exception unspecified: any exception thrown by the generic constructor from a UDT.

   .. cpp:function:: template <typename T> const T *extract() const noexcept
   .. cpp:function:: template <typename T> T *extract() noexcept

      Extract a (const) pointer to the internal UDT instance.

      If ``T`` is the type of the UDT currently stored within this object, then this function
      will return a (const) pointer to the internal UDT instance. Otherwise, ``nullptr`` will be returned.

      The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
      of ``this``, and ``delete`` must never be called on the pointer.

      .. warning::

         The non-const overload of this function is provided only in order to allow to call non-const
         member functions on the internal UDT instance. Assigning a new UDT via pointers obtained
         through this function is undefined behaviour.

      :return: a (const) pointer to the internal UDT instance, or ``nullptr``.

   .. cpp:function:: template <typename T> bool is() const noexcept

      Check the type of the UDT.

      :return: ``true`` if ``T`` is the type of the UDT currently stored within this object, ``false`` otherwise.

   .. cpp:function:: std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t n) const

      Get the connections to a vertex.

      This function will invoke the ``get_connections()`` member function of the UDT, which is expected to return
      a pair of vectors containing respectively:

      * the indices of the vertices which are connecting to *n* (that is, the list of vertices for which a directed
        edge towards *n* exists),
      * the weights (i.e., the migration probabilities) of the edges linking the connecting vertices to *n*.

      This function will also run sanity checks on the output of the ``get_connections()`` member function of the UDT.

      :param n: the index of the vertex whose incoming connections' details will be returned.

      :return: a pair of vectors describing *n*'s incoming connections.

      :exception std\:\:invalid_argument: if the sizes of the returned vectors differ, or if any element of the second
        vector is not in the :math:`[0.,1.]` range.
      :exception unspecified: any exception thrown by the ``get_connections()`` member function of the UDT.

   .. cpp:function:: void push_back()

      Add a vertex.

      This member function will invoke the ``push_back()`` member function of the UDT, which is expected to add a new vertex to the
      topology, assigning it the next available index and establishing connections to other vertices.

      :exception unspecified: any exception thrown by the ``push_back()`` member function of the UDT.

   .. cpp:function:: void push_back(unsigned n)

      Add multiple vertices.

      This member function will call :cpp:func:`~pagmo::topology::push_back()` *n* times.

      :param n: the number of times :cpp:func:`~pagmo::topology::push_back()` will be called.

      :exception unspecified: any exception thrown by :cpp:func:`~pagmo::topology::push_back()`.

   .. cpp:function:: std::string get_name() const

      Get the name of this topology.

      If the UDT satisfies :cpp:class:`pagmo::has_name`, then this member function will return the output of its ``get_name()`` member function.
      Otherwise, an implementation-defined name based on the type of the UDT will be returned.

      :return: the name of this topology.

      :exception unspecified: any exception thrown by copying an ``std::string`` object.

   .. cpp:function:: std::string get_extra_info() const

      Extra info for this topology.

      If the UDT satisfies :cpp:class:`pagmo::has_extra_info`, then this member function will return the output of its
      ``get_extra_info()`` member function. Otherwise, an empty string will be returned.

      :return: extra info about the UDT.

      :exception unspecified: any exception thrown by the ``get_extra_info()`` member function of the UDT, or by copying an ``std::string`` object.

   .. cpp:function:: bool is_valid() const

      Check if this topology is in a valid state.

      :return: ``false`` if *this* was moved from, ``true`` otherwise.

   .. cpp:function:: template <typename Archive> void save(Archive &ar, unsigned) const
   .. cpp:function:: template <typename Archive> void load(Archive &ar, unsigned)

      Serialisation support.

      These two member functions are used to implement the (de)serialisation of a topology to/from an archive.

      :param ar: the input/output archive.

      :exception unspecified: any exception raised by the (de)serialisation of primitive types or of the UDT.

Functions
---------

.. cpp:function:: std::ostream &operator<<(std::ostream &os, const topology &t)

   Stream insertion operator.

   This function will direct to *os* a human-readable representation of the input
   :cpp:class:`~pagmo::topology` *t*.

   :param os: the input ``std::ostream``.
   :param t: the topology that will be directed to *os*.

   :return: a reference to *os*.

   :exception unspecified: any exception thrown by querying various properties of the topology and directing them to *os*.

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
