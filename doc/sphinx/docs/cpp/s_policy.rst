Selection policy
================

.. versionadded:: 2.11

*#include <pagmo/s_policy.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: s_policy

   Selection policy.

   A selection policy establishes
   how, during migration within an :cpp:class:`~pagmo::archipelago`,
   candidate migrants are selected from an :cpp:class:`~pagmo::island`.

   Following the same schema adopted for :cpp:class:`~pagmo::problem`, :cpp:class:`~pagmo::algorithm`, etc.,
   :cpp:class:`~pagmo::s_policy` exposes a type-erased generic
   interface to *user-defined selection policies* (or UDSP for short).
   UDSPs are classes providing a certain set
   of member functions that implement the logic of the selection policy. Once
   defined and instantiated, a UDSP can then be used to construct an instance of this class,
   :cpp:class:`~pagmo::s_policy`, which
   provides a generic interface to selection policies for use by :cpp:class:`~pagmo::island`.

   Every UDSP must implement at least the following member function:

   .. code-block:: c++

      individuals_group_t select(const individuals_group_t &, const vector_double::size_type &,
                                 const vector_double::size_type &, const vector_double::size_type &,
                                 const vector_double::size_type &, const vector_double::size_type &,
                                 const vector_double &) const;

   The ``select()`` function takes in input the following parameters:

   * a group of individuals *inds* (represented as an :cpp:type:`~pagmo::individuals_group_t`),
   * a set of arguments describing the properties of the :cpp:class:`~pagmo::problem` the individuals refer to:

     * the total dimension *nx*,
     * the integral dimension *nix*,
     * the number of objectives *nobj*,
     * the number of equality constraints *nec*,
     * the number of inequality constraints *nic*,
     * the problem's constraint tolerances *tol*,

   and it produces in output another set of individuals resulting from selecting individuals in *inds*
   (following some logic established by the UDSP).

   In addition to providing the above member function, a UDSP must also be default, copy and move constructible.

   Additional optional member functions can be implemented in a UDSP:

   .. code-block:: c++

      std::string get_name() const;
      std::string get_extra_info() const;

   See the documentation of the corresponding member functions in this class for details on how the optional
   member functions in the UDSP are used by :cpp:class:`~pagmo::s_policy`.

   Selection policies are used in asynchronous operations involving migration in archipelagos,
   and thus they need to provide a certain degree of thread safety. Specifically, the
   ``select()`` member function of the UDSP might be invoked concurrently with
   any other member function of the UDSP interface (except for the destructor, the move
   constructor, and, if implemented, the deserialisation function). It is up to the
   authors of user-defined selection policies to ensure that this safety requirement is satisfied.

   .. warning::

      The only operations allowed on a moved-from :cpp:class:`pagmo::s_policy` are destruction,
      assignment, and the invocation of the :cpp:func:`~pagmo::s_policy::is_valid()` member function.
      Any other operation will result in undefined behaviour.

   .. cpp:function:: s_policy()

      Default constructor.

      The default constructor will initialize an :cpp:class:`~pagmo::s_policy` containing a
      :cpp:class:`~pagmo::select_best` selection policy.

      :exception unspecified: any exception raised by the constructor from a generic UDSP.

   .. cpp:function:: s_policy(const s_policy &)
   .. cpp:function:: s_policy(s_policy &&) noexcept
   .. cpp:function:: s_policy &operator=(const s_policy &)
   .. cpp:function:: s_policy &operator=(s_policy &&) noexcept

      :cpp:class:`~pagmo::s_policy` is copy/move constructible, and copy/move assignable.
      Copy construction/assignment will perform deep copies, move operations will leave the moved-from object in
      a state which is destructible and assignable.

      :exception unspecified: when performing copy operations, any exception raised by the UDSP upon copying, or by memory allocation failures.

   .. cpp:function:: template <typename T> explicit s_policy(T &&x)

      Generic constructor from a UDSP.

      This constructor participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::s_policy` and if it satisfies :cpp:class:`pagmo::is_udsp`.

      This constructor will construct an :cpp:class:`~pagmo::s_policy` from the UDSP (user-defined selection policy)
      *x* of type ``T``. The input parameter *x* will be perfectly forwarded to construct the internal UDSP instance.

      :param x: the input UDSP.

      :exception unspecified: any exception thrown by the public API of the UDSP, or by memory allocation failures.

   .. cpp:function:: template <typename T> s_policy &operator=(T &&x)

      Generic assignment operator from a UDSP.

      This operator participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::s_policy` and if it satisfies :cpp:class:`pagmo::is_udsp`.

      This operator will set the internal UDSP to *x* by constructing an :cpp:class:`~pagmo::s_policy` from *x*,
      and then move-assigning the result to *this*.

      :param x: the input UDSP.

      :return: a reference to *this*.

      :exception unspecified: any exception thrown by the generic constructor from a UDSP.

   .. cpp:function:: template <typename T> const T *extract() const noexcept
   .. cpp:function:: template <typename T> T *extract() noexcept

      Extract a (const) pointer to the internal UDSP instance.

      If ``T`` is the type of the UDSP currently stored within this object, then this function
      will return a (const) pointer to the internal UDSP instance. Otherwise, ``nullptr`` will be returned.

      The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
      of ``this``, and ``delete`` must never be called on the pointer.

      .. warning::

         The non-const overload of this function is provided only in order to allow to call non-const
         member functions on the internal UDSP instance. Assigning a new UDSP via pointers obtained
         through this function is undefined behaviour.

      :return: a (const) pointer to the internal UDSP instance, or ``nullptr``.

   .. cpp:function:: template <typename T> bool is() const noexcept

      Check the type of the UDSP.

      :return: ``true`` if ``T`` is the type of the UDSP currently stored within this object, ``false`` otherwise.

   .. cpp:function:: individuals_group_t select(const individuals_group_t &inds, const vector_double::size_type &nx, \
         const vector_double::size_type &nix, const vector_double::size_type &nobj, \
         const vector_double::size_type &nec, const vector_double::size_type &nic, \
         const vector_double &tol) const

      Select individuals from a group.

      This member function will invoke the ``select()`` member function of the UDSP.
      Given a set of individuals, *inds*, the ``select()`` member function of the UDSP
      is expected to return a new set of individuals selected from *inds*.
      The other arguments of this member function describe the properties of the :cpp:class:`~pagmo::problem`
      that the individuals in *inds* refer to.

      In addition to invoking the ``select()`` member function of the UDSP, this function will also
      perform a variety of sanity checks on both the input arguments and on the output produced by the
      UDSP.

      :param inds: the original group of individuals.
      :param nx: the dimension of the problem *inds* refers to.
      :param nix: the integral dimension of the problem *inds* refers to.
      :param nobj: the number of objectives of the problem *inds* refers to.
      :param nec: the number of equality constraints of the problem *inds* refers to.
      :param nic: the number of inequality constraints of the problem *inds* refers to.
      :param tol: the vector of constraints tolerances of the problem *inds* refers to.

      :return: a new set of individuals resulting from selecting individuals in *inds*.

      :exception std\:\:invalid_argument: if either:

         * *inds* or the return value are not consistent with the problem properties,
         * the ID, decision and fitness vectors in *inds* or the return value have inconsistent sizes,
         * the problem properties are invalid (e.g., *nobj* is zero, *nix* > *nx*, etc.).

      :exception unspecified: any exception raised by the ``select()`` member function of the UDSP.

   .. cpp:function:: std::string get_name() const

      Get the name of this selection policy.

      If the UDSP satisfies :cpp:class:`pagmo::has_name`, then this member function will return the output of its ``get_name()`` member function.
      Otherwise, an implementation-defined name based on the type of the UDSP will be returned.

      :return: the name of this selection policy.

      :exception unspecified: any exception thrown by copying an ``std::string`` object.

   .. cpp:function:: std::string get_extra_info() const

      Extra info for this selection policy.

      If the UDSP satisfies :cpp:class:`pagmo::has_extra_info`, then this member function will return the output of its
      ``get_extra_info()`` member function. Otherwise, an empty string will be returned.

      :return: extra info about the UDSP.

      :exception unspecified: any exception thrown by the ``get_extra_info()`` member function of the UDSP, or by copying an ``std::string`` object.

   .. cpp:function:: bool is_valid() const

      Check if this selection policy is in a valid state.

      :return: ``false`` if *this* was moved from, ``true`` otherwise.

   .. cpp:function:: template <typename Archive> void save(Archive &ar, unsigned) const
   .. cpp:function:: template <typename Archive> void load(Archive &ar, unsigned)

      Serialisation support.

      These two member functions are used to implement the (de)serialisation of a selection policy to/from an archive.

      :param ar: the input/output archive.

      :exception unspecified: any exception raised by the (de)serialisation of primitive types or of the UDSP.

Functions
---------

.. cpp:function:: std::ostream &operator<<(std::ostream &os, const s_policy &s)

   Stream insertion operator.

   This function will direct to *os* a human-readable representation of the input
   :cpp:class:`~pagmo::s_policy` *s*.

   :param os: the input ``std::ostream``.
   :param s: the selection policy that will be directed to *os*.

   :return: a reference to *os*.

   :exception unspecified: any exception thrown by querying various properties of the selection policy and directing them to *os*.

Associated type traits
----------------------

.. cpp:class:: template <typename T> has_select

   The :cpp:any:`value` of this type trait will be ``true`` if
   ``T`` provides a member function with signature:

   .. code-block:: c++

      individuals_group_t select(const individuals_group_t &, const vector_double::size_type &,
                                 const vector_double::size_type &, const vector_double::size_type &,
                                 const vector_double::size_type &, const vector_double::size_type &,
                                 const vector_double &) const;

   The ``select()`` member function is part of the interface for the definition of an
   :cpp:class:`~pagmo::s_policy`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:class:: template <typename T> is_udsp

   This type trait detects if ``T`` is a user-defined selections policy (or UDSP).

   Specifically, the :cpp:any:`value` of this type trait will be ``true`` if:

   * ``T`` is not a reference or cv qualified,
   * ``T`` is destructible, default, copy and move constructible, and
   * ``T`` satisfies :cpp:class:`pagmo::has_select`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:namespace-pop::
