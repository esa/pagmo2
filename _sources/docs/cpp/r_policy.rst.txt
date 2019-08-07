Replacement policy
==================

.. versionadded:: 2.11

*#include <pagmo/r_policy.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: r_policy

   Replacement policy.

   A replacement policy establishes
   how, during migration within an :cpp:class:`~pagmo::archipelago`,
   a group of migrants replaces individuals in an existing
   :cpp:class:`~pagmo::population`. In other words, a replacement
   policy is tasked with producing a new set of individuals from
   an original set of individuals and a set of candidate migrants.

   Following the same schema adopted for :cpp:class:`~pagmo::problem`, :cpp:class:`~pagmo::algorithm`, etc.,
   :cpp:class:`~pagmo::r_policy` exposes a type-erased generic
   interface to *user-defined replacement policies* (or UDRP for short).
   UDRPs are classes providing a certain set
   of member functions that implement the logic of the replacement policy. Once
   defined and instantiated, a UDRP can then be used to construct an instance of this class,
   :cpp:class:`~pagmo::r_policy`, which
   provides a generic interface to replacement policies for use by :cpp:class:`~pagmo::island`.

   Every UDRP must implement at least the following member function:

   .. code-block:: c++

      individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                  const vector_double::size_type &, const vector_double::size_type &,
                                  const vector_double::size_type &, const vector_double::size_type &,
                                  const vector_double &, const individuals_group_t &) const;

   The ``replace()`` function takes in input the following parameters:

   * a group of individuals *inds* (represented as an :cpp:type:`~pagmo::individuals_group_t`),
   * a set of arguments describing the properties of the :cpp:class:`~pagmo::problem` the individuals refer to:

     * the total dimension *nx*,
     * the integral dimension *nix*,
     * the number of objectives *nobj*,
     * the number of equality constraints *nec*,
     * the number of inequality constraints *nic*,
     * the problem's constraint tolerances *tol*,

   * a set of migrants *mig*,

   and it produces in output another set of individuals resulting from replacing individuals in *inds* with
   individuals from *mig* (following some logic established by the UDRP).

   In addition to providing the above member function, a UDRP must also be default, copy and move constructible.

   Additional optional member functions can be implemented in a UDRP:

   .. code-block:: c++

      std::string get_name() const;
      std::string get_extra_info() const;

   See the documentation of the corresponding member functions in this class for details on how the optional
   member functions in the UDRP are used by :cpp:class:`~pagmo::r_policy`.

   Replacement policies are used in asynchronous operations involving migration in archipelagos,
   and thus they need to provide a certain degree of thread safety. Specifically, the
   ``replace()`` member function of the UDRP might be invoked concurrently with
   any other member function of the UDRP interface (except for the destructor, the move
   constructor, and, if implemented, the deserialisation function). It is up to the
   authors of user-defined replacement policies to ensure that this safety requirement is satisfied.

   .. warning::

      The only operations allowed on a moved-from :cpp:class:`pagmo::r_policy` are destruction,
      assignment, and the invocation of the :cpp:func:`~pagmo::r_policy::is_valid()` member function.
      Any other operation will result in undefined behaviour.

   .. cpp:function:: r_policy()

      Default constructor.

      The default constructor will initialize an :cpp:class:`~pagmo::r_policy` containing a
      :cpp:class:`~pagmo::fair_replace` replacement policy.

      :exception unspecified: any exception raised by the constructor from a generic UDRP.

   .. cpp:function:: r_policy(const r_policy &)
   .. cpp:function:: r_policy(r_policy &&) noexcept
   .. cpp:function:: r_policy &operator=(const r_policy &)
   .. cpp:function:: r_policy &operator=(r_policy &&) noexcept

      :cpp:class:`~pagmo::r_policy` is copy/move constructible, and copy/move assignable.
      Copy construction/assignment will perform deep copies, move operations will leave the moved-from object in
      a state which is destructible and assignable.

      :exception unspecified: when performing copy operations, any exception raised by the UDRP upon copying, or by memory allocation failures.

   .. cpp:function:: template <typename T> explicit r_policy(T &&x)

      Generic constructor from a UDRP.

      This constructor participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::r_policy` and if it satisfies :cpp:class:`pagmo::is_udrp`.

      This constructor will construct an :cpp:class:`~pagmo::r_policy` from the UDRP (user-defined replacement policy)
      *x* of type ``T``. The input parameter *x* will be perfectly forwarded to construct the internal UDRP instance.

      :param x: the input UDRP.

      :exception unspecified: any exception thrown by the public API of the UDRP, or by memory allocation failures.

   .. cpp:function:: template <typename T> r_policy &operator=(T &&x)

      Generic assignment operator from a UDRP.

      This operator participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::r_policy` and if it satisfies :cpp:class:`pagmo::is_udrp`.

      This operator will set the internal UDRP to *x* by constructing an :cpp:class:`~pagmo::r_policy` from *x*,
      and then move-assigning the result to *this*.

      :param x: the input UDRP.

      :return: a reference to *this*.

      :exception unspecified: any exception thrown by the generic constructor from a UDRP.

   .. cpp:function:: template <typename T> const T *extract() const noexcept
   .. cpp:function:: template <typename T> T *extract() noexcept

      Extract a (const) pointer to the internal UDRP instance.

      If ``T`` is the type of the UDRP currently stored within this object, then this function
      will return a (const) pointer to the internal UDRP instance. Otherwise, ``nullptr`` will be returned.

      The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
      of ``this``, and ``delete`` must never be called on the pointer.

      .. warning::

         The non-const overload of this function is provided only in order to allow to call non-const
         member functions on the internal UDRP instance. Assigning a new UDRP via pointers obtained
         through this function is undefined behaviour.

      :return: a (const) pointer to the internal UDRP instance, or ``nullptr``.

   .. cpp:function:: template <typename T> bool is() const noexcept

      Check the type of the UDRP.

      :return: ``true`` if ``T`` is the type of the UDRP currently stored within this object, ``false`` otherwise.

   .. cpp:function:: individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &nx, \
         const vector_double::size_type &nix, const vector_double::size_type &nobj, \
         const vector_double::size_type &nec, const vector_double::size_type &nic, \
         const vector_double &tol, const individuals_group_t &mig) const

      Replace individuals in a group with migrants from another group.

      This member function will invoke the ``replace()`` member function of the UDRP.
      Given a set of individuals, *inds*, and a set of migrants, *mig*, the ``replace()`` member function of the UDRP
      is expected to replace individuals in *inds*
      with individuals from *mig*, and return the new set of individuals resulting from the replacement.
      The other arguments of this member function describe the properties of the :cpp:class:`~pagmo::problem`
      that the individuals in *inds* and *mig* refer to.

      In addition to invoking the ``replace()`` member function of the UDRP, this function will also
      perform a variety of sanity checks on both the input arguments and on the output produced by the
      UDRP.

      :param inds: the original group of individuals.
      :param nx: the dimension of the problem *inds* and *mig* refer to.
      :param nix: the integral dimension of the problem *inds* and *mig* refer to.
      :param nobj: the number of objectives of the problem *inds* and *mig* refer to.
      :param nec: the number of equality constraints of the problem *inds* and *mig* refer to.
      :param nic: the number of inequality constraints of the problem *inds* and *mig* refer to.
      :param tol: the vector of constraints tolerances of the problem *inds* and *mig* refer to.
      :param mig: the group of migrants.

      :return: a new set of individuals resulting from replacing individuals in *inds* with individuals from *mig*.

      :exception std\:\:invalid_argument: if either:

         * *inds*, *mig* or the return value are not consistent with the problem properties,
         * the ID, decision and fitness vectors in *inds*, *mig* or the return value have inconsistent sizes,
         * the problem properties are invalid (e.g., *nobj* is zero, *nix* > *nx*, etc.).

      :exception unspecified: any exception raised by the ``replace()`` member function of the UDRP.

   .. cpp:function:: std::string get_name() const

      Get the name of this replacement policy.

      If the UDRP satisfies :cpp:class:`pagmo::has_name`, then this member function will return the output of its ``get_name()`` member function.
      Otherwise, an implementation-defined name based on the type of the UDRP will be returned.

      :return: the name of this replacement policy.

      :exception unspecified: any exception thrown by copying an ``std::string`` object.

   .. cpp:function:: std::string get_extra_info() const

      Extra info for this replacement policy.

      If the UDRP satisfies :cpp:class:`pagmo::has_extra_info`, then this member function will return the output of its
      ``get_extra_info()`` member function. Otherwise, an empty string will be returned.

      :return: extra info about the UDRP.

      :exception unspecified: any exception thrown by the ``get_extra_info()`` member function of the UDRP, or by copying an ``std::string`` object.

   .. cpp:function:: bool is_valid() const

      Check if this replacement policy is in a valid state.

      :return: ``false`` if *this* was moved from, ``true`` otherwise.

   .. cpp:function:: template <typename Archive> void save(Archive &ar, unsigned) const
   .. cpp:function:: template <typename Archive> void load(Archive &ar, unsigned)

      Serialisation support.

      These two member functions are used to implement the (de)serialisation of a replacement policy to/from an archive.

      :param ar: the input/output archive.

      :exception unspecified: any exception raised by the (de)serialisation of primitive types or of the UDRP.

Functions
---------

.. cpp:function:: std::ostream &operator<<(std::ostream &os, const r_policy &r)

   Stream insertion operator.

   This function will direct to *os* a human-readable representation of the input
   :cpp:class:`~pagmo::r_policy` *r*.

   :param os: the input ``std::ostream``.
   :param r: the replacement policy that will be directed to *os*.

   :return: a reference to *os*.

   :exception unspecified: any exception thrown by querying various properties of the replacement policy and directing them to *os*.

Associated type traits
----------------------

.. cpp:class:: template <typename T> has_replace

   The :cpp:any:`value` of this type trait will be ``true`` if
   ``T`` provides a member function with signature:

   .. code-block:: c++

      individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                  const vector_double::size_type &, const vector_double::size_type &,
                                  const vector_double::size_type &, const vector_double::size_type &,
                                  const vector_double &, const individuals_group_t &) const;

   The ``replace()`` member function is part of the interface for the definition of an
   :cpp:class:`~pagmo::r_policy`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:class:: template <typename T> is_udrp

   This type trait detects if ``T`` is a user-defined replacement policy (or UDRP).

   Specifically, the :cpp:any:`value` of this type trait will be ``true`` if:

   * ``T`` is not a reference or cv qualified,
   * ``T`` is destructible, default, copy and move constructible, and
   * ``T`` satisfies :cpp:class:`pagmo::has_replace`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:namespace-pop::
