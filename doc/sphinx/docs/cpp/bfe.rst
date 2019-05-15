Batch fitness evaluator
=======================

.. versionadded:: 2.11

*#include <pagmo/bfe.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: bfe

   This class implements the evaluation of decision vectors in batch mode. That is,
   whereas a :cpp:class:`pagmo::problem` provides the means to evaluate a single decision
   vector via the :cpp:func:`pagmo::problem::fitness()` member function, a
   :cpp:class:`~pagmo::bfe` (short for *batch fitness evaluator*) enables a :cpp:class:`~pagmo::problem`
   to evaluate the fitnesses of a group (or a *batch*) of decision vectors, possibly
   in a parallel/vectorised fashion.

   Together with the :cpp:func:`pagmo::problem::batch_fitness()` member function,
   :cpp:class:`~pagmo::bfe` is one of the mechanisms provided
   by pagmo to enable a form of parallelism on a finer level than the
   :cpp:class:`~pagmo::archipelago` and :cpp:class:`~pagmo::island` classes.
   However, while the :cpp:func:`pagmo::problem::batch_fitness()` member function must be
   implemented on a UDP-by-UDP basis, a :cpp:class:`~pagmo::bfe`
   provides generic batch fitness evaluation capabilities for any :cpp:class:`~pagmo::problem`,
   and it can thus be used also with UDPs which do not implement the
   :cpp:func:`pagmo::problem::batch_fitness()` member function.

   Like :cpp:class:`~pagmo::problem`, :cpp:class:`~pagmo::algorithm`, and many other
   pagmo classes, :cpp:class:`~pagmo::bfe` is a generic container
   (or, in the parlance of C++, a *type-erased* class) which stores internally
   a user-defined batch fitness evaluator (UDBFE for short) which actually
   implements the fitness evaluation in batch mode. Users are free to either
   use one of the evaluators provided with pagmo, or to write their own UDBFE.

   Every UDBFE must be a callable with a signature equivalent to

   .. code-block:: c++

      vector_double (const problem &, const vector_double &)

   UDBFEs receive in input a :cpp:class:`~pagmo::problem` and a batch of decision vectors
   stored contiguously in a :cpp:type:`~pagmo::vector_double`, and they return
   a :cpp:type:`~pagmo::vector_double` containing the fitness vectors
   corresponding to the input batch of decision vectors (as evaluated by the input problem and
   stored contiguously).

   Additionally, UDBFEs must also be destructible and default, copy and move constructible.
   Note that pointers to plain C++ functions with an appropriate signature
   are UDBFEs, but lambda functions are not (as they currently are not default-constructible).

   UDBFEs can also implement the following (optional) member functions:

   .. code-block:: c++

      std::string get_name() const;
      std::string get_extra_info() const;
      thread_safety get_thread_safety() const;

   See the documentation of the corresponding member functions in this class for details on how the optional
   member functions in the UDBFE are used by :cpp:class:`~pagmo::bfe`.

   .. warning::

      The only operations allowed on a moved-from :cpp:class:`pagmo::bfe` are destruction,
      assignment, and the invocation of the :cpp:func:`~pagmo::bfe::is_valid()` member function.
      Any other operation will result in undefined behaviour.

   .. cpp:function:: bfe()

      Default constructor.

      The default constructor will initialize a :cpp:class:`~pagmo::bfe` containing a :cpp:class:`~pagmo::default_bfe`.

      :exception unspecified: any exception raised by the constructor from a generic UDBFE.

   .. cpp:function:: bfe(const bfe &)
   .. cpp:function:: bfe(bfe &&) noexcept
   .. cpp:function:: bfe &operator=(const bfe &)
   .. cpp:function:: bfe &operator=(bfe &&) noexcept

      :cpp:class:`~pagmo::bfe` is copy/move constructible, and copy/move assignable.
      Copy construction/assignment will perform deep copies, move operations will leave the moved-from object in
      a state which is destructible and assignable.

      :exception unspecified: when performing copy operations, any exception raised by the UDBFE upon copying, or by memory allocation failures.

   .. cpp:function:: template <typename T> explicit bfe(T &&x)

      Generic constructor from a UDBFE.

      This constructor participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::bfe` and if it satisfies :cpp:class:`pagmo::is_udbfe`.

      Additionally, the constructor will also be enabled if ``T``, after the removal of reference and cv qualifiers, is a function type with
      the following signature

      .. code-block:: c++

         vector_double (const problem &, const vector_double &)

      The input parameter *x* will be perfectly forwarded to construct the internal UDBFE instance.

      :param x: the input UDBFE.

      :exception unspecified: any exception thrown by the public API of the UDBFE, or by memory allocation failures.

   .. cpp:function:: template <typename T> bfe &operator=(T &&x)

      Generic assignment operator from a UDBFE.

      This operator participates in overload resolution only if ``T``, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::bfe` and if it satisfies :cpp:class:`pagmo::is_udbfe`.

      Additionally, the operator will also be enabled if ``T``, after the removal of reference and cv qualifiers,
      is a function type with the following signature

      .. code-block:: c++

         vector_double (const problem &, const vector_double &)

      This operator will set the internal UDBFE to *x* by constructing a :cpp:class:`~pagmo::bfe` from *x*,
      and then move-assigning the result to *this*.

      :param x: the input UDBFE.

      :return: a reference to *this*.

      :exception unspecified: any exception thrown by the generic constructor from a UDBFE.

   .. cpp:function:: template <typename T> const T *extract() const noexcept
   .. cpp:function:: template <typename T> T *extract() noexcept

      Extract a (const) pointer to the internal UDBFE instance.

      If ``T`` is the type of the UDBFE currently stored within this object, then this function
      will return a (const) pointer to the internal UDBFE instance. Otherwise, ``nullptr`` will be returned.

      The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
      of ``this``, and ``delete`` must never be called on the pointer.

      .. warning::

         The non-const overload of this function is provided only in order to allow to call non-const
         member functions on the internal UDBFE instance. Assigning a new UDBFE via pointers obtained
         through this function is undefined behaviour.

      :return: a (const) pointer to the internal UDBFE instance, or ``nullptr``.

   .. cpp:function:: template <typename T> bool is() const noexcept

      Check the type of the UDBFE.

      :return: ``true`` if ``T`` is the type of the UDBFE currently stored within this object, ``false`` otherwise.

   .. cpp:function:: vector_double operator()(const problem &p, const vector_double &dvs) const

      Call operator.

      The call operator will invoke the internal UDBFE instance to perform the evaluation in batch mode
      of the decision vectors stored in *dvs* using the input problem *p*, and it will return the corresponding
      fitness vectors.

      The input decision vectors must be stored contiguously in *dvs*: for a problem with dimension :math:`n`, the first
      decision vector in *dvs* occupies the index range :math:`\left[0, n\right)`, the second decision vector
      occupies the range :math:`\left[n, 2n\right)`, and so on. Similarly, the output fitness vectors must be
      laid out contiguously in the return value: for a problem with fitness dimension :math:`f`, the first fitness
      vector will occupy the index range :math:`\left[0, f\right)`, the second fitness vector
      will occupy the range :math:`\left[f, 2f\right)`, and so on.

      This function will perform a variety of sanity checks on both *dvs* and on the return value.

      :param p: the input :cpp:class:`~pagmo::problem`.
      :param dvs: the input decision vectors that will be evaluated in batch mode.

      :return: the fitness vectors corresponding to the input decision vectors in *dvs*.

      :exception std\:\:invalid_argument: if *dvs* or the return value produced by the UDBFE are incompatible with the input problem *p*.
      :exception unspecified: any exception raised by the invocation of the UDBFE.

   .. cpp:function:: std::string get_name() const

      Get the name of this batch fitness evaluator.

      If the UDBFE satisfies :cpp:class:`pagmo::has_name`, then this member function will return the output of its ``get_name()`` member function.
      Otherwise, an implementation-defined name based on the type of the UDBFE will be returned.

      :return: the name of this batch fitness evaluator.

      :exception unspecified: any exception thrown by copying an ``std::string`` object.

   .. cpp:function:: std::string get_extra_info() const

      Extra info for this batch fitness evaluator.

      If the UDBFE satisfies :cpp:class:`pagmo::has_extra_info`, then this member function will return the output of its
      ``get_extra_info()`` member function. Otherwise, an empty string will be returned.

      :return: extra info about the UDBFE.

      :exception unspecified: any exception thrown by the ``get_extra_info()`` member function of the UDBFE, or by copying an ``std::string`` object.

   .. cpp:function:: thread_safety get_thread_safety() const

      Thread safety level of this batch fitness evaluator.

      If the UDBFE satisfies :cpp:class:`pagmo::has_get_thread_safety`, then this member function will return the output of its
      ``get_thread_safety()`` member function. Otherwise, :cpp:enumerator:`pagmo::thread_safety::basic` will be returned.
      That is, pagmo assumes by default that is it safe to operate concurrently on distinct UDBFE instances.

      :return: the thread safety level of the UDBFE.

   .. cpp:function:: bool is_valid() const

      Check if this bfe is in a valid state.

      :return: ``false`` if *this* was moved from, ``true`` otherwise.

   .. cpp:function:: template <typename Archive> void save(Archive &ar, unsigned) const
   .. cpp:function:: template <typename Archive> void load(Archive &ar, unsigned)

      Serialisation support.

      These two member functions are used to implement the (de)serialisation of an evaluator to/from an archive.

      :param ar: the input/output archive.

      :exception unspecified: any exception raised by the (de)serialisation of primitive types or of the UDBFE.

Functions
---------

.. cpp:function:: std::ostream &operator<<(std::ostream &os, const bfe &b)

   Stream insertion operator.

   This function will direct to *os* a human-readable representation of the input
   :cpp:class:`~pagmo::bfe` *b*.

   :param os: the input ``std::ostream``.
   :param b: the batch fitness evaluator that will be directed to *os*.

   :return: a reference to *os*.

   :exception unspecified: any exception thrown by querying various properties of the evaluator and directing them to *os*.

Associated type traits
----------------------

.. cpp:class:: template <typename T> has_bfe_call_operator

   This type trait detects if ``T`` is a callable whose signature is compatible with the one
   required by :cpp:class:`~pagmo::bfe`.

   Specifically, the :cpp:any:`value` of this type trait will be ``true`` if the expression
   ``B(p, dvs)``, where

   * ``B`` is a const reference to an instance of ``T``,
   * ``p`` is a const reference to a :cpp:class:`~pagmo::problem`, and
   * ``dvs`` is a const reference to a :cpp:type:`~pagmo::vector_double`,

   is well-formed and if it returns a :cpp:type:`~pagmo::vector_double`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:class:: template <typename T> is_udbfe

   This type trait detects if ``T`` is a user-defined batch fitness evaluator (or UDBFE).

   Specifically, the :cpp:any:`value` of this type trait will be ``true`` if:

   * ``T`` is not a reference or cv qualified,
   * ``T`` is destructible, default, copy and move constructible, and
   * ``T`` satisfies :cpp:class:`pagmo::has_bfe_call_operator`.

   .. cpp:member:: static const bool value

      The value of the type trait.

.. cpp:namespace-pop::
