Batch fitness evaluator
=======================

.. cpp:namespace-push:: pagmo

.. cpp:class:: batch_fitness_evaluator

   This class implements the evaluation of decision vectors in batch mode. That is,
   whereas a :cpp:class:`pagmo::problem` provides the means to evaluate a single decision
   vector via the :cpp:func:`pagmo::problem::fitness()` member function, a
   :cpp:class:`~pagmo::batch_fitness_evaluator` enables a :cpp:class:`~pagmo::problem`
   to evaluate the fitnesses of a group (or a *batch*) of decision vectors, possibly
   in a parallel/vectorised fashion.

   Together with the :cpp:func:`pagmo::problem::batch_fitness()` member function,
   :cpp:class:`~pagmo::batch_fitness_evaluator` is one of the mechanisms provided
   by pagmo to enable a form of parallelism on a finer level than the
   :cpp:class:`~pagmo::archipelago` and :cpp:class:`~pagmo::island` classes.
   However, while the :cpp:func:`pagmo::problem::batch_fitness()` member function must be
   implemented on a UDP-by-UDP basis, a :cpp:class:`~pagmo::batch_fitness_evaluator`
   provides generic batch fitness evaluation capabilities for any :cpp:class:`~pagmo::problem`,
   and it can thus be used also with UDPs which do not implement the
   :cpp:func:`pagmo::problem::batch_fitness()` member function.

   Like :cpp:class:`~pagmo::problem`, :cpp:class:`~pagmo::algorithm`, and many other
   pagmo classes, :cpp:class:`~pagmo::batch_fitness_evaluator` is a generic container
   (or, in the parlance of C++, a *type-erased* class) which stores internally
   a user-defined batch fitness evaluator (or, UDBFE for short) which actually
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
      thread_safety get_thread_safety() const;

   See the documentation of the corresponding methods in this class for details on how the optional
   methods in the UDBFE are used by :cpp:class:`~pagmo::batch_fitness_evaluator`.

   .. warning::

      A moved-from :cpp:class:`~pagmo::batch_fitness_evaluator` is destructible and assignable. Any other operation will result
      in undefined behaviour.

   .. cpp:function:: batch_fitness_evaluator()

      Default constructor.

      The default constructor will initialize a :cpp:class:`~pagmo::batch_fitness_evaluator` containing a :cpp:class:`~pagmo::default_bfe`.

      :exception unspecified: any exception raised by the constructor from a generic UDBFE.

   .. cpp:function:: batch_fitness_evaluator(const batch_fitness_evaluator &)
   .. cpp:function:: batch_fitness_evaluator(batch_fitness_evaluator &&) noexcept
   .. cpp:function:: batch_fitness_evaluator &operator=(const batch_fitness_evaluator &)
   .. cpp:function:: batch_fitness_evaluator &operator=(batch_fitness_evaluator &&) noexcept

      :cpp:class:`~pagmo::batch_fitness_evaluator` is copy/move constructible, and copy/move assignable.
      Copy construction/assignment will perform deep copies, move operations will leave the moved-from object in
      a state which is destructible and assignable.

      :exception unspecified: when performing copy operations, any exception raised by the UDBFE upon copying, or by memory allocation failures.

   .. cpp:function:: template <typename T> explicit batch_fitness_evaluator(T &&x)

      Generic constructor from a UDBFE.

      This constructor participates in overload resolution only if :cpp:type:`T`, after the removal of reference
      and cv qualifiers, is not :cpp:class:`~pagmo::batch_fitness_evaluator` and if it satisfies :cpp:class:`pagmo::is_udbfe`.

      This constructor will construct a :cpp:class:`~pagmo::batch_fitness_evaluator` from the UDBFE :cpp:any:`x`
      of type :cpp:type:`T`. :cpp:any:`x` will be perfectly forwarded to construct the internal UDBFE.

      :param x: the input UDBFE.

      :exception unspecified: any exception thrown by the public API of the UDBFE, or by memory allocation failures.

   .. cpp:function:: template <typename T> const T *extract() const noexcept
   .. cpp:function:: template <typename T> T *extract() noexcept

      Extract a (const) pointer to the internal UDBFE instance.



.. cpp:namespace-pop::
