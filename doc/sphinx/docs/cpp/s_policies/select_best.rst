Best selection policy
=====================

.. versionadded:: 2.11

*#include <pagmo/s_policies/select_best.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: select_best

   This user-defined selection policy (UDSP) will select the *best*
   individuals from a group.

   In this context, *best* means the following:

   * in single-objective unconstrained problems, individuals are ranked
     according to their fitness function,
   * in single-objective constrained problems, individuals are ranked
     via :cpp:func:`~pagmo::sort_population_con()`,
   * in multi-objective unconstrained problems, individuals are ranked
     via :cpp:func:`~pagmo::sort_population_mo()`.

   See the documentation of :cpp:func:`~pagmo::select_best::select()` for
   more details on the selection algorithm implemented by this UDSP.

   Note that this user-defined selection policy currently does *not* support
   multi-objective constrained problems.

   .. cpp:function:: select_best()

      Default constructor.

      The default constructor initialises a policy with an absolute migration rate
      of 1 (that is, 1 individual will be selected from the input population).

   .. cpp:function:: template <typename T> explicit select_best(T x)

      Constructor from a migration rate.

      This constructor participates in overload resolution only if ``T`` is a C++
      integral or a floating-point type. The input migration rate, *x*, is used to indicate
      how many individuals will be selected from an input population by the
      :cpp:func:`~pagmo::select_best::select()` member function.

      If *x* is a floating point value in the :math:`\left[0,1\right]` range,
      then it represents a *fractional* migration rate. That is, it indicates,
      the fraction of individuals that will be selected from the input population:
      a value of 0 means that no individuals will be selected, a value of 1 means that
      all individuals will be selected.

      If *x* is an integral value, then it represents an *absolute* migration rate, that is,
      the exact number of individuals that will be selected from the input population.

      :param x: the fractional or absolute migration rate.

      :exception std\:\:invalid_argument: if the supplied fractional migration rate is not finite
         or not in the :math:`\left[0,1\right]` range.
      :exception unspecified: any exception raised by ``boost::numeric_cast()`` while trying
         to convert the input absolute migration rate to :cpp:type:`~pagmo::pop_size_t`.

   .. cpp:function:: individuals_group_t select(const individuals_group_t &inds, const vector_double::size_type &, \
                                                const vector_double::size_type &, const vector_double::size_type &nobj, \
                                                const vector_double::size_type &nec, const vector_double::size_type &nic, \
                                                const vector_double &tol) const

      This member function will select individuals from *inds*.

      The selection algorithm determines first how many individuals in *inds* will be selected. This depends both on
      the migration rate specified upon construction, and on the size of *inds*.

      After having established the number :math:`N` of individuals to be selected from *inds*,
      the algorithm then ranks the individuals in *inds* and selects the top :math:`N` individuals.
      The ranking method depends on the problem's properties:

      * in single-objective unconstrained problems, the individuals are ranked according to their
        (scalar) fitnesses,
      * in single-objective constrained problems, the ranking of individuals
        is done via :cpp:func:`~pagmo::sort_population_con()`,
      * in multi-objective unconstrained problems, the ranking of individuals
        is done via :cpp:func:`~pagmo::sort_population_mo()`.

      Note that this user-defined selection policy currently does *not* support
      multi-objective constrained problems.

      :param inds: the input individuals.
      :param nobj: the number of objectives of the problem the individuals in *inds* refer to.
      :param nec: the number of equality constraints of the problem the individuals in *inds* refer to.
      :param nic: the number of inequality constraints of the problem the individuals in *inds* refer to.
      :param tol: the vector of constraint tolerances of the problem the individuals in *inds* refer to.

      :return: the group of top :math:`N` individuals from *inds*.

      :exception std\:\:invalid_argument: in the following cases:

         * the problem the individuals in *inds* refer to is
           multi-objective and constrained,
         * an absolute migration rate larger than the number of input individuals
           was specified.

      :exception unspecified: any exception raised by one of the invoked ranking functions or by memory
         allocation errors in standard containers.

   .. cpp:function:: std::string get_name() const

      Get the name of the policy.

      :return: ``"Select best"``.

   .. cpp:function:: std::string get_extra_info() const

      :return: Human-readable extra info about this selection policy.

   .. cpp:function:: template <typename Archive> void serialize(Archive &ar, unsigned)

      Serialisation support.

      This member function is used to implement the (de)serialisation of this selection policy to/from an archive.

      :param ar: the input/output archive.

      :exception unspecified: any exception raised by the (de)serialisation of primitive types.

.. cpp:namespace-pop::
