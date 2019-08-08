Fair replacement policy
=======================

.. versionadded:: 2.11

*#include <pagmo/r_policies/fair_replace.hpp>*

.. cpp:namespace-push:: pagmo

.. cpp:class:: fair_replace

   This user-defined replacement policy (UDRP) will replace individuals in
   a group only if the candidate replacement individuals are *better* than
   the original individuals.

   In this context, *better* means the following:

   * in single-objective unconstrained problems, an individual is better
     than another one if its fitness is lower,
   * in single-objective constrained problems, individuals are ranked
     via :cpp:func:`~pagmo::sort_population_con()`,
   * in multi-objective unconstrained problems, individuals are ranked
     via :cpp:func:`~pagmo::sort_population_mo()`.

   See the documentation of :cpp:func:`~pagmo::fair_replace::replace()` for
   more details on the replacement algorithm implemented by this UDRP.

   Note that this user-defined replacement policy currently does *not* support
   multi-objective constrained problems.

   .. cpp:function:: fair_replace()

      Default constructor.

      The default constructor initialises a policy with an absolute migration rate of 1
      (that is, 1 individual in the original population is considered for replacement).

   .. cpp:function:: template <typename T> explicit fair_replace(T x)

      Constructor from a migration rate.

      This constructor participates in overload resolution only if ``T`` is a C++
      integral or a floating-point type. The input migration rate, *x*, is used to indicate
      how many individuals will be replaced in an input population by the
      :cpp:func:`~pagmo::fair_replace::replace()` member function.

      If *x* is a floating point value in the :math:`\left[0,1\right]` range,
      then it represents a *fractional* migration rate. That is, it indicates,
      the fraction of individuals that may be replaced in the input population:
      a value of 0 means that no individuals will be replaced, a value of 1 means that
      all individuals may be replaced.

      If *x* is an integral value, then it represents an *absolute* migration rate, that is,
      the exact number of individuals that may be replaced in the input population.

      :param x: the fractional or absolute migration rate.

      :exception std\:\:invalid_argument: if the supplied fractional migration rate is not finite
         or not in the :math:`\left[0,1\right]` range.
      :exception unspecified: any exception raised by ``boost::numeric_cast()`` while trying
         to convert the input absolute migration rate to :cpp:type:`~pagmo::pop_size_t`.

   .. cpp:function:: individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &, \
                                                 const vector_double::size_type &, const vector_double::size_type &nobj, \
                                                 const vector_double::size_type &nec, const vector_double::size_type &nic, \
                                                 const vector_double &tol, const individuals_group_t &mig) const

      This member function will replace individuals in *inds* with individuals from *mig*.

      The replacement algorithm determines first how many individuals in *inds* can be replaced. This depends both on
      the migration rate specified upon construction, and on the size :math:`S` of *inds*.

      After having established the number :math:`N` of individuals that can be replaced in *inds*,
      the algorithm then selects the top :math:`N` individuals from *mig*, merges them
      with *inds* into a new population, and returns the top :math:`S` individuals
      from the new population. The ranking of individuals in *mig* and in the new population
      depends on the problem's properties:

      * in single-objective unconstrained problems, the individuals are ranked according to their
        (scalar) fitnesses,
      * in single-objective constrained problems, the ranking of individuals
        is done via :cpp:func:`~pagmo::sort_population_con()`,
      * in multi-objective unconstrained problems, the ranking of individuals
        is done via :cpp:func:`~pagmo::sort_population_mo()`.

      Note that this user-defined replacement policy currently does *not* support
      multi-objective constrained problems.

      :param inds: the input individuals.
      :param nobj: the number of objectives of the problem the individuals in *inds* and *mig* refer to.
      :param nec: the number of equality constraints of the problem the individuals in *inds* and *mig* refer to.
      :param nic: the number of inequality constraints of the problem the individuals in *inds* and *mig* refer to.
      :param tol: the vector of constraint tolerances of the problem the individuals in *inds* and *mig* refer to.
      :param mig: the individuals that may replace individuals in *inds*.

      :return: the new population resulting from replacing individuals in *inds* with individuals from *mig*.

      :exception std\:\:invalid_argument: in the following cases:

         * the problem the individuals in *inds* and *mig* refer to is
           multi-objective and constrained,
         * an absolute migration rate larger than the number of input individuals
           was specified.

      :exception unspecified: any exception raised by one of the invoked ranking functions or by memory
         allocation errors in standard containers.

   .. cpp:function:: std::string get_name() const

      Get the name of the policy.

      :return: ``"Fair replace"``.

   .. cpp:function:: std::string get_extra_info() const

      :return: Human-readable extra info about this replacement policy.

   .. cpp:function:: template <typename Archive> void serialize(Archive &ar, unsigned)

      Serialisation support.

      This member function is used to implement the (de)serialisation of this replacement policy to/from an archive.

      :param ar: the input/output archive.

      :exception unspecified: any exception raised by the (de)serialisation of primitive types.

.. cpp:namespace-pop::
