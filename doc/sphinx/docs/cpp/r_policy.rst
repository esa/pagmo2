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
   :cpp:func:`~pagmo::topology::replace()` member function might be invoked
   concurrently with any other member function. It is up to the authors of user-defined
   replacement policies to ensure that this safety requirement is satisfied.

   .. warning::

      The only operations allowed on a moved-from :cpp:class:`pagmo::r_policy` are destruction,
      assignment, and the invocation of the :cpp:func:`~pagmo::r_policy::is_valid()` member function.
      Any other operation will result in undefined behaviour.

.. cpp:namespace-pop::
