/* Copyright 2017-2020 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#ifndef PAGMO_ALGORITHMS_NLOPT_HPP
#define PAGMO_ALGORITHMS_NLOPT_HPP

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_NLOPT)

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <nlopt.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// NLopt algorithms.
/**
 * \image html nlopt.png "NLopt logo." width=3cm
 *
 * This user-defined algorithm wraps a selection of solvers from the <a
 * href="https://nlopt.readthedocs.io/en/latest/">NLopt</a> library, focusing on
 * local optimisation (both gradient-based and derivative-free). The complete list of supported
 * NLopt algorithms is:
 * - COBYLA,
 * - BOBYQA,
 * - NEWUOA + bound constraints,
 * - PRAXIS,
 * - Nelder-Mead simplex,
 * - sbplx,
 * - MMA (Method of Moving Asymptotes),
 * - CCSA,
 * - SLSQP,
 * - low-storage BFGS,
 * - preconditioned truncated Newton,
 * - shifted limited-memory variable-metric,
 * - augmented Lagrangian algorithm.
 *
 * The desired NLopt solver is selected upon construction of a pagmo::nlopt algorithm. Various properties
 * of the solver (e.g., the stopping criteria) can be configured after construction via methods provided
 * by this class. Multiple stopping criteria can be active at the same time: the optimisation will
 * stop as soon as at least one stopping criterion is satisfied. By default, only the ``xtol_rel`` stopping
 * criterion is active (see get_xtol_rel()).
 *
 * All NLopt solvers support only single-objective optimisation, and, as usual in pagmo, minimisation
 * is always assumed. The gradient-based algorithms require the optimisation problem to provide a gradient.
 * Some solvers support equality and/or inequality constraints. The constraints' tolerances will
 * be set to those specified in the pagmo::problem being optimised (see pagmo::problem::set_c_tol()).
 *
 * In order to support pagmo's population-based optimisation model, nlopt::evolve() will select
 * a single individual from the input pagmo::population to be optimised by the NLopt solver.
 * If the optimisation produces a better individual (as established by pagmo::compare_fc()),
 * the optimised individual will be inserted back into the population.
 * The selection and replacement strategies can be configured via set_selection(const std::string &),
 * set_selection(population::size_type), set_replacement(const std::string &) and
 * set_replacement(population::size_type).
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from :cpp:class:`pagmo::nlopt` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * .. note::
 *
 *    This user-defined algorithm is available only if pagmo was compiled with the ``PAGMO_WITH_NLOPT`` option
 *    enabled (see the :ref:`installation instructions <install>`).
 *
 * .. seealso::
 *
 *    The `NLopt website <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`__ contains a detailed description
 *    of each supported solver.
 *
 * \endverbatim
 */
// NOTE:
// - investigate the use of a fitness cache, after we have good perf testing in place.
// - it looks like those lower_bound calls in the C objfun/constr computations related
//   to the gradient sparsity pattern can be done once on construction, instead of
//   repeatedly at every objfun.
class PAGMO_DLL_PUBLIC nlopt : public not_population_based
{
public:
    /// Single data line for the algorithm's log.
    /**
     * A log data line is a tuple consisting of:
     * - the number of objective function evaluations made so far,
     * - the objective function value for the current decision vector,
     * - the number of constraints violated by the current decision vector,
     * - the constraints violation norm for the current decision vector,
     * - a boolean flag signalling the feasibility of the current decision vector.
     */
    using log_line_type = std::tuple<unsigned long, double, vector_double::size_type, double, bool>;
    /// Log type.
    /**
     * The algorithm log is a collection of nlopt::log_line_type data lines, stored in chronological order
     * during the optimisation if the verbosity of the algorithm is set to a nonzero value
     * (see nlopt::set_verbosity()).
     */
    using log_type = std::vector<log_line_type>;

    // Default constructor.
    nlopt();

    // Constructor from solver name.
    explicit nlopt(const std::string &);

    // Copy constructor.
    nlopt(const nlopt &);

    /// Move constructor.
    nlopt(nlopt &&) = default;

    /// Move assignment operator.
    /**
     * @return a reference to \p this.
     */
    nlopt &operator=(nlopt &&) = default;

    // Evolve population.
    population evolve(population) const;

    // Algorithm's name.
    std::string get_name() const;

    /// Set verbosity.
    /**
     * This method will set the algorithm's verbosity. If \p n is zero, no output is produced during the optimisation
     * and no logging is performed. If \p n is nonzero, then every \p n objective function evaluations the status
     * of the optimisation will be both printed to screen and recorded internally. See nlopt::log_line_type and
     * nlopt::log_type for information on the logging format. The internal log can be fetched via get_log().
     *
     * Example (verbosity 5):
     * @code{.unparsed}
     * objevals:       objval:      violated:    viol. norm:
     *        1       47.9474              1        2.07944 i
     *        6       17.1986              2       0.150557 i
     *       11        17.014              0              0
     *       16        17.014              0              0
     * @endcode
     * The ``i`` at the end of some rows indicates that the decision vector is infeasible. Feasibility
     * is checked against the problem's tolerance.
     *
     * By default, the verbosity level is zero.
     *
     * @param n the desired verbosity level.
     */
    void set_verbosity(unsigned n)
    {
        m_verbosity = n;
    }

    // Get extra information about the algorithm.
    std::string get_extra_info() const;

    /// Get the optimisation log.
    /**
     * See nlopt::log_type for a description of the optimisation log. Logging is turned on/off via
     * set_verbosity().
     *
     * @return a const reference to the log.
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    /// Get the name of the solver that was used to construct this pagmo::nlopt algorithm.
    /**
     * @return the name of the NLopt solver used upon construction.
     */
    std::string get_solver_name() const
    {
        return m_algo;
    }

    /// Get the result of the last optimisation.
    /**
     * @return the result of the last evolve() call, or ``NLOPT_SUCCESS`` if no optimisations have been
     * run yet.
     */
    ::nlopt_result get_last_opt_result() const
    {
        return m_last_opt_result;
    }

    /// Get the ``stopval`` stopping criterion.
    /**
     * The ``stopval`` stopping criterion instructs the solver to stop when an objective value less than
     * or equal to ``stopval`` is found. Defaults to the C constant ``-HUGE_VAL`` (that is, this stopping criterion
     * is disabled by default).
     *
     * @return the ``stopval`` stopping criterion for this pagmo::nlopt.
     */
    double get_stopval() const
    {
        return m_sc_stopval;
    }

    // Set the ``stopval`` stopping criterion.
    void set_stopval(double);

    /// Get the ``ftol_rel`` stopping criterion.
    /**
     * The ``ftol_rel`` stopping criterion instructs the solver to stop when an optimization step (or an estimate of the
     * optimum) changes the objective function value by less than ``ftol_rel`` multiplied by the absolute value of the
     * function value. Defaults to 0 (that is, this stopping criterion is disabled by default).
     *
     * @return the ``ftol_rel`` stopping criterion for this pagmo::nlopt.
     */
    double get_ftol_rel() const
    {
        return m_sc_ftol_rel;
    }

    // Set the ``ftol_rel`` stopping criterion.
    void set_ftol_rel(double);

    /// Get the ``ftol_abs`` stopping criterion.
    /**
     * The ``ftol_abs`` stopping criterion instructs the solver to stop when an optimization step
     * (or an estimate of the optimum) changes the function value by less than ``ftol_abs``.
     * Defaults to 0 (that is, this stopping criterion is disabled by default).
     *
     * @return the ``ftol_abs`` stopping criterion for this pagmo::nlopt.
     */
    double get_ftol_abs() const
    {
        return m_sc_ftol_abs;
    }

    // Set the ``ftol_abs`` stopping criterion.
    void set_ftol_abs(double);

    /// Get the ``xtol_rel`` stopping criterion.
    /**
     * The ``xtol_rel`` stopping criterion instructs the solver to stop when an optimization step (or an estimate of the
     * optimum) changes every parameter by less than ``xtol_rel`` multiplied by the absolute value of the parameter.
     * Defaults to 1E-8.
     *
     * @return the ``xtol_rel`` stopping criterion for this pagmo::nlopt.
     */
    double get_xtol_rel() const
    {
        return m_sc_xtol_rel;
    }

    // Set the ``xtol_rel`` stopping criterion.
    void set_xtol_rel(double);

    /// Get the ``xtol_abs`` stopping criterion.
    /**
     * The ``xtol_abs`` stopping criterion instructs the solver to stop when an optimization step (or an estimate of the
     * optimum) changes every parameter by less than ``xtol_abs``.
     * Defaults to 0 (that is, this stopping criterion is disabled by default).
     *
     * @return the ``xtol_abs`` stopping criterion for this pagmo::nlopt.
     */
    double get_xtol_abs() const
    {
        return m_sc_xtol_abs;
    }

    // Set the ``xtol_abs`` stopping criterion.
    void set_xtol_abs(double);

    /// Get the ``maxeval`` stopping criterion.
    /**
     * The ``maxeval`` stopping criterion instructs the solver to stop when the number of function evaluations exceeds
     * ``maxeval``. Defaults to 0 (that is, this stopping criterion is disabled by default).
     *
     * @return the ``maxeval`` stopping criterion for this pagmo::nlopt.
     */
    int get_maxeval() const
    {
        return m_sc_maxeval;
    }

    /// Set the ``maxeval`` stopping criterion.
    /**
     * @param n the desired value for the ``maxeval`` stopping criterion (see get_maxeval()).
     */
    void set_maxeval(int n)
    {
        m_sc_maxeval = n;
    }

    /// Get the ``maxtime`` stopping criterion.
    /**
     * The ``maxtime`` stopping criterion instructs the solver to stop when the optimization time (in seconds) exceeds
     * ``maxtime``. Defaults to 0 (that is, this stopping criterion is disabled by default).
     *
     * @return the ``maxtime`` stopping criterion for this pagmo::nlopt.
     */
    int get_maxtime() const
    {
        return m_sc_maxtime;
    }

    /// Set the ``maxtime`` stopping criterion.
    /**
     * @param n the desired value for the ``maxtime`` stopping criterion (see get_maxtime()).
     */
    void set_maxtime(int n)
    {
        m_sc_maxtime = n;
    }

    // Set the local optimizer.
    void set_local_optimizer(nlopt);

    /// Get the local optimizer.
    /**
     * This method returns a raw const pointer to the local optimizer, if it has been set via set_local_optimizer().
     * Otherwise, \p nullptr will be returned.
     *
     * \verbatim embed:rst:leading-asterisk
     *
     * .. note::
     *
     *    The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
     *    of ``this``, and ``delete`` must never be called on the pointer.
     *
     * \endverbatim
     *
     * @return a const pointer to the local optimizer.
     */
    const nlopt *get_local_optimizer() const
    {
        return m_loc_opt.get();
    }

    /// Get the local optimizer.
    /**
     * This method returns a raw pointer to the local optimizer, if it has been set via set_local_optimizer().
     * Otherwise, \p nullptr will be returned.
     *
     * \verbatim embed:rst:leading-asterisk
     *
     * .. note::
     *
     *    The returned value is a raw non-owning pointer: the lifetime of the pointee is tied to the lifetime
     *    of ``this``, and ``delete`` must never be called on the pointer.
     *
     * .. note::
     *
     *    The ability to extract a mutable pointer is provided only in order to allow to call non-const
     *    methods on the local optimizer. Assigning a new local optimizer via this pointer is undefined behaviour.
     *
     * \endverbatim
     *
     * @return a pointer to the local optimizer.
     */
    nlopt *get_local_optimizer()
    {
        return m_loc_opt.get();
    }

    // Unset the local optimizer.
    void unset_local_optimizer();

    // Save to archive.
    template <typename Archive>
    void save(Archive &, unsigned) const;

    // Load from archive.
    template <typename Archive>
    void load(Archive &, unsigned);

    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    std::string m_algo;
    mutable ::nlopt_result m_last_opt_result = NLOPT_SUCCESS;
    // Stopping criteria.
    double m_sc_stopval = -HUGE_VAL;
    double m_sc_ftol_rel = 0.;
    double m_sc_ftol_abs = 0.;
    double m_sc_xtol_rel = 1E-8;
    double m_sc_xtol_abs = 0.;
    int m_sc_maxeval = 0;
    int m_sc_maxtime = 0;
    // Verbosity/log.
    unsigned m_verbosity = 0;
    mutable log_type m_log;
    // Local/subsidiary optimizer.
    std::unique_ptr<nlopt> m_loc_opt;
};
} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nlopt)

#else // PAGMO_WITH_NLOPT

#error The nlopt.hpp header was included, but pagmo was not compiled with NLopt support

#endif // PAGMO_WITH_NLOPT

#endif
