/* Copyright 2017-2018 PaGMO development team

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

#ifndef PAGMO_IPOPT_HPP
#define PAGMO_IPOPT_HPP

#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_IPOPT)

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <IpReturnCodes.hpp>
#include <IpTypes.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/not_population_based.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

namespace detail
{

PAGMO_DLL_PUBLIC unsigned ipopt_internal_test();

}

/// Ipopt.
/**
 * \image html ipopt.png "COIN_OR logo." width=3cm
 *
 * \verbatim embed:rst:leading-asterisk
 * .. versionadded:: 2.2
 * \endverbatim
 *
 * This class is a user-defined algorithm (UDA) that wraps the Ipopt (Interior Point OPTimizer) solver,
 * a software package for large-scale nonlinear optimization. Ipopt is a powerful solver that
 * is able to handle robustly and efficiently constrained nonlinear opimization problems at high dimensionalities.
 *
 * Ipopt supports only single-objective minimisation, and it requires the availability of the gradient in the
 * optimisation problem. If possible, for best results the Hessians should be provided as well (but Ipopt
 * can estimate numerically the Hessians if needed).
 *
 * In order to support pagmo's population-based optimisation model, ipopt::evolve() will select
 * a single individual from the input pagmo::population to be optimised.
 * If the optimisation produces a better individual (as established by pagmo::compare_fc()),
 * the optimised individual will be inserted back into the population.
 * The selection and replacement strategies can be configured via set_selection(const std::string &),
 * set_selection(population::size_type), set_replacement(const std::string &) and
 * set_replacement(population::size_type).
 *
 * Configuring the optimsation run
 * -------------------------------
 *
 * Ipopt supports a large amount of options for the configuration of the optimisation run. The options
 * are divided into three categories:
 * - *string* options (i.e., the type of the option is ``std::string``),
 * - *integer* options (i.e., the type of the option is ``Ipopt::Index`` - an alias for some integer type, typically
 *   ``int``),
 * - *numeric* options (i.e., the type of the option is ``double``).
 *
 * The full list of options is available on the
 * <a href="https://www.coin-or.org/Ipopt/documentation/node40.html">Ipopt website</a>. pagmo::ipopt allows to configure
 * any Ipopt option via methods such as ipopt::set_string_options(), ipopt::set_string_option(),
 * ipopt::set_integer_options(), etc., which need to be used before invoking ipopt::evolve().
 *
 * If the user does not set any option, pagmo::ipopt will use Ipopt's default values for the options (see the
 * <a href="https://www.coin-or.org/Ipopt/documentation/node40.html">documentation</a>), with the following
 * modifications:
 * - if the ``"print_level"`` integer option is **not** set by the user, it will be set to 0 by pagmo::ipopt (this will
 *   suppress most screen output produced by the solver - note that we support an alternative form of logging via
 *   the ipopt::set_verbosity() method);
 * - if the ``"hessian_approximation"`` string option is **not** set by the user and the optimisation problem does
 *   **not** provide the Hessians, then the option will be set to ``"limited-memory"`` by pagmo::ipopt. This makes it
 *   possible to optimise problems without Hessians out-of-the-box (i.e., Ipopt will approximate numerically the
 *   Hessians for you);
 * - if the ``"constr_viol_tol"`` numeric option is **not** set by the user and the optimisation problem is constrained,
 *   then pagmo::ipopt will compute the minimum value ``min_tol`` in the vector returned by pagmo::problem::get_c_tol()
 *   for the optimisation problem at hand. If ``min_tol`` is nonzero, then the ``"constr_viol_tol"`` Ipopt option will
 *   be set to ``min_tol``, otherwise the default Ipopt value (1E-4) will be used for the option. This ensures that,
 *   if the constraint tolerance is not explicitly set by the user, a solution deemed feasible by Ipopt is also
 *   deemed feasible by pagmo (but the opposite is not necessarily true).
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
 *
 *    A moved-from :cpp:class:`pagmo::ipopt` is destructible and assignable. Any other operation will result
 *    in undefined behaviour.
 *
 * .. note::
 *
 *    This user-defined algorithm is available only if pagmo was compiled with the ``PAGMO_WITH_IPOPT`` option
 *    enabled (see the :ref:`installation instructions <install>`).
 *
 * .. seealso::
 *
 *    https://projects.coin-or.org/Ipopt.
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC ipopt : public not_population_based
{
    template <typename Pair>
    static void opt_checker(bool status, const Pair &p, const std::string &op_type)
    {
        if (!status) {
            pagmo_throw(std::invalid_argument, "failed to set the ipopt " + op_type + " option '" + p.first
                                                   + "' to the value: " + detail::to_string(p.second));
        }
    }

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
     * The algorithm log is a collection of ipopt::log_line_type data lines, stored in chronological order
     * during the optimisation if the verbosity of the algorithm is set to a nonzero value
     * (see ipopt::set_verbosity()).
     */
    using log_type = std::vector<log_line_type>;

    // Evolve population.
    population evolve(population) const;

    /// Get the result of the last optimisation.
    /**
     * @return the result of the last evolve() call, or ``Ipopt::Solve_Succeeded`` if no optimisations have been
     * run yet.
     */
    Ipopt::ApplicationReturnStatus get_last_opt_result() const
    {
        return m_last_opt_res;
    }

    /// Get the algorithm's name.
    /**
     * @return <tt>"Ipopt"</tt>.
     */
    std::string get_name() const
    {
        return "Ipopt: Interior Point Optimization";
    }

    // Get extra information about the algorithm.
    std::string get_extra_info() const;

    /// Set verbosity.
    /**
     * This method will set the algorithm's verbosity. If \p n is zero, no output is produced during the optimisation
     * and no logging is performed. If \p n is nonzero, then every \p n objective function evaluations the status
     * of the optimisation will be both printed to screen and recorded internally. See ipopt::log_line_type and
     * ipopt::log_type for information on the logging format. The internal log can be fetched via get_log().
     *
     * Example (verbosity 1):
     * @code{.unparsed}
     * objevals:        objval:      violated:    viol. norm:
     *         1        48.9451              1        1.25272 i
     *         2         30.153              1       0.716591 i
     *         3        26.2884              1        1.04269 i
     *         4        14.6958              2        7.80753 i
     *         5        14.7742              2        5.41342 i
     *         6         17.093              1      0.0905025 i
     *         7        17.1772              1      0.0158448 i
     *         8        17.0254              2      0.0261289 i
     *         9        17.0162              2     0.00435195 i
     *        10        17.0142              2    0.000188461 i
     *        11         17.014              1    1.90997e-07 i
     *        12         17.014              0              0
     * @endcode
     * The ``i`` at the end of some rows indicates that the decision vector is infeasible. Feasibility
     * is checked against the problem's tolerance.
     *
     * By default, the verbosity level is zero.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    The number of constraints violated, the constraints violation norm and the feasibility flag stored in the log
     *    are all determined via the facilities and the tolerances specified within :cpp:class:`pagmo::problem`. That
     *    is, they might not necessarily be consistent with Ipopt's notion of feasibility. See the explanation
     *    of how the ``"constr_viol_tol"`` numeric option is handled in :cpp:class:`pagmo::ipopt`.
     *
     * .. note::
     *
     *    Ipopt supports its own logging format and protocol, including the ability to print to screen and write to
     *    file. Ipopt's screen logging is disabled by default (i.e., the Ipopt verbosity setting is set to 0 - see
     *    :cpp:class:`pagmo::ipopt`). On-screen logging can be enabled via the ``"print_level"`` string option.
     *
     * \endverbatim
     *
     * @param n the desired verbosity level.
     */
    void set_verbosity(unsigned n)
    {
        m_verbosity = n;
    }

    /// Get the optimisation log.
    /**
     * See ipopt::log_type for a description of the optimisation log. Logging is turned on/off via
     * set_verbosity().
     *
     * @return a const reference to the log.
     */
    const log_type &get_log() const
    {
        return m_log;
    }

    // Save to archive.
    template <typename Archive>
    void save(Archive &, unsigned) const;

    // Load from archive.
    template <typename Archive>
    void load(Archive &, unsigned);

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // Set string option.
    void set_string_option(const std::string &, const std::string &);

    // Set integer option.
    void set_integer_option(const std::string &, Ipopt::Index);

    // Set numeric option.
    void set_numeric_option(const std::string &, double);

    // Set string options.
    void set_string_options(const std::map<std::string, std::string> &);

    // Set integer options.
    void set_integer_options(const std::map<std::string, Ipopt::Index> &);

    // Set numeric options.
    void set_numeric_options(const std::map<std::string, double> &);

    // Get string options.
    std::map<std::string, std::string> get_string_options() const;

    // Get integer options.
    std::map<std::string, Ipopt::Index> get_integer_options() const;

    // Get numeric options.
    std::map<std::string, double> get_numeric_options() const;

    // Clear all string options.
    void reset_string_options();

    // Clear all integer options.
    void reset_integer_options();

    /// Clear all numeric options.
    void reset_numeric_options();

    /// Thread safety level.
    /**
     * According to the official Ipopt documentation, it is not safe to use Ipopt in a multithreaded environment.
     *
     * @return thread_safety::none.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. seealso::
     *    https://projects.coin-or.org/Ipopt/wiki/FAQ
     *
     * \endverbatim
     */
    thread_safety get_thread_safety() const
    {
        return thread_safety::none;
    }

private:
    // Options maps.
    std::map<std::string, std::string> m_string_opts;
    std::map<std::string, Ipopt::Index> m_integer_opts;
    std::map<std::string, double> m_numeric_opts;
    // Solver return status.
    mutable Ipopt::ApplicationReturnStatus m_last_opt_res = Ipopt::Solve_Succeeded;
    // Verbosity/log.
    unsigned m_verbosity = 0;
    mutable log_type m_log;
};
} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::ipopt)

#else // PAGMO_WITH_IPOPT

#error The ipopt.hpp header was included, but pagmo was not compiled with Ipopt support

#endif // PAGMO_WITH_IPOPT

#endif
