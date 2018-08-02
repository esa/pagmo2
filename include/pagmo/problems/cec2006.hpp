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

#ifndef PAGMO_PROBLEM_CEC2006_HPP
#define PAGMO_PROBLEM_CEC2006_HPP

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp> // needed for cereal registration macro
#include <pagmo/types.hpp>

// Let's disable a few compiler warnings emitted by the cec2006 code.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace pagmo
{
// forward declearing the class to allow the following definition of pointers to its methods
class cec2006;

namespace detail
{
// Usual template trick to have static members in header only libraries
// see: http://stackoverflow.com/questions/18860895/how-to-initialize-static-members-in-the-header
template <typename = void>
struct cec2006_statics {
    /// Pointer type to the methods to compute the objective and constraints
    typedef void (cec2006::*func_ptr)(vector_double &, const vector_double &) const;
    /// Problem dimension
    static const std::vector<unsigned short> m_dim;
    /// Equality constraints dimensions
    static const std::vector<unsigned short> m_nec;
    /// Inequality constraints dimension
    static const std::vector<unsigned short> m_nic;
    /// Bounds
    static const std::vector<std::pair<vector_double, vector_double>> m_bounds;
    /// Best solutions known
    static const std::vector<vector_double> m_best_known;
    /// Pointers to the member functions to be used in fitness
    static const std::vector<func_ptr> m_o_ptr;
    static const std::vector<func_ptr> m_c_ptr;
};

template <typename T>
const std::vector<unsigned short> cec2006_statics<T>::m_dim
    = {13, 20, 10, 5, 4, 2, 10, 2, 7, 8, 2, 3, 5, 10, 3, 5, 6, 9, 15, 24, 7, 22, 9, 2};

template <typename T>
const std::vector<unsigned short> cec2006_statics<T>::m_nec
    = {0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1, 0, 3, 3, 2, 0, 4, 0, 0, 14, 5, 19, 4, 0};

template <typename T>
const std::vector<unsigned short> cec2006_statics<T>::m_nic
    = {9, 2, 0, 6, 2, 2, 8, 2, 4, 6, 0, 1, 0, 0, 0, 38, 0, 13, 5, 6, 1, 1, 2, 2};

template <typename T>
const std::vector<std::pair<vector_double, vector_double>> cec2006_statics<T>::m_bounds = {

    {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, {1., 1., 1., 1., 1., 1., 1., 1., 1., 100., 100., 100., 1.}},
    {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
     {10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.}},
    {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}},
    {{78., 33., 27., 27., 27.}, {102., 45., 45., 45., 45.}},
    {{0., 0., -0.55, -0.55}, {1200., 1200., 0.55, 0.55}},
    {{13., 0.}, {100., 100.}},
    {{-10., -10., -10., -10., -10., -10., -10., -10., -10., -10.}, {10., 10., 10., 10., 10., 10., 10., 10., 10., 10.}},
    {{0., 0.}, {10., 10.}},
    {{-10., -10., -10., -10., -10., -10., -10.}, {10., 10., 10., 10., 10., 10., 10.}},
    {{100., 1000., 1000., 10., 10., 10., 10., 10.}, {10000., 10000., 10000., 1000., 1000., 1000., 1000., 1000.}},
    {{-1., -1.}, {1., 1.}},
    {{0., 0., 0.}, {10., 10., 10.}},
    {{-2.3, -2.3, -3.2, -3.2, -3.2}, {2.3, 2.3, 3.2, 3.2, 3.2}},
    {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, {10., 10., 10., 10., 10., 10., 10., 10., 10., 10.}},
    {{0., 0., 0.}, {10., 10., 10.}},
    {{704.4148, 68.6, 0., 193., 25.}, {906.3855, 288.88, 134.75, 287.0966, 84.1988}},
    {{0., 0., 340., 340., -1000., 0.}, {400., 1000., 420., 420., 1000., 0.5236}},
    {{-10., -10., -10., -10., -10., -10., -10., -10., 0.}, {10., 10., 10., 10., 10., 10., 10., 10., 20.}},
    {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
     {10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.}},
    {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
     {10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.,
      10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.}},
    {{0., 0., 0., 100., 6.3, 5.9, 4.5}, {1000., 40., 40., 300., 6.7, 6.4, 6.25}},
    {{0., 0., 0., 0., 0., 0., 0., 100., 100., 100.01, 100., 100., 0., 0., 0., 0.01, 0.01, -4.7, -4.7, -4.7, -4.7, -4.7},
     {20000., 1e6, 1e6, 1e6, 4e7,  4e7,  4e7,  299.99, 399.99, 300,  400,
      600,    500, 500, 500, 300., 400., 6.25, 6.25,   6.25,   6.25, 6.25}},
    {{0., 0., 0., 0., 0., 0., 0., 0., 0.01}, {300., 300., 100., 200., 100., 300., 100., 200., 0.03}},
    {{0., 0.}, {3., 4.}}};

template <typename T>
const std::vector<vector_double> cec2006_statics<T>::m_best_known = {
    {1., 1., 1., 1., 1., 1., 1., 1., 1., 3., 3., 3., 1.},
    {3.16246061572185, 3.12833142812967, 3.09479212988791, 3.06145059523469, 3.02792915885555,
     2.99382606701730, 2.95866871765285, 2.92184227312450, 0.49482511456933, 0.48835711005490,
     0.48231642711865, 0.47664475092742, 0.47129550835493, 0.46623099264167, 0.46142004984199,
     0.45683664767217, 0.45245876903267, 0.44826762241853, 0.44424700958760, 0.44038285956317},
    {0.31624357647283069, 0.316243577414338339, 0.316243578012345927, 0.316243575664017895, 0.316243578205526066,
     0.31624357738855069, 0.316243575472949512, 0.316243577164883938, 0.316243578155920302, 0.316243576147374916},
    {78, 33, 29.9952560256815985, 45, 36.7758129057882073},
    {679.945148297028709, 1026.06697600004691, 0.118876369094410433, -0.39623348521517826},
    {14.09500000000000064, 0.8429607892154795668},
    {2.17199634142692, 2.3636830416034, 8.77392573913157, 5.09598443745173, 0.990654756560493, 1.43057392853463,
     1.32164415364306, 9.82872576524495, 8.2800915887356, 8.3759266477347},
    {1.22797135260752599, 4.24537336612274885},
    {2.33049935147405174, 1.95137236847114592, -0.477541399510615805, 4.36572624923625874, -0.624486959100388983,
     1.03813099410962173, 1.5942266780671519},
    {579.306685017979589, 1359.97067807935605, 5109.97065743133317, 182.01769963061534, 295.601173702746792,
     217.982300369384632, 286.41652592786852, 395.601173702746735},
    {-0.707036070037170616, 0.500000004333606807},
    {5., 5., 5.},
    {-1.71714224003, 1.59572124049468, 1.8272502406271, -0.763659881912867, -0.76365986736498},
    {0.0406684113216282, 0.147721240492452, 0.783205732104114, 0.00141433931889084, 0.485293636780388,
     0.000693183051556082, 0.0274052040687766, 0.0179509660214818, 0.0373268186859717, 0.0968844604336845},
    {3.51212812611795133, 0.216987510429556135, 3.55217854929179921},
    {705.174537070090537, 68.5999999999999943, 102.899999999999991, 282.324931593660324, 37.5841164258054832},
    {201.784467214523659, 99.9999999999999005, 383.071034852773266, 420, -10.9076584514292652, 0.0731482312084287128},
    {-0.657776192427943163, -0.153418773482438542, 0.323413871675240938, -0.946257611651304398, -0.657776194376798906,
     -0.753213434632691414, 0.323413874123576972, -0.346462947962331735, 0.59979466285217542},
    {1.66991341326291344e-17, 3.95378229282456509e-16, 3.94599045143233784, 1.06036597479721211e-16, 3.2831773458454161,
     9.99999999999999822, 1.12829414671605333e-17, 1.2026194599794709e-17, 2.50706276000769697e-15,
     2.24624122987970677e-15, 0.370764847417013987, 0.278456024942955571, 0.523838487672241171, 0.388620152510322781,
     0.298156764974678579},
    {1.28582343498528086e-18,
     4.83460302526130664e-34,
     0,
     0,
     6.30459929660781851e-18,
     7.57192526201145068e-34,
     5.03350698372840437e-34,
     9.28268079616618064e-34,
     0,
     1.76723384525547359e-17,
     3.55686101822965701e-34,
     2.99413850083471346e-34,
     0.158143376337580827,
     2.29601774161699833e-19,
     1.06106938611042947e-18,
     1.31968344319506391e-18,
     0.530902525044209539,
     0,
     2.89148310257773535e-18,
     3.34892126180666159e-18,
     0,
     0.310999974151577319,
     5.41244666317833561e-05,
     4.84993165246959553e-16},
    {193.724510070034967, 5.56944131553368433e-27, 17.3191887294084914, 100.047897801386839, 6.68445185362377892,
     5.99168428444264833, 6.21451648886070451},
    {236.430975504001054, 135.82847151732463,  204.818152544824585, 6446.54654059436416, 3007540.83940215595,
     4074188.65771341929, 32918270.5028952882, 130.075408394314167, 170.817294970528621, 299.924591605478554,
     399.258113423595205, 330.817294971142758, 184.51831230897065,  248.64670239647424,  127.658546694545862,
     269.182627528746707, 160.000016724090955, 5.29788288102680571, 5.13529735903945728, 5.59531526444068827,
     5.43444479314453499, 5.07517453535834395},
    {0.00510000000000259465, 99.9947000000000514, 9.01920162996045897e-18, 99.9999000000000535, 0.000100000000027086086,
     2.75700683389584542e-14, 99.9999999999999574, 200, 0.0100000100000100008},
    {2.32952019747762, 3.17849307411774}};

} // end namespace detail

/// The CEC 2006 problems: Constrained Real-Parameter Optimization
/**
 *
 * This class allows to instantiate any of the 24 problems of the competition
 * on constrained real-parameter optimization problems that was organized in the
 * framework of the 2006 IEEE Congress on Evolutionary Computation.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The code for these UDAs is adapted from the original C code distributed during the competition and linked below.
 *
 * .. note::
 *
 *    All problems are constrained, continuous, single objective problems.
 *
 * .. seealso:
 *
 *    http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC-06/CEC06.htm
 *
 * \endverbatim
 */
class cec2006 : private detail::cec2006_statics<>
{
public:
    /// Constructor
    /**
     * Will construct one of the 24 CEC2006 problems
     *
     * @param prob_id The problem id. One of [1,2,...,24]
     *
     * @throws invalid_argument if \p prob_id is not in [1,24]
     */
    cec2006(unsigned prob_id = 1u) : m_prob_id(prob_id)
    {
        if (prob_id < 1u || prob_id > 24u) {
            pagmo_throw(std::invalid_argument,
                        "Error: CEC2006 Test functions are only defined for prob_id in [1, 24], a prob_id of "
                            + std::to_string(prob_id) + " was detected.");
        }
    }
    /// Equality constraint dimension
    /**
     *
     * It returns the number of equality constraints
     *
     * @return the number of equality constraints
     */
    vector_double::size_type get_nec() const
    {
        return m_nec[m_prob_id - 1u];
    }

    /// Inequality constraint dimension
    /**
     *
     * It returns the number of inequality constraints
     *
     * @return the number of inequality constraints
     */
    vector_double::size_type get_nic() const
    {
        return m_nic[m_prob_id - 1u];
    }
    /// Box-bounds
    /**
     *
     * It returns the box-bounds for this UDP.
     *
     * @return the lower and upper bounds for each of the decision vector components
     */
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return m_bounds[m_prob_id - 1];
    }
    /// Fitness computation
    /**
     * Computes the fitness for this UDP
     *
     * @param x the decision vector.
     *
     * @return the fitness of \p x.
     */
    vector_double fitness(const vector_double &x) const
    {
        return fitness_impl(m_c_ptr[m_prob_id - 1], m_o_ptr[m_prob_id - 1], x);
    }
    /// Optimal solution
    /**
     * @return the decision vector corresponding to the best solution for this problem.
     */
    vector_double best_known() const
    {
        return m_best_known[m_prob_id - 1];
    }
    /// Problem name
    /**
     *
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        std::string retval("CEC2006 - g");
        retval.append(std::to_string(m_prob_id));
        return retval;
    }
    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_prob_id);
    }

private:
    // Static data containers needs friendship as to gain access to the private methods definitions
    friend cec2006_statics;
    // Pointers to member functions are used
    vector_double fitness_impl(func_ptr c, func_ptr o, const vector_double &x) const
    {
        vector_double retval(m_nec[m_prob_id - 1u] + m_nic[m_prob_id - 1u], 0.);
        vector_double f(1, 0.);
        // Syntax is ugly as these are member function pointers.
        ((*this).*(c))(retval, x); // calls c
        ((*this).*(o))(f, x);      // calls o
        retval.insert(retval.begin(), f.begin(), f.end());
        return retval;
    }
    // For the coverage analysis we do not cover the code below as its derived from a third party source
    // LCOV_EXCL_START

    // -------------------------------------------

    /// Implementation of the objective function.
    void g01_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = 5.0 * (x[0] + x[1] + x[2] + x[3]) - 5.0 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);

        for (vector_double::size_type j = 4u; j < 13u; ++j)
            f[0] = f[0] - x[j];
    }

    /// Implementation of the constraint function.
    void g01_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        c[0] = 2.0 * x[0] + 2.0 * x[1] + x[9] + x[10] - 10.;
        c[1] = 2.0 * x[0] + 2.0 * x[2] + x[9] + x[11] - 10.;
        c[2] = 2.0 * x[1] + 2.0 * x[2] + x[10] + x[11] - 10.;
        c[3] = -8.0 * x[0] + x[9];
        c[4] = -8.0 * x[1] + x[10];
        c[5] = -8.0 * x[2] + x[11];
        c[6] = -2.0 * x[3] - x[4] + x[9];
        c[7] = -2.0 * x[5] - x[6] + x[10];
        c[8] = -2.0 * x[7] - x[8] + x[11];
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g02_objfun_impl(vector_double &f, const vector_double &x) const
    {
        auto nx = m_dim[m_prob_id - 1u];

        /* objective function */
        double f1 = 0.;
        double f2 = 1.;
        double f3 = 0.;

        for (decltype(nx) j = 0u; j < nx; ++j) {
            f1 = f1 + std::pow(std::cos(x[j]), 4);
            f2 = f2 * std::cos(x[j]) * std::cos(x[j]);
            f3 = f3 + ((double)(j + 1u)) * x[j] * x[j];
        }
        f[0] = -std::abs((f1 - 2 * f2) / std::sqrt(f3));
    }

    /// Implementation of the constraint function.
    void g02_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        auto nx = m_dim[m_prob_id - 1u];

        double g1 = 1.;
        double g2 = 0.;

        for (unsigned j = 0u; j < nx; ++j) {
            g1 = g1 * x[j];
            g2 = g2 + x[j];
        }

        c[0] = 0.75 - g1;
        c[1] = g2 - 7.5 * ((double)nx);
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g03_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        auto nx = m_dim[m_prob_id - 1u];

        double f1 = 1.;
        double f3 = std::sqrt((double)nx);

        for (unsigned j = 0u; j < nx; ++j) {
            f1 = f3 * f1 * x[j];
        }

        f[0] = -f1;
    }

    /// Implementation of the constraint function.
    void g03_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        auto nx = m_dim[m_prob_id - 1u];

        double f2 = 0.;

        for (unsigned j = 0u; j < nx; ++j) {
            f2 = f2 + x[j] * x[j];
        }

        /* constraints h=0 */
        c[0] = f2 - 1.0;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g04_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = 5.3578547 * x[2] * x[2] + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141;
    }

    /// Implementation of the constraint function.
    void g04_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        c[0] = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92.;
        c[1] = -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4];
        c[2] = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] * x[2] - 110.;
        c[3] = -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] * x[2] + 90.;
        c[4] = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25.;
        c[5] = -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20.;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g05_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = 3.0 * x[0] + 0.000001 * std::pow(x[0], 3) + 2.0 * x[1] + (0.000002 / 3.0) * std::pow(x[1], 3);
    }

    /// Implementation of the constraint function.
    void g05_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints h=0 */
        c[0] = 1000.0 * std::sin(-x[2] - 0.25) + 1000.0 * std::sin(-x[3] - 0.25) + 894.8 - x[0];
        c[1] = 1000.0 * std::sin(x[2] - 0.25) + 1000.0 * std::sin(x[2] - x[3] - 0.25) + 894.8 - x[1];
        c[2] = 1000.0 * std::sin(x[3] - 0.25) + 1000.0 * std::sin(x[3] - x[2] - 0.25) + 1294.8;

        /* constraints g<=0 */
        c[3] = -x[3] + x[2] - 0.55;
        c[4] = -x[2] + x[3] - 0.55;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g06_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = std::pow((x[0] - 10.), 3) + std::pow((x[1] - 20.), 3);
    }

    /// Implementation of the constraint function.
    void g06_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        c[0] = 100. - (x[0] - 5.) * (x[0] - 5.) - (x[1] - 5.) * (x[1] - 5.);
        c[1] = (x[0] - 6.) * (x[0] - 6.) + (x[1] - 5.) * (x[1] - 5.) - 82.81;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g07_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = x[0] * x[0] + x[1] * x[1] + x[0] * x[1] - 14.0 * x[0] - 16.0 * x[1] + (x[2] - 10.0) * (x[2] - 10.0)
               + 4.0 * (x[3] - 5.0) * (x[3] - 5.0) + (x[4] - 3.0) * (x[4] - 3.0) + 2.0 * (x[5] - 1.0) * (x[5] - 1.0)
               + 5.0 * x[6] * x[6] + 7.0 * (x[7] - 11) * (x[7] - 11) + 2.0 * (x[8] - 10.0) * (x[8] - 10.0)
               + (x[9] - 7.0) * (x[9] - 7.0) + 45.;
    }

    /// Implementation of the constraint function.
    void g07_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        c[0] = -105.0 + 4.0 * x[0] + 5.0 * x[1] - 3.0 * x[6] + 9.0 * x[7];
        c[1] = 10.0 * x[0] - 8.0 * x[1] - 17.0 * x[6] + 2.0 * x[7];
        c[2] = -8.0 * x[0] + 2.0 * x[1] + 5.0 * x[8] - 2.0 * x[9] - 12.0;
        c[3] = 3.0 * (x[0] - 2.0) * (x[0] - 2.0) + 4.0 * (x[1] - 3.0) * (x[1] - 3.0) + 2.0 * x[2] * x[2] - 7.0 * x[3]
               - 120.0;
        c[4] = 5.0 * x[0] * x[0] + 8.0 * x[1] + (x[2] - 6.0) * (x[2] - 6.0) - 2.0 * x[3] - 40.0;
        c[5] = x[0] * x[0] + 2.0 * (x[1] - 2.0) * (x[1] - 2.0) - 2.0 * x[0] * x[1] + 14.0 * x[4] - 6.0 * x[5];
        c[6] = 0.5 * (x[0] - 8.0) * (x[0] - 8.0) + 2.0 * (x[1] - 4.0) * (x[1] - 4.0) + 3.0 * x[4] * x[4] - x[5] - 30.0;
        c[7] = -3.0 * x[0] + 6.0 * x[1] + 12.0 * (x[8] - 8.0) * (x[8] - 8.0) - 7.0 * x[9];
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g08_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = -std::pow(std::sin(2 * detail::pi() * x[0]), 3) * std::sin(2 * detail::pi() * x[1])
               / (std::pow(x[0], 3) * (x[0] + x[1]));
    }

    /// Implementation of the constraint function.
    void g08_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        c[0] = x[0] * x[0] - x[1] + 1.0;
        c[1] = 1.0 - x[0] + (x[1] - 4.0) * (x[1] - 4.0);
    }

    vector_double g08_fitness_impl(const vector_double &x) const
    {
        vector_double retval(m_nec[m_prob_id - 1u] + m_nic[m_prob_id - 1u], 0.);
        vector_double f(1, 0.);
        g08_compute_constraints_impl(retval, x);
        g08_objfun_impl(f, x);
        retval.insert(retval.begin(), f.begin(), f.end());
        return retval;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g09_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = (x[0] - 10.0) * (x[0] - 10.0) + 5.0 * (x[1] - 12.0) * (x[1] - 12.0) + std::pow(x[2], 4)
               + 3.0 * (x[3] - 11.0) * (x[3] - 11.0) + 10.0 * std::pow(x[4], 6) + 7.0 * x[5] * x[5] + std::pow(x[6], 4)
               - 4.0 * x[5] * x[6] - 10.0 * x[5] - 8.0 * x[6];
    }

    /// Implementation of the constraint function.
    void g09_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        c[0] = -127.0 + 2 * x[0] * x[0] + 3.0 * std::pow(x[1], 4) + x[2] + 4.0 * x[3] * x[3] + 5.0 * x[4];
        c[1] = -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] * x[2] + x[3] - x[4];
        c[2] = -196.0 + 23.0 * x[0] + x[1] * x[1] + 6.0 * x[5] * x[5] - 8.0 * x[6];
        c[3] = 4.0 * x[0] * x[0] + x[1] * x[1] - 3.0 * x[0] * x[1] + 2.0 * x[2] * x[2] + 5.0 * x[5] - 11.0 * x[6];
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g10_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = x[0] + x[1] + x[2];
    }

    /// Implementation of the constraint function.
    void g10_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints g<=0 */
        c[0] = -1.0 + 0.0025 * (x[3] + x[5]);
        c[1] = -1.0 + 0.0025 * (x[4] + x[6] - x[3]);
        c[2] = -1.0 + 0.01 * (x[7] - x[4]);
        c[3] = -x[0] * x[5] + 833.33252 * x[3] + 100.0 * x[0] - 83333.333;
        c[4] = -x[1] * x[6] + 1250.0 * x[4] + x[1] * x[3] - 1250.0 * x[3];
        c[5] = -x[2] * x[7] + 1250000.0 + x[2] * x[4] - 2500.0 * x[4];
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g11_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = x[0] * x[0] + (x[1] - 1.0) * (x[1] - 1.0);
    }

    /// Implementation of the constraint function.
    void g11_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints h=0 */
        c[0] = x[1] - x[0] * x[0];
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g12_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = -(100. - (x[0] - 5.) * (x[0] - 5.) - (x[1] - 5.) * (x[1] - 5.) - (x[2] - 5.) * (x[2] - 5.)) / 100.;
    }

    /// Implementation of the constraint function.
    void g12_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        double gt;

        /* constraints g<=0 */
        c[0] = (x[0] - 1.) * (x[0] - 1.) + (x[1] - 1.) * (x[1] - 1.) + (x[2] - 1.) * (x[2] - 1.) - 0.0625;
        for (int i = 1; i <= 9; ++i) {
            for (int j = 1; j <= 9; ++j) {
                for (int k = 1; k <= 9; k++) {
                    gt = (x[0] - i) * (x[0] - i) + (x[1] - j) * (x[1] - j) + (x[2] - k) * (x[2] - k) - 0.0625;
                    if (gt < c[0]) c[0] = gt;
                }
            }
        }
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g13_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = exp(x[0] * x[1] * x[2] * x[3] * x[4]);
    }

    /// Implementation of the constraint function.
    void g13_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints h(x) = 0 */
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + x[4] * x[4] - 10.0;
        c[1] = x[1] * x[2] - 5.0 * x[3] * x[4];
        c[2] = pow(x[0], 3) + pow(x[1], 3) + 1.0;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g14_objfun_impl(vector_double &f, const vector_double &x) const
    {
        double sumlog = 0.;
        double sum = 0.;
        double C[10] = {-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179};

        /* objective function */
        for (unsigned i = 0u; i < 10u; ++i)
            sumlog += x[i];
        for (unsigned i = 0u; i < 10u; ++i)
            sum += x[i] * (C[i] + std::log(x[i] / sumlog));
        f[0] = sum;
    }

    /// Implementation of the constraint function.
    void g14_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints h=0 */
        c[0] = x[0] + 2.0 * x[1] + 2.0 * x[2] + x[5] + x[9] - 2.0;
        c[1] = x[3] + 2.0 * x[4] + x[5] + x[6] - 1.0;
        c[2] = x[2] + x[6] + x[7] + 2.0 * x[8] + x[9] - 1.0;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g15_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = 1000.0 - pow(x[0], 2.0) - 2.0 * x[1] * x[1] - x[2] * x[2] - x[0] * x[1] - x[0] * x[2];
    }

    /// Implementation of the constraint function.
    void g15_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraints h=0 */
        c[0] = pow(x[0], 2.0) + pow(x[1], 2.0) + pow(x[2], 2.0) - 25.0;
        c[1] = 8.0 * x[0] + 14.0 * x[1] + 7.0 * x[2] - 56.0;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g16_objfun_impl(vector_double &f, const vector_double &x) const
    {
        double C[17], Y[17];

        double x1 = x[0];
        double x2 = x[1];
        double x3 = x[2];
        double x4 = x[3];
        double x5 = x[4];

        Y[0] = x2 + x3 + 41.6;
        C[0] = 0.024 * x4 - 4.62;
        Y[1] = (12.5 / C[0]) + 12.0;
        C[1] = 0.0003535 * pow(x1, 2.0) + 0.5311 * x1 + 0.08705 * Y[1] * x1;
        C[2] = 0.052 * x1 + 78.0 + 0.002377 * Y[1] * x1;
        Y[2] = C[1] / C[2];
        Y[3] = 19.0 * Y[2];
        C[3] = 0.04782 * (x1 - Y[2]) + ((0.1956 * pow(x1 - Y[2], 2.0)) / x2) + 0.6376 * Y[3] + 1.594 * Y[2];
        C[4] = 100 * x2;
        C[5] = x1 - Y[2] - Y[3];
        C[6] = 0.950 - (C[3] / C[4]);
        Y[4] = C[5] * C[6];
        Y[5] = x1 - Y[4] - Y[3] - Y[2];
        C[7] = (Y[4] + Y[3]) * 0.995;
        Y[6] = C[7] / Y[0];
        Y[7] = C[7] / 3798.0;
        C[8] = Y[6] - (0.0663 * Y[6] / Y[7]) - 0.3153;
        Y[8] = (96.82 / C[8]) + 0.321 * Y[0];
        Y[9] = 1.29 * Y[4] + 1.258 * Y[3] + 2.29 * Y[2] + 1.71 * Y[5];
        Y[10] = 1.71 * x1 - 0.452 * Y[3] + 0.580 * Y[2];
        C[9] = 12.3 / 752.3;
        C[10] = 1.75 * Y[1] * 0.995 * x1;
        C[11] = 0.995 * Y[9] + 1998.0;
        Y[11] = C[9] * x1 + (C[10] / C[11]);
        Y[12] = C[11] - 1.75 * Y[1];
        Y[13] = 3623.0 + 64.4 * x2 + 58.4 * x3 + (146312.0 / (Y[8] + x5));
        C[12] = 0.995 * Y[9] + 60.8 * x2 + 48 * x4 - 0.1121 * Y[13] - 5095.0;
        Y[14] = Y[12] / C[12];
        Y[15] = 148000.0 - 331000.0 * Y[14] + 40.0 * Y[12] - 61.0 * Y[14] * Y[12];
        C[13] = 2324 * Y[9] - 28740000 * Y[1];
        Y[16] = 14130000 - 1328.0 * Y[9] - 531.0 * Y[10] + (C[13] / C[11]);
        C[14] = (Y[12] / Y[14]) - (Y[12] / 0.52);
        C[15] = 1.104 - 0.72 * Y[14];
        C[16] = Y[8] + x5;

        /* objective function */
        f[0] = 0.0000005843 * Y[16] - 0.000117 * Y[13] - 0.1365 - 0.00002358 * Y[12] - 0.000001502 * Y[15]
               - 0.0321 * Y[11] - 0.004324 * Y[4] - 0.0001 * (C[14] / C[15]) - 37.48 * (Y[1] / C[11]);
        f[0] = -f[0]; /* Max-->Min, Modified by Jane,Nov 22 2005 */
    }

    /// Implementation of the constraint function.
    void g16_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        double C[17];
        double Y[17];

        double x1 = x[0];
        double x2 = x[1];
        double x3 = x[2];
        double x4 = x[3];
        double x5 = x[4];

        Y[0] = x2 + x3 + 41.6;
        C[0] = 0.024 * x4 - 4.62;
        Y[1] = (12.5 / C[0]) + 12.0;
        C[1] = 0.0003535 * pow(x1, 2.0) + 0.5311 * x1 + 0.08705 * Y[1] * x1;
        C[2] = 0.052 * x1 + 78.0 + 0.002377 * Y[1] * x1;
        Y[2] = C[1] / C[2];
        Y[3] = 19.0 * Y[2];
        C[3] = 0.04782 * (x1 - Y[2]) + ((0.1956 * pow(x1 - Y[2], 2.0)) / x2) + 0.6376 * Y[3] + 1.594 * Y[2];
        C[4] = 100.0 * x2;
        C[5] = x1 - Y[2] - Y[3];
        C[6] = 0.950 - (C[3] / C[4]);
        Y[4] = C[5] * C[6];
        Y[5] = x1 - Y[4] - Y[3] - Y[2];
        C[7] = (Y[4] + Y[3]) * 0.995;
        Y[6] = C[7] / Y[0];
        Y[7] = C[7] / 3798.0;
        C[8] = Y[6] - (0.0663 * Y[6] / Y[7]) - 0.3153;
        Y[8] = (96.82 / C[8]) + 0.321 * Y[0];
        Y[9] = 1.29 * Y[4] + 1.258 * Y[3] + 2.29 * Y[2] + 1.71 * Y[5];
        Y[10] = 1.71 * x1 - 0.452 * Y[3] + 0.580 * Y[2];
        C[9] = 12.3 / 752.3;
        C[10] = 1.75 * Y[1] * 0.995 * x1;
        C[11] = 0.995 * Y[9] + 1998.0;
        Y[11] = C[9] * x1 + (C[10] / C[11]);
        Y[12] = C[11] - 1.75 * Y[1];
        Y[13] = 3623.0 + 64.4 * x2 + 58.4 * x3 + (146312.0 / (Y[8] + x5));
        C[12] = 0.995 * Y[9] + 60.8 * x2 + 48 * x4 - 0.1121 * Y[13] - 5095.0;
        Y[14] = Y[12] / C[12];
        Y[15] = 148000.0 - 331000.0 * Y[14] + 40.0 * Y[12] - 61.0 * Y[14] * Y[12];
        C[13] = 2324.0 * Y[9] - 28740000.0 * Y[1];
        Y[16] = 14130000 - 1328.0 * Y[9] - 531.0 * Y[10] + (C[13] / C[11]);
        C[14] = (Y[12] / Y[14]) - (Y[12] / 0.52);
        C[15] = 1.104 - 0.72 * Y[14];
        C[16] = Y[8] + x5;

        /* constraints g(x) <= 0 */
        c[0] = -Y[3] + (0.28 / 0.72) * Y[4];
        c[1] = -1.5 * x2 + x3;
        c[2] = -21.0 + 3496.0 * (Y[1] / C[11]);
        c[3] = -(62212.0 / C[16]) + 110.6 + Y[0];
        c[4] = 213.1 - Y[0];
        c[5] = Y[0] - 405.23;
        c[6] = 17.505 - Y[1];
        c[7] = Y[1] - 1053.6667;
        c[8] = 11.275 - Y[2];
        c[9] = Y[2] - 35.03;
        c[10] = 214.228 - Y[3];
        c[11] = Y[3] - 665.585;
        c[12] = 7.458 - Y[4];
        c[13] = Y[4] - 584.463;
        c[14] = 0.961 - Y[5];
        c[15] = Y[5] - 265.916;
        c[16] = 1.612 - Y[6];
        c[17] = Y[6] - 7.046;
        c[18] = 0.146 - Y[7];
        c[19] = Y[7] - 0.222;
        c[20] = 107.99 - Y[8];
        c[21] = Y[8] - 273.366;
        c[22] = 922.693 - Y[9];
        c[23] = Y[9] - 1286.105;
        c[24] = 926.832 - Y[10];
        c[25] = Y[10] - 1444.046;
        c[26] = 18.766 - Y[11];
        c[27] = Y[11] - 537.141;
        c[28] = 1072.163 - Y[12];
        c[29] = Y[12] - 3247.039;
        c[30] = 8961.448 - Y[13];
        c[31] = Y[13] - 26844.086;
        c[32] = 0.063 - Y[14];
        c[33] = Y[14] - 0.386;
        c[34] = 71084.33 - Y[15];
        c[35] = Y[15] - 140000.0;
        c[36] = 2802713.0 - Y[16];
        c[37] = Y[16] - 12146108.0;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g17_objfun_impl(vector_double &f, const vector_double &x) const
    {
        double f1 = 0;
        double f2 = 0;

        double x1 = x[0];
        double x2 = x[1];
        double x3 = x[2];
        double x4 = x[3];
        double x6 = x[5];

        double aux1
            = 300.0
              - (((x3 * x4) * std::cos(1.48477 - x6)) - ((0.90798 * std::pow(x3, 2.0)) * std::cos(1.47588))) / 131.078;
        double aux2
            = -(((x3 * x4) * std::cos(1.48477 + x6)) - ((0.90798 * std::pow(x4, 2.0)) * std::cos(1.47588))) / 131.078;

        /* objective fucntion */
        if (x1 >= 0.0 && x1 < 300.0) {
            f1 = 30.0 * aux1;
        } else {
            if (x1 >= 300.0 && x1 <= 400.0) {
                f1 = 31.0 * aux1;
            }
        }
        if (x2 >= 0.0 && x2 < 100.0) {
            f2 = 28.0 * aux2;
        } else {
            if (x2 >= 100.0 && x2 < 200.0) {
                f2 = 29.0 * aux2;
            } else {
                if (x2 >= 200.0 && x2 <= 1000.0) {
                    f2 = 30.0 * aux2;
                }
            }
        }
        f[0] = f1 + f2;
    }

    /// Implementation of the constraint function.
    void g17_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        double x1 = x[0];
        double x2 = x[1];
        double x3 = x[2];
        double x4 = x[3];
        double x5 = x[4];
        double x6 = x[5];

        double aux1
            = 300.0
              - (((x3 * x4) * std::cos(1.48477 - x6)) - ((0.90798 * std::pow(x3, 2.0)) * std::cos(1.47588))) / 131.078;
        double aux2
            = -(((x3 * x4) * std::cos(1.48477 + x6)) - ((0.90798 * std::pow(x4, 2.0)) * std::cos(1.47588))) / 131.078;
        double aux5
            = -(((x3 * x4) * std::sin(1.48477 + x6)) - ((0.90798 * std::pow(x4, 2.0)) * std::sin(1.47588))) / 131.078;
        double aux4
            = 200.0
              - (((x3 * x4) * std::sin(1.48477 - x6)) - ((0.90798 * std::pow(x3, 2.0)) * std::sin(1.47588))) / 131.078;

        /* constraint function h = 0 */
        c[0] = aux1 - x1;
        c[1] = aux2 - x2;
        c[2] = aux5 - x5;
        c[3] = aux4;
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g18_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = -0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6]);
    }

    /// Implementation of the constraint function.
    void g18_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraint function g <= 0 */
        c[0] = -1.0 + std::pow(x[2], 2.0) + std::pow(x[3], 2.0);
        c[1] = -1.0 + std::pow(x[8], 2.0);
        c[2] = -1.0 + std::pow(x[4], 2.0) + std::pow(x[5], 2.0);
        c[3] = -1.0 + std::pow(x[0], 2.0) + std::pow(x[1] - x[8], 2.0);
        c[4] = -1.0 + std::pow(x[0] - x[4], 2.0) + std::pow(x[1] - x[5], 2.0);
        c[5] = -1.0 + std::pow(x[0] - x[6], 2.0) + std::pow(x[1] - x[7], 2.0);
        c[6] = -1.0 + std::pow(x[2] - x[4], 2.0) + std::pow(x[3] - x[5], 2.0);
        c[7] = -1.0 + std::pow(x[2] - x[6], 2.0) + std::pow(x[3] - x[7], 2.0);
        c[8] = -1.0 + std::pow(x[6], 2.0) + std::pow(x[7] - x[8], 2.0);
        c[9] = -x[0] * x[3] + x[1] * x[2];
        c[10] = -x[2] * x[8];
        c[11] = x[4] * x[8];
        c[12] = -x[4] * x[7] + x[5] * x[6];
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g19_objfun_impl(vector_double &f, const vector_double &x) const
    {
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;

        double B[10] = {-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0};
        double C[5][5] = {{30.0, -20.0, -10.0, 32.0, -10.0},
                          {-20.0, 39.0, -6.0, -31.0, 32.0},
                          {-10.0, -6.0, 10.0, -6.0, -10.0},
                          {32.0, -31.0, -6.0, 39.0, -20.0},
                          {-10.0, 32.0, -10.0, -20.0, 30.0}};
        double D[5] = {4.0, 8.0, 10.0, 6.0, 2.0};

        /* objective function */
        for (unsigned i = 0u; i < 10u; ++i) {
            sum1 += B[i] * x[i];
        }
        for (unsigned i = 0u; i < 5u; ++i) {
            for (unsigned j = 0; j < 5; ++j) {
                sum2 += C[i][j] * x[10 + i] * x[10 + j];
            }
        }
        for (unsigned i = 0u; i < 5u; ++i) {
            sum3 += D[i] * pow(x[10 + i], 3.0);
        }

        f[0] = sum1 - sum2 - 2.0 * sum3;
        f[0] = -f[0];
    }

    /// Implementation of the constraint function.
    void g19_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        double sum1 = 0.;
        double sum2 = 0.;

        double A[10][5] = {{-16.0, 2.0, 0.0, 1.0, 0.0},    {0.0, -2.0, 0.0, 0.4, 2.0},     {-3.5, 0.0, 2.0, 0.0, 0.0},
                           {0.0, -2.0, 0.0, -4.0, -1.0},   {0.0, -9.0, -2.0, 1.0, -2.8},   {2.0, 0.0, -4.0, 0.0, 0.0},
                           {-1.0, -1.0, -1.0, -1.0, -1.0}, {-1.0, -2.0, -3.0, -2.0, -1.0}, {1.0, 2.0, 3.0, 4.0, 5.0},
                           {1.0, 1.0, 1.0, 1.0, 1.0}};

        double C[5][5] = {{30.0, -20.0, -10.0, 32.0, -10.0},
                          {-20.0, 39.0, -6.0, -31.0, 32.0},
                          {-10.0, -6.0, 10.0, -6.0, -10.0},
                          {32.0, -31.0, -6.0, 39.0, -20.0},
                          {-10.0, 32.0, -10.0, -20.0, 30.0}};

        double D[5] = {4.0, 8.0, 10.0, 6.0, 2.0};
        double E[5] = {-15.0, -27.0, -36.0, -18.0, -12.0};

        /* constraints g <= 0 */
        for (unsigned j = 0u; j < 5u; ++j) {
            sum1 = 0.0;
            for (unsigned i = 0u; i < 5u; ++i)
                sum1 += C[i][j] * x[10 + i];
            sum2 = 0.0;
            for (unsigned i = 0u; i < 10u; ++i)
                sum2 += A[i][j] * x[i];
            c[j] = -((2.0 * sum1) + (3.0 * D[j] * pow(x[10 + j], 2.0)) + E[j] - sum2);
        }
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g20_objfun_impl(vector_double &f, const vector_double &x) const
    {
        double A[24] = {0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
                        0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09};

        /* objective function */
        f[0] = 0.0;
        for (unsigned j = 0u; j < 24u; ++j) {
            f[0] += A[j] * x[j];
        }
    }

    /// Implementation of the constraint function.
    void g20_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        double sum1, sum2, sumtotal;

        double B[24] = {44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097,
                        44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097};
        double C[12] = {123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64};
        double D[12] = {31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1};
        double E[6] = {0.1, 0.3, 0.4, 0.3, 0.6, 0.3};

        /* constraints h(x) = 0 */
        sum1 = 0.0;
        for (unsigned j = 0u; j < 12u; ++j)
            sum1 += x[j] / B[j];
        sum2 = 0.0;
        for (unsigned j = 12u; j < 24u; ++j)
            sum2 += x[j] / B[j];
        for (unsigned i = 0u; i < 12u; ++i)
            c[i] = (x[i + 12] / (B[i + 12] * sum2)) - ((C[i] * x[i]) / (40.0 * B[i] * sum1));
        sumtotal = 0.0;
        for (unsigned j = 0u; j < 24u; ++j)
            sumtotal += x[j];
        c[12] = sumtotal - 1.0;
        sum1 = 0.0;
        for (unsigned j = 0u; j < 12u; ++j)
            sum1 += x[j] / D[j];
        sum2 = 0.0;
        for (unsigned j = 12u; j < 24u; ++j)
            sum2 += x[j] / B[j];
        c[13] = sum1 + (0.7302 * 530.0 * (14.7 / 40)) * sum2 - 1.671;

        /* constraints g(x) <= 0 */
        for (unsigned j = 0u; j < 3u; ++j)
            c[14 + j] = (x[j] + x[j + 12]) / (sumtotal + E[j]);
        for (unsigned j = 3u; j < 6u; ++j)
            c[14 + j] = (x[j + 3] + x[j + 15]) / (sumtotal + E[j]);
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g21_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = x[0];
    }

    /// Implementation of the constraint function.
    void g21_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraint functions h(x) = 0 */
        c[0] = -300.0 * x[2] + 7500 * x[4] - 7500 * x[5] - 25.0 * x[3] * x[4] + 25.0 * x[3] * x[5] + x[2] * x[3];
        c[1] = 100.0 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25.0 * x[3] * x[6] - 15536.5;
        c[2] = -x[4] + log(-x[3] + 900.0);
        c[3] = -x[5] + log(x[3] + 300.0);
        c[4] = -x[6] + log(-2.0 * x[3] + 700.0);

        /* constraint functions g(x) <= 0 */
        c[5] = -x[0] + 35.0 * pow(x[1], 0.6) + 35.0 * pow(x[2], 0.6);
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g22_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = x[0];
    }

    /// Implementation of the constraint function.
    void g22_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraint functions h(x) = 0 */
        c[0] = x[4] - 100000.0 * x[7] + 10000000.0;
        c[1] = x[5] + 100000.0 * x[7] - 100000.0 * x[8];
        c[2] = x[6] + 100000.0 * x[8] - 50000000.0;
        c[3] = x[4] + 100000.0 * x[9] - 33000000.0;
        c[4] = x[5] + 100000 * x[10] - 44000000.0;
        c[5] = x[6] + 100000 * x[11] - 66000000.0;
        c[6] = x[4] - 120.0 * x[1] * x[12];
        c[7] = x[5] - 80.0 * x[2] * x[13];
        c[8] = x[6] - 40.0 * x[3] * x[14];
        c[9] = x[7] - x[10] + x[15];
        c[10] = x[8] - x[11] + x[16];
        c[11] = -x[17] + log(x[9] - 100.0);
        c[12] = -x[18] + log(-x[7] + 300.0);
        c[13] = -x[19] + log(x[15]);
        c[14] = -x[20] + log(-x[8] + 400.0);
        c[15] = -x[21] + log(x[16]);
        c[16] = -x[7] - x[9] + x[12] * x[17] - x[12] * x[18] + 400.0;
        c[17] = x[7] - x[8] - x[10] + x[13] * x[19] - x[13] * x[20] + 400.0;
        c[18] = x[8] - x[11] - 4.60517 * x[14] + x[14] * x[21] + 100.0;

        /* constraint functions g(x) <= 0 */
        c[19] = -x[0] + std::pow(x[1], 0.6) + std::pow(x[2], 0.6) + std::pow(x[3], 0.6);
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g23_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = -9.0 * x[4] - 15.0 * x[7] + 6.0 * x[0] + 16.0 * x[1] + 10.0 * (x[5] + x[6]);
    }

    /// Implementation of the constraint function.
    void g23_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraint functions h(x) = 0 */
        c[0] = x[0] + x[1] - x[2] - x[3];
        c[1] = 0.03 * x[0] + 0.01 * x[1] - x[8] * (x[2] + x[3]);
        c[2] = x[2] + x[5] - x[4];
        c[3] = x[3] + x[6] - x[7];

        /* constraint functions g(x) <= 0 */
        c[4] = x[8] * x[2] + 0.02 * x[5] - 0.025 * x[4];
        c[5] = x[8] * x[3] + 0.02 * x[6] - 0.015 * x[7];
    }

    // -------------------------------------------

    /// Implementation of the objective function.
    void g24_objfun_impl(vector_double &f, const vector_double &x) const
    {
        /* objective function */
        f[0] = -x[0] - x[1];
    }

    /// Implementation of the constraint function.
    void g24_compute_constraints_impl(vector_double &c, const vector_double &x) const
    {
        /* constraint functions g(x) <= 0 */
        c[0] = -2.0 * std::pow(x[0], 4.0) + 8.0 * std::pow(x[0], 3.0) - 8.0 * std::pow(x[0], 2.0) + x[1] - 2.0;
        c[1] = -4.0 * std::pow(x[0], 4.0) + 32.0 * std::pow(x[0], 3.0) - 88.0 * std::pow(x[0], 2.0) + 96.0 * x[0] + x[1]
               - 36.0;
    }

    // -------------------------------------------

    // LCOV_EXCL_STOP

    // problem id
    unsigned m_prob_id;
};

// Bunch of member function pointers as static member
namespace detail
{
template <typename T>
const std::vector<typename cec2006_statics<T>::func_ptr> cec2006_statics<T>::m_o_ptr
    = {&cec2006::g01_objfun_impl, &cec2006::g02_objfun_impl, &cec2006::g03_objfun_impl, &cec2006::g04_objfun_impl,
       &cec2006::g05_objfun_impl, &cec2006::g06_objfun_impl, &cec2006::g07_objfun_impl, &cec2006::g08_objfun_impl,
       &cec2006::g09_objfun_impl, &cec2006::g10_objfun_impl, &cec2006::g11_objfun_impl, &cec2006::g12_objfun_impl,
       &cec2006::g13_objfun_impl, &cec2006::g14_objfun_impl, &cec2006::g15_objfun_impl, &cec2006::g16_objfun_impl,
       &cec2006::g17_objfun_impl, &cec2006::g18_objfun_impl, &cec2006::g19_objfun_impl, &cec2006::g20_objfun_impl,
       &cec2006::g21_objfun_impl, &cec2006::g22_objfun_impl, &cec2006::g23_objfun_impl, &cec2006::g24_objfun_impl};

template <typename T>
const std::vector<typename cec2006_statics<T>::func_ptr> cec2006_statics<T>::m_c_ptr
    = {&cec2006::g01_compute_constraints_impl, &cec2006::g02_compute_constraints_impl,
       &cec2006::g03_compute_constraints_impl, &cec2006::g04_compute_constraints_impl,
       &cec2006::g05_compute_constraints_impl, &cec2006::g06_compute_constraints_impl,
       &cec2006::g07_compute_constraints_impl, &cec2006::g08_compute_constraints_impl,
       &cec2006::g09_compute_constraints_impl, &cec2006::g10_compute_constraints_impl,
       &cec2006::g11_compute_constraints_impl, &cec2006::g12_compute_constraints_impl,
       &cec2006::g13_compute_constraints_impl, &cec2006::g14_compute_constraints_impl,
       &cec2006::g15_compute_constraints_impl, &cec2006::g16_compute_constraints_impl,
       &cec2006::g17_compute_constraints_impl, &cec2006::g18_compute_constraints_impl,
       &cec2006::g19_compute_constraints_impl, &cec2006::g20_compute_constraints_impl,
       &cec2006::g21_compute_constraints_impl, &cec2006::g22_compute_constraints_impl,
       &cec2006::g23_compute_constraints_impl, &cec2006::g24_compute_constraints_impl};
} // namespace detail
} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::cec2006)

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif
