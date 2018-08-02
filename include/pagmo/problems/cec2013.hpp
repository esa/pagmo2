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

#ifndef PAGMO_PROBLEM_CEC2013_HPP
#define PAGMO_PROBLEM_CEC2013_HPP

#include <pagmo/config.hpp>

#if !defined(PAGMO_ENABLE_CEC2013)

#error The cec2013.hpp header was included but the CEC 2013 problem is not supported on the current platform.

#endif

#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/cec2013_data.hpp>
#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp> // needed for cereal registration macro
#include <pagmo/types.hpp>

#define E 2.7182818284590452353602874713526625

namespace pagmo
{
/// The CEC 2013 problems: Real-Parameter Single Objective Optimization Competition
/**
 *
 * \image html cec2013.png
 *
 * The 28 problems of the competition on real-parameter single objective optimization problems that
 * was organized for the 2013 IEEE Congress on Evolutionary Computation.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The code for these UDAs is adapted from the original C code distributed during the competition and
 *    linked below.
 *
 * .. note::
 *
 *    All problems are box-bounded, continuous, single objective problems.
 *
 * .. seealso:
 *
 *    http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC2013/CEC2013.htm
 *    http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC2013/cec13-c-code.zip
 *
 * \endverbatim
 */
class cec2013
{
public:
    /// Constructor
    /**
     * Will construct one of the 28 CEC2013 problems
     *
     * @param prob_id The problem id. One of [1,2,...,28]
     * @param dim problem dimension. One of [2,5,10,20,30,...,100]
     *
     * @throws invalid_argument if \p prob_id is not in [1,18] or if \p dim is not one of
     * [2,5,10,20,30,40,50,60,70,80,90,100]
     */
    cec2013(unsigned int prob_id = 1u, unsigned int dim = 2u)
        : m_prob_id(prob_id), m_rotation_matrix(), m_origin_shift(), m_y(dim), m_z(dim)
    {
        if (!(dim == 2u || dim == 5u || dim == 10u || dim == 20u || dim == 30u || dim == 40u || dim == 50u || dim == 60u
              || dim == 70u || dim == 80u || dim == 90u || dim == 100u)) {
            pagmo_throw(std::invalid_argument, "Error: CEC2013 Test functions are only defined for dimensions "
                                               "2,5,10,20,30,40,50,60,70,80,90,100, a dimension of "
                                                   + std::to_string(dim) + " was detected.");
        }
        if (prob_id < 1u || prob_id > 28u) {
            pagmo_throw(std::invalid_argument,
                        "Error: CEC2013 Test functions are only defined for prob_id in [1, 28], a prob_id of "
                            + std::to_string(prob_id) + " was detected.");
        }
        m_origin_shift = detail::cec2013_data::shift_data;
        auto it = detail::cec2013_data::MD.find(dim);
        assert(it != detail::cec2013_data::MD.end());
        m_rotation_matrix = it->second;
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
        unsigned int nx = static_cast<unsigned int>(m_z.size()); // maximum is 100
        vector_double f(1);
        switch (m_prob_id) {
            case 1:
                sphere_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 0);
                f[0] += -1400.0;
                break;
            case 2:
                ellips_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -1300.0;
                break;
            case 3:
                bent_cigar_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -1200.0;
                break;
            case 4:
                discus_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -1100.0;
                break;
            case 5:
                dif_powers_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 0);
                f[0] += -1000.0;
                break;
            case 6:
                rosenbrock_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -900.0;
                break;
            case 7:
                schaffer_F7_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -800.0;
                break;
            case 8:
                ackley_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -700.0;
                break;
            case 9:
                weierstrass_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -600.0;
                break;
            case 10:
                griewank_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -500.0;
                break;
            case 11:
                rastrigin_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 0);
                f[0] += -400.0;
                break;
            case 12:
                rastrigin_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -300.0;
                break;
            case 13:
                step_rastrigin_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += -200.0;
                break;
            case 14:
                schwefel_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 0);
                f[0] += -100.0;
                break;
            case 15:
                schwefel_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 100.0;
                break;
            case 16:
                katsuura_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 200.0;
                break;
            case 17:
                bi_rastrigin_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 0);
                f[0] += 300.0;
                break;
            case 18:
                bi_rastrigin_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 400.0;
                break;
            case 19:
                grie_rosen_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 500.0;
                break;
            case 20:
                escaffer6_func(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 600.0;
                break;
            case 21:
                cf01(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 700.0;
                break;
            case 22:
                cf02(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 0);
                f[0] += 800.0;
                break;
            case 23:
                cf03(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 900.0;
                break;
            case 24:
                cf04(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 1000.0;
                break;
            case 25:
                cf05(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 1100.0;
                break;
            case 26:
                cf06(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 1200.0;
                break;
            case 27:
                cf07(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 1300.0;
                break;
            case 28:
                cf08(&x[0], &f[0], nx, &m_origin_shift[0], &m_rotation_matrix[0], 1);
                f[0] += 1400.0;
                break;
        }
        return f;
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
        // all CEC 2013 problems have the same bounds
        vector_double lb(m_z.size(), -100.);
        vector_double ub(m_z.size(), 100.);
        return std::make_pair(std::move(lb), std::move(ub));
    }
    /// Problem name
    /**
     *
     *
     * @return a string containing the problem name
     */
    std::string get_name() const
    {
        std::string retval("CEC2013 - f");
        retval.append(std::to_string(m_prob_id));
        switch (m_prob_id) {
            case 1:
                retval.append("(sphere_func)");
                break;
            case 2:
                retval.append("(ellips_func)");
                break;
            case 3:
                retval.append("(bent_cigar_func)");
                break;
            case 4:
                retval.append("(discus_func)");
                break;
            case 5:
                retval.append("(dif_powers_func_non_rotated)");
                break;
            case 6:
                retval.append("(rosenbrock_func)");
                break;
            case 7:
                retval.append("(schaffer_F7_func)");
                break;
            case 8:
                retval.append("(ackley_func)");
                break;
            case 9:
                retval.append("(weierstrass_func)");
                break;
            case 10:
                retval.append("(griewank_func)");
                break;
            case 11:
                retval.append("(rastrigin_func_non_rotated)");
                break;
            case 12:
                retval.append("(rastrigin_func)");
                break;
            case 13:
                retval.append("(step_rastrigin_func)");
                break;
            case 14:
                retval.append("(schwefel_func_non_rotated)");
                break;
            case 15:
                retval.append("(schwefel_func)");
                break;
            case 16:
                retval.append("(katsuura_func)");
                break;
            case 17:
                retval.append("(bi_rastrigin_func_non_rotated)");
                break;
            case 18:
                retval.append("(bi_rastrigin_func)");
                break;
            case 19:
                retval.append("(grie_rosen_func)");
                break;
            case 20:
                retval.append("(escaffer6_func)");
                break;
            case 21:
                retval.append("(cf01)");
                break;
            case 22:
                retval.append("(cf02)");
                break;
            case 23:
                retval.append("(cf03)");
                break;
            case 24:
                retval.append("(cf04)");
                break;
            case 25:
                retval.append("(cf05)");
                break;
            case 26:
                retval.append("(cf06)");
                break;
            case 27:
                retval.append("(cf07)");
                break;
            case 28:
                retval.append("(cf08)");
                break;
        }
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
        ar(m_prob_id, m_rotation_matrix, m_origin_shift, m_y, m_z);
    }

private:
    // For the coverage analysis we do not cover the code below as its derived from a third party source
    // LCOV_EXCL_START
    void sphere_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                     int r_flag) const /* Sphere */
    {
        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (unsigned int i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];
        f[0] = 0.0;
        for (unsigned int i = 0u; i < nx; ++i) {
            f[0] += m_z[i] * m_z[i];
        }
    }

    void ellips_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                     int r_flag) const /* Ellipsoidal */
    {
        unsigned int i;
        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];
        oszfunc(&m_z[0], &m_y[0], nx);
        f[0] = 0.0;
        for (i = 0u; i < nx; ++i) {
            f[0] += std::pow(10.0, (6. * i) / (nx - 1u)) * m_y[i] * m_y[i];
        }
    }

    void bent_cigar_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                         int r_flag) const /* Bent_Cigar */
    {
        unsigned int i;
        double beta = 0.5;
        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];
        asyfunc(&m_z[0], &m_y[0], nx, beta);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        f[0] = m_z[0] * m_z[0];
        for (i = 1u; i < nx; ++i) {
            f[0] += std::pow(10.0, 6.0) * m_z[i] * m_z[i];
        }
    }

    void discus_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                     int r_flag) const /* Discus */
    {
        unsigned int i;
        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];
        oszfunc(&m_z[0], &m_y[0], nx);

        f[0] = std::pow(10.0, 6.0) * m_y[0] * m_y[0];
        for (i = 1u; i < nx; ++i) {
            f[0] += m_y[i] * m_y[i];
        }
    }

    void dif_powers_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                         int r_flag) const /* Different Powers */
    {
        unsigned int i;
        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];
        f[0] = 0.0;
        for (i = 0u; i < nx; ++i) {
            f[0] += std::pow(std::abs(m_z[i]), 2. + (4. * i) / (nx - 1u));
        }
        f[0] = std::pow(f[0], 0.5);
    }

    void rosenbrock_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                         int r_flag) const /* Rosenbrock's */
    {
        unsigned int i;
        double tmp1, tmp2;
        shiftfunc(x, &m_y[0], nx, Os); // shift
        for (i = 0u; i < nx; ++i)      // shrink to the orginal search range
        {
            m_y[i] = m_y[i] * 2.048 / 100.;
        }
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr); // rotate
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];
        for (i = 0u; i < nx; ++i) // shift to orgin
        {
            m_z[i] = m_z[i] + 1;
        }

        f[0] = 0.0;
        for (i = 0u; i < nx - 1; ++i) {
            tmp1 = m_z[i] * m_z[i] - m_z[i + 1];
            tmp2 = m_z[i] - 1.0;
            f[0] += 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
        }
    }

    void schaffer_F7_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                          int r_flag) const /* Schwefel's 1.2  */
    {
        unsigned int i;
        double tmp;
        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];
        asyfunc(&m_z[0], &m_y[0], nx, 0.5);
        for (i = 0u; i < nx; ++i)
            m_z[i] = m_y[i] * std::pow(10.0, (1. * i) / (nx - 1u) / 2.0);
        if (r_flag == 1)
            rotatefunc(&m_z[0], &m_y[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_y[i] = m_z[i];

        for (i = 0u; i < nx - 1u; ++i)
            m_z[i] = std::pow(m_y[i] * m_y[i] + m_y[i + 1] * m_y[i + 1], 0.5);
        f[0] = 0.0;
        for (i = 0u; i < nx - 1u; ++i) {
            tmp = std::sin(50.0 * std::pow(m_z[i], 0.2));
            f[0] += std::pow(m_z[i], 0.5) + std::pow(m_z[i], 0.5) * tmp * tmp;
        }
        f[0] = f[0] * f[0] / (nx - 1) / (nx - 1);
    }

    void ackley_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                     int r_flag) const /* Ackley's  */
    {
        unsigned int i;
        double sum1, sum2;

        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        asyfunc(&m_z[0], &m_y[0], nx, 0.5);
        for (i = 0u; i < nx; ++i)
            m_z[i] = m_y[i] * std::pow(10.0, (1. * i) / (nx - 1u) / 2.0);
        if (r_flag == 1)
            rotatefunc(&m_z[0], &m_y[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_y[i] = m_z[i];

        sum1 = 0.0;
        sum2 = 0.0;
        for (i = 0u; i < nx; ++i) {
            sum1 += m_y[i] * m_y[i];
            sum2 += std::cos(2.0 * detail::pi() * m_y[i]);
        }
        sum1 = -0.2 * std::sqrt(sum1 / nx);
        sum2 /= nx;
        f[0] = E - 20.0 * std::exp(sum1) - std::exp(sum2) + 20.0;
    }

    void weierstrass_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                          int r_flag) const /* Weierstrass's  */
    {
        unsigned int i, j, k_max;
        double sum = 0, sum2 = 0, a, b;

        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] = m_y[i] * 0.5 / 100;
        }
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        asyfunc(&m_z[0], &m_y[0], nx, 0.5);
        for (i = 0u; i < nx; ++i)
            m_z[i] = m_y[i] * std::pow(10.0, (1. * i) / (nx - 1u) / 2.0);
        if (r_flag == 1)
            rotatefunc(&m_z[0], &m_y[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_y[i] = m_z[i];

        a = 0.5;
        b = 3.0;
        k_max = 20;
        f[0] = 0.0;
        for (i = 0u; i < nx; ++i) {
            sum = 0.0;
            sum2 = 0.0;
            for (j = 0u; j <= k_max; ++j) {
                sum += std::pow(a, j) * std::cos(2.0 * detail::pi() * std::pow(b, j) * (m_y[i] + 0.5));
                sum2 += std::pow(a, j) * std::cos(2.0 * detail::pi() * std::pow(b, j) * 0.5);
            }
            f[0] += sum;
        }
        f[0] -= nx * sum2;
    }

    void griewank_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                       int r_flag) const /* Griewank's  */
    {
        unsigned int i;
        double s, p;

        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] = m_y[i] * 600.0 / 100.0;
        }
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        for (i = 0u; i < nx; ++i)
            m_z[i] = m_z[i] * std::pow(100.0, (1. * i) / (nx - 1u) / 2.0);

        s = 0.0;
        p = 1.0;
        for (i = 0u; i < nx; ++i) {
            s += m_z[i] * m_z[i];
            p *= std::cos(m_z[i] / std::sqrt(1.0 + i));
        }
        f[0] = 1.0 + s / 4000.0 - p;
    }

    void rastrigin_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                        int r_flag) const /* Rastrigin's  */
    {
        unsigned int i;
        double alpha = 10.0, beta = 0.2;
        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] = m_y[i] * 5.12 / 100;
        }

        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        oszfunc(&m_z[0], &m_y[0], nx);
        asyfunc(&m_y[0], &m_z[0], nx, beta);

        if (r_flag == 1)
            rotatefunc(&m_z[0], &m_y[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_y[i] = m_z[i];

        for (i = 0u; i < nx; ++i) {
            m_y[i] *= std::pow(alpha, (1. * i) / (nx - 1u) / 2);
        }

        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        f[0] = 0.0;
        for (i = 0u; i < nx; ++i) {
            f[0] += (m_z[i] * m_z[i] - 10.0 * std::cos(2.0 * detail::pi() * m_z[i]) + 10.0);
        }
    }

    void step_rastrigin_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                             int r_flag) const /* Noncontinuous Rastrigin's  */
    {
        unsigned int i;
        double alpha = 10.0, beta = 0.2;
        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] = m_y[i] * 5.12 / 100;
        }

        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        for (i = 0u; i < nx; ++i) {
            if (std::abs(m_z[i]) > 0.5) m_z[i] = std::floor(2. * m_z[i] + 0.5) / 2.;
        }

        oszfunc(&m_z[0], &m_y[0], nx);
        asyfunc(&m_y[0], &m_z[0], nx, beta);

        if (r_flag == 1)
            rotatefunc(&m_z[0], &m_y[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_y[i] = m_z[i];

        for (i = 0u; i < nx; ++i) {
            m_y[i] *= std::pow(alpha, (1. * i) / (nx - 1u) / 2.);
        }

        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        f[0] = 0.0;
        for (i = 0u; i < nx; ++i) {
            f[0] += (m_z[i] * m_z[i] - 10.0 * std::cos(2.0 * detail::pi() * m_z[i]) + 10.0);
        }
    }

    void schwefel_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                       int r_flag) const /* Schwefel's  */
    {
        unsigned int i;
        double tmp;
        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] *= 1000. / 100.;
        }
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        for (i = 0u; i < nx; ++i)
            m_y[i] = m_z[i] * std::pow(10.0, (1. * i) / (nx - 1u) / 2.0);

        for (i = 0u; i < nx; ++i)
            m_z[i] = m_y[i] + 4.209687462275036e+002;

        f[0] = 0;
        for (i = 0u; i < nx; ++i) {
            if (m_z[i] > 500) {
                f[0] -= (500.0 - std::fmod(m_z[i], 500)) * std::sin(std::pow(500.0 - std::fmod(m_z[i], 500), 0.5));
                tmp = (m_z[i] - 500.0) / 100;
                f[0] += tmp * tmp / nx;
            } else if (m_z[i] < -500) {
                f[0] -= (-500.0 + std::fmod(std::abs(m_z[i]), 500))
                        * std::sin(std::pow(500.0 - std::fmod(std::abs(m_z[i]), 500), 0.5));
                tmp = (m_z[i] + 500.0) / 100;
                f[0] += tmp * tmp / nx;
            } else
                f[0] -= m_z[i] * std::sin(std::pow(std::abs(m_z[i]), 0.5));
        }
        f[0] = 4.189828872724338e+002 * nx + f[0];
    }

    void katsuura_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                       int r_flag) const /* Katsuura  */
    {
        unsigned int i, j;
        double temp, tmp1, tmp2, tmp3;
        tmp3 = std::pow(1.0 * nx, 1.2);
        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] *= 5.0 / 100.0;
        }
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        for (i = 0u; i < nx; ++i)
            m_z[i] *= std::pow(100.0, (1. * i) / (nx - 1u) / 2.0);

        if (r_flag == 1)
            rotatefunc(&m_z[0], &m_y[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_y[i] = m_z[i];

        f[0] = 1.0;
        for (i = 0u; i < nx; ++i) {
            temp = 0.0;
            for (j = 1u; j <= 32u; ++j) {
                tmp1 = std::pow(2.0, j);
                tmp2 = tmp1 * m_y[i];
                temp += std::abs(tmp2 - std::floor(tmp2 + 0.5)) / tmp1;
            }
            f[0] *= std::pow(1.0 + (i + 1u) * temp, 10.0 / tmp3);
        }
        tmp1 = 10.0 / nx / nx;
        f[0] = f[0] * tmp1 - tmp1;
    }

    void bi_rastrigin_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                           int r_flag) const /* Lunacek Bi_rastrigin Function */
    {
        unsigned int i;
        double mu0 = 2.5, d = 1.0, s, mu1, tmp, tmp1, tmp2;
        double *tmpx;
        tmpx = static_cast<double *>(malloc(sizeof(double) * nx));
        s = 1.0 - 1.0 / (2.0 * std::pow(nx + 20.0, 0.5) - 8.2);
        mu1 = -std::pow((mu0 * mu0 - d) / s, 0.5);

        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] *= 10.0 / 100.0;
        }

        for (i = 0u; i < nx; ++i) {
            tmpx[i] = 2 * m_y[i];
            if (Os[i] < 0.) tmpx[i] *= -1.;
        }

        for (i = 0u; i < nx; ++i) {
            m_z[i] = tmpx[i];
            tmpx[i] += mu0;
        }
        if (r_flag == 1)
            rotatefunc(&m_z[0], &m_y[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_y[i] = m_z[i];

        for (i = 0u; i < nx; ++i)
            m_y[i] *= std::pow(100.0, (1. * i) / (nx - 1u) / 2.0);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        tmp1 = 0.0;
        tmp2 = 0.0;
        for (i = 0u; i < nx; ++i) {
            tmp = tmpx[i] - mu0;
            tmp1 += tmp * tmp;
            tmp = tmpx[i] - mu1;
            tmp2 += tmp * tmp;
        }
        tmp2 *= s;
        tmp2 += d * nx;
        tmp = 0;
        for (i = 0u; i < nx; ++i) {
            tmp += std::cos(2.0 * detail::pi() * m_z[i]);
        }

        if (tmp1 < tmp2)
            f[0] = tmp1;
        else
            f[0] = tmp2;
        f[0] += 10.0 * (nx - tmp);
        free(tmpx);
    }

    void grie_rosen_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                         int r_flag) const /* Griewank-Rosenbrock  */
    {
        unsigned int i;
        double temp, tmp1, tmp2;

        shiftfunc(x, &m_y[0], nx, Os);
        for (i = 0u; i < nx; ++i) // shrink to the orginal search range
        {
            m_y[i] = m_y[i] * 5 / 100;
        }
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        for (i = 0u; i < nx; ++i) // shift to orgin
        {
            m_z[i] = m_y[i] + 1;
        }

        f[0] = 0.0;
        for (i = 0u; i < nx - 1u; ++i) {
            tmp1 = m_z[i] * m_z[i] - m_z[i + 1];
            tmp2 = m_z[i] - 1.0;
            temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
            f[0] += (temp * temp) / 4000.0 - std::cos(temp) + 1.0;
        }
        tmp1 = m_z[nx - 1] * m_z[nx - 1] - m_z[0];
        tmp2 = m_z[nx - 1] - 1.0;
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
        ;
        f[0] += (temp * temp) / 4000.0 - std::cos(temp) + 1.0;
    }

    void escaffer6_func(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
                        int r_flag) const /* Expanded Scaffer¡¯s F6  */
    {
        unsigned int i;
        double temp1, temp2;
        shiftfunc(x, &m_y[0], nx, Os);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, Mr);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        asyfunc(&m_z[0], &m_y[0], nx, 0.5);
        if (r_flag == 1)
            rotatefunc(&m_y[0], &m_z[0], nx, &Mr[nx * nx]);
        else
            for (i = 0u; i < nx; ++i)
                m_z[i] = m_y[i];

        f[0] = 0.0;
        for (i = 0u; i < nx - 1u; ++i) {
            temp1 = std::sin(std::sqrt(m_z[i] * m_z[i] + m_z[i + 1] * m_z[i + 1]));
            temp1 = temp1 * temp1;
            temp2 = 1.0 + 0.001 * (m_z[i] * m_z[i] + m_z[i + 1] * m_z[i + 1]);
            f[0] += 0.5 + (temp1 - 0.5) / (temp2 * temp2);
        }
        temp1 = std::sin(std::sqrt(m_z[nx - 1] * m_z[nx - 1] + m_z[0] * m_z[0]));
        temp1 = temp1 * temp1;
        temp2 = 1.0 + 0.001 * (m_z[nx - 1] * m_z[nx - 1] + m_z[0] * m_z[0]);
        f[0] += 0.5 + (temp1 - 0.5) / (temp2 * temp2);
    }

    void cf01(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 1 */
    {
        unsigned int i, cf_num = 5;
        double fit[5];
        double delta[5] = {10, 20, 30, 40, 50};
        double bias[5] = {0, 100, 200, 300, 400};

        i = 0u;
        rosenbrock_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 1e+4;
        i = 1u;
        dif_powers_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 1e+10;
        i = 2u;
        bent_cigar_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 1e+30;
        i = 3u;
        discus_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 1e+10;
        i = 4u;
        sphere_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 0);
        fit[i] = 10000 * fit[i] / 1e+5;
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void cf02(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 2 */
    {
        unsigned int i, cf_num = 3u;
        double fit[3];
        double delta[3] = {20, 20, 20};
        double bias[3] = {0, 100, 200};
        for (i = 0u; i < cf_num; ++i) {
            schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        }
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void cf03(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 3 */
    {
        unsigned int i, cf_num = 3u;
        double fit[3];
        double delta[3] = {20, 20, 20};
        double bias[3] = {0, 100, 200};
        for (i = 0u; i < cf_num; ++i) {
            schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        }
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void cf04(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 4 */
    {
        unsigned int i, cf_num = 3u;
        double fit[3];
        double delta[3] = {20, 20, 20};
        double bias[3] = {0, 100, 200};
        i = 0u;
        schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 4e+3;
        i = 1u;
        rastrigin_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 1e+3;
        i = 2u;
        weierstrass_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 400;
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void cf05(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 4 */
    {
        unsigned int i, cf_num = 3u;
        double fit[3];
        double delta[3] = {10, 30, 50};
        double bias[3] = {0, 100, 200};
        i = 0u;
        schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 4e+3;
        i = 1u;
        rastrigin_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 1e+3;
        i = 2u;
        weierstrass_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 400;
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void cf06(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 6 */
    {
        unsigned int i, cf_num = 5u;
        double fit[5];
        double delta[5] = {10, 10, 10, 10, 10};
        double bias[5] = {0, 100, 200, 300, 400};
        i = 0u;
        schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 4e+3;
        i = 1u;
        rastrigin_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 1e+3;
        i = 2u;
        ellips_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 1e+10;
        i = 3u;
        weierstrass_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 400;
        i = 4u;
        griewank_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 1000 * fit[i] / 100;
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void cf07(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 7 */
    {
        unsigned int i, cf_num = 5u;
        double fit[5];
        double delta[5] = {10, 10, 10, 20, 20};
        double bias[5] = {0, 100, 200, 300, 400};
        i = 0u;
        griewank_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 100;
        i = 1u;
        rastrigin_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 1e+3;
        i = 2u;
        schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 4e+3;
        i = 3u;
        weierstrass_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 400;
        i = 4u;
        sphere_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 0);
        fit[i] = 10000 * fit[i] / 1e+5;
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void cf08(const double *x, double *f, const unsigned int nx, const double *Os, const double *Mr,
              int r_flag) const /* Composition Function 8 */
    {
        unsigned int i, cf_num = 5u;
        double fit[5];
        double delta[5] = {10, 20, 30, 40, 50};
        double bias[5] = {0, 100, 200, 300, 400};
        i = 0u;
        grie_rosen_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 4e+3;
        i = 1u;
        schaffer_F7_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 4e+6;
        i = 2u;
        schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 4e+3;
        i = 3u;
        escaffer6_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], r_flag);
        fit[i] = 10000 * fit[i] / 2e+7;
        i = 4u;
        sphere_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 0);
        fit[i] = 10000 * fit[i] / 1e+5;
        cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
    }

    void shiftfunc(const double *x, double *xshift, const unsigned int nx, const double *Os) const
    {
        unsigned int i;
        for (i = 0u; i < nx; ++i) {
            xshift[i] = x[i] - Os[i];
        }
    }

    void rotatefunc(const double *x, double *xrot, const unsigned int nx, const double *Mr) const
    {
        unsigned int i, j;
        for (i = 0u; i < nx; ++i) {
            xrot[i] = 0;
            for (j = 0u; j < nx; ++j) {
                xrot[i] = xrot[i] + x[j] * Mr[i * nx + j];
            }
        }
    }

    void asyfunc(const double *x, double *xasy, const unsigned int nx, double beta) const
    {
        unsigned int i;
        for (i = 0u; i < nx; ++i) {
            if (x[i] > 0) xasy[i] = std::pow(x[i], 1.0 + (beta * i) / (nx - 1u) * std::pow(x[i], 0.5));
        }
    }

    void oszfunc(const double *x, double *xosz, const unsigned int nx) const
    {
        unsigned int i;
        int sx;
        double c1, c2, xx = 0;
        for (i = 0u; i < nx; ++i) {
            if (i == 0u || i == nx - 1u) {
                if (x[i] != 0) xx = std::log(std::abs(x[i]));
                if (x[i] > 0) {
                    c1 = 10;
                    c2 = 7.9;
                } else {
                    c1 = 5.5;
                    c2 = 3.1;
                }
                if (x[i] > 0)
                    sx = 1;
                else if (x[i] == 0)
                    sx = 0;
                else
                    sx = -1;
                xosz[i] = sx * std::exp(xx + 0.049 * (std::sin(c1 * xx) + std::sin(c2 * xx)));
            } else
                xosz[i] = x[i];
        }
    }

    void cf_cal(const double *x, double *f, const unsigned int nx, const double *Os, double *delta, double *bias,
                double *fit, unsigned int cf_num) const
    {
        unsigned int i, j;
        double *w;
        double w_max = 0, w_sum = 0;
        w = static_cast<double *>(malloc(cf_num * sizeof(double)));
        for (i = 0u; i < cf_num; ++i) {
            fit[i] += bias[i];
            w[i] = 0;
            for (j = 0u; j < nx; ++j) {
                w[i] += std::pow(x[j] - Os[i * nx + j], 2.0);
            }
            if (w[i] != 0)
                w[i] = std::pow(1.0 / w[i], 0.5) * std::exp(-w[i] / 2.0 / nx / std::pow(delta[i], 2.0));
            else
                w[i] = 1.0e99;
            if (w[i] > w_max) w_max = w[i];
        }

        for (i = 0u; i < cf_num; ++i) {
            w_sum = w_sum + w[i];
        }
        if (w_max == 0) {
            for (i = 0u; i < cf_num; ++i)
                w[i] = 1;
            w_sum = cf_num;
        }
        f[0] = 0.0;
        for (i = 0u; i < cf_num; ++i) {
            f[0] = f[0] + w[i] / w_sum * fit[i];
        }
        free(w);
    }
    // LCOV_EXCL_STOP

    // problem id
    unsigned int m_prob_id;
    // problem data
    std::vector<double> m_rotation_matrix;
    std::vector<double> m_origin_shift;

    // pre-allocated stuff for speed
    mutable std::vector<double> m_y;
    mutable std::vector<double> m_z;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::cec2013)

#undef E

#endif
