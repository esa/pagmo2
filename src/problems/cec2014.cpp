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

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2014.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

#include "cec2014_data.hpp"

namespace pagmo
{

namespace
{

// "Research code", ladies and gentlemen!
constexpr double INF = 1.0e99;
constexpr double E = 2.7182818284590452353602874713526625;
constexpr double PI = 3.1415926535897932384626433832795029;

} // namespace

cec2014::cec2014(unsigned prob_id, unsigned dim) : m_z(dim), m_y(dim), func_num(prob_id)
{
    if (!(dim == 2u || dim == 10u || dim == 20u || dim == 30u || dim == 50u || dim == 100u)) {
        pagmo_throw(std::invalid_argument, "Error: CEC2014 Test functions are only defined for dimensions "
                                           "2,10,20,30,50,100, a dimension of "
                                               + std::to_string(dim) + " was detected.");
    }
    if (prob_id < 1u || prob_id > 30u) {
        pagmo_throw(std::invalid_argument,
                    "Error: CEC2014 Test functions are only defined for prob_id in [1, 28], a prob_id of "
                        + std::to_string(prob_id) + " was detected.");
    }

    if (dim == 2 && ((func_num >= 17u && func_num <= 22u) || (func_num >= 29u && func_num <= 30u))) {
        pagmo_throw(std::invalid_argument, "hf01,hf02,hf03,hf04,hf05,hf06,cf07&cf08 are NOT defined for D=2.");
    }

    /* Load Rotation Matrix */
    auto rotation_func_it = detail::cec2014_data::rotation_data.find(func_num);
    auto rotation_data_dim = rotation_func_it->second;
    auto rotation_dim_it = rotation_data_dim.find(dim);
    m_rotation_matrix = rotation_dim_it->second;

    /* Load shift_data */
    auto loader_it = detail::cec2014_data::shift_data.find(func_num);
    m_origin_shift = loader_it->second;

    // Uses first dim elements of each line for multidimensional functions (id > 23)
    auto it = m_origin_shift.begin();
    int i = -1;
    while (it != m_origin_shift.end()) {
        i++;
        if ((i % 100) >= static_cast<int>(dim)) {
            it = m_origin_shift.erase(it);
        } else {
            ++it;
        }
    }

    /* Load shuffle data */
    if (((func_num >= 17) && (func_num <= 22)) || (func_num == 29) || (func_num == 30)) {
        auto shuffle_func_it = detail::cec2014_data::shuffle_data.find(func_num);
        auto shuffle_data_dim = shuffle_func_it->second;
        auto shuffle_dim_it = shuffle_data_dim.find(dim);
        m_shuffle = shuffle_dim_it->second;
    }
}

/// Box-bounds
/**
 * It returns the box-bounds for this UDP.
 *
 * @return the lower and upper bounds for each of the decision vector components
 */
std::pair<vector_double, vector_double> cec2014::get_bounds() const
{
    // all CEC 2014 problems have the same bounds
    vector_double lb(m_z.size(), -100.);
    vector_double ub(m_z.size(), 100.);
    return std::make_pair(std::move(lb), std::move(ub));
}

/// Fitness computation
/**
 * Computes the fitness for this UDP
 *
 * @param x the decision vector.
 *
 * @return the fitness of \p x.
 */
vector_double cec2014::fitness(const vector_double &x) const
{
    vector_double f(1);
    auto nx = static_cast<unsigned>(m_z.size());
    switch (func_num) {
        case 1:
            ellips_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 100.0;
            break;
        case 2:
            bent_cigar_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 200.0;
            break;
        case 3:
            discus_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 300.0;
            break;
        case 4:
            rosenbrock_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 400.0;
            break;
        case 5:
            ackley_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 500.0;
            break;
        case 6:
            weierstrass_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 600.0;
            break;
        case 7:
            griewank_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 700.0;
            break;
        case 8:
            rastrigin_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 0);
            f[0] += 800.0;
            break;
        case 9:
            rastrigin_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 900.0;
            break;
        case 10:
            schwefel_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 0);
            f[0] += 1000.0;
            break;
        case 11:
            schwefel_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 1100.0;
            break;
        case 12:
            katsuura_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 1200.0;
            break;
        case 13:
            happycat_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 1300.0;
            break;
        case 14:
            hgbat_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 1400.0;
            break;
        case 15:
            grie_rosen_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 1500.0;
            break;
        case 16:
            escaffer6_func(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1, 1);
            f[0] += 1600.0;
            break;
        case 17:
            hf01(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1, 1);
            f[0] += 1700.0;
            break;
        case 18:
            hf02(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1, 1);
            f[0] += 1800.0;
            break;
        case 19:
            hf03(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1, 1);
            f[0] += 1900.0;
            break;
        case 20:
            hf04(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1, 1);
            f[0] += 2000.0;
            break;
        case 21:
            hf05(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1, 1);
            f[0] += 2100.0;
            break;
        case 22:
            hf06(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1, 1);
            f[0] += 2200.0;
            break;
        case 23:
            cf01(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1);
            f[0] += 2300.0;
            break;
        case 24:
            cf02(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1);
            f[0] += 2400.0;
            break;
        case 25:
            cf03(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1);
            f[0] += 2500.0;
            break;
        case 26:
            cf04(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1);
            f[0] += 2600.0;
            break;
        case 27:
            cf05(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1);
            f[0] += 2700.0;
            break;
        case 28:
            cf06(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), 1);
            f[0] += 2800.0;
            break;
        case 29:
            cf07(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1);
            f[0] += 2900.0;
            break;
        case 30:
            cf08(x.data(), f.data(), nx, m_origin_shift.data(), m_rotation_matrix.data(), m_shuffle.data(), 1);
            f[0] += 3000.0;
            break;
    }

    return f;
}

/// Problem name
/**
 * @return a string containing the problem name
 */
std::string cec2014::get_name() const
{
    std::string retval("CEC2014 - f");
    retval.append(std::to_string(func_num));
    switch (func_num) {
        case 1:
            retval.append("(ellips_func)");
            break;
        case 2:
            retval.append("(bent_cigar_func)");
            break;
        case 3:
            retval.append("(discus_func)");
            break;
        case 4:
            retval.append("(rosenbrock_func)");
            break;
        case 5:
            retval.append("(ackley_func)");
            break;
        case 6:
            retval.append("(weierstrass_func)");
            break;
        case 7:
            retval.append("(griewank_func)");
            break;
        case 8:
            retval.append("(rastrigin_func_non_rotated)");
            break;
        case 9:
            retval.append("(rastrigin_func)");
            break;
        case 10:
            retval.append("(schwefel_func_non_rotated)");
            break;
        case 11:
            retval.append("(schwefel_func)");
            break;
        case 12:
            retval.append("(katsuura_func)");
            break;
        case 13:
            retval.append("(happycat_func)");
            break;
        case 14:
            retval.append("(hgbat_func)");
            break;
        case 15:
            retval.append("(grie_rosen_func)");
            break;
        case 16:
            retval.append("(escaffer6_func)");
            break;
        case 17:
            retval.append("(hf01)");
            break;
        case 18:
            retval.append("(hf02)");
            break;
        case 19:
            retval.append("(hf03)");
            break;
        case 20:
            retval.append("(hf04)");
            break;
        case 21:
            retval.append("(hf05)");
            break;
        case 22:
            retval.append("(hf06)");
            break;
        case 23:
            retval.append("(cf01)");
            break;
        case 24:
            retval.append("(cf02)");
            break;
        case 25:
            retval.append("(cf03)");
            break;
        case 26:
            retval.append("(cf04)");
            break;
        case 27:
            retval.append("(cf05)");
            break;
        case 28:
            retval.append("(cf06)");
            break;
        case 29:
            retval.append("(cf07)");
            break;
        case 30:
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
void cec2014::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, func_num, m_rotation_matrix, m_origin_shift, m_shuffle, m_y, m_z);
}

// For the coverage analysis we do not cover the code below as its derived from a third party source
// LCOV_EXCL_START
/* Sphere */
void cec2014::sphere_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int s_flag,
                          int r_flag) const
{

    unsigned i;
    f[0] = 0.0;
    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */
    for (i = 0; i < nx; i++) {
        f[0] += m_z[i] * m_z[i];
    }
}

/* Ellipsoidal */
void cec2014::ellips_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int s_flag,
                          int r_flag) const
{

    unsigned i;
    f[0] = 0.0;
    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */
    for (i = 0; i < nx; i++) {
        f[0] += std::pow(10.0, 6.0 * i / (nx - 1)) * m_z[i] * m_z[i];
    }
}

/* Bent_Cigar */
void cec2014::bent_cigar_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int s_flag, int r_flag) const
{

    unsigned i;
    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    f[0] = m_z[0] * m_z[0];
    for (i = 1; i < nx; i++) {
        f[0] += std::pow(10.0, 6.0) * m_z[i] * m_z[i];
    }
}

/* Discus */
void cec2014::discus_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int s_flag,
                          int r_flag) const
{

    unsigned i;
    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */
    f[0] = std::pow(10.0, 6.0) * m_z[0] * m_z[0];
    for (i = 1; i < nx; i++) {
        f[0] += m_z[i] * m_z[i];
    }
}

/* Different Powers */
void cec2014::dif_powers_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int s_flag, int r_flag) const
{

    unsigned i;
    f[0] = 0.0;
    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        f[0] += std::pow(std::abs(m_z[i]), 2 + 4 * i / (nx - 1));
    }
    f[0] = std::pow(f[0], 0.5);
}

/* Rosenbrock's */
void cec2014::rosenbrock_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int s_flag, int r_flag) const
{

    unsigned i;
    double tmp1, tmp2;
    f[0] = 0.0;
    sr_func(x, m_z.data(), nx, Os, Mr, 2.048 / 100.0, s_flag, r_flag); /* shift and rotate */
    m_z[0] += 1.0;                                                     // shift to orgin
    for (i = 0; i < nx - 1; i++) {
        m_z[i + 1] += 1.0; // shift to orgin
        tmp1 = m_z[i] * m_z[i] - m_z[i + 1];
        tmp2 = m_z[i] - 1.0;
        f[0] += 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
    }
}

/* Schwefel's 1.2  */
void cec2014::schaffer_F7_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                               int s_flag, int r_flag) const
{

    unsigned i;
    double tmp;
    f[0] = 0.0;
    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */
    for (i = 0; i < nx - 1; i++) {
        m_z[i] = std::pow(m_y[i] * m_y[i] + m_y[i + 1] * m_y[i + 1], 0.5);
        tmp = std::sin(50.0 * std::pow(m_z[i], 0.2));
        f[0] += std::pow(m_z[i], 0.5) + std::pow(m_z[i], 0.5) * tmp * tmp;
    }
    f[0] = f[0] * f[0] / (nx - 1) / (nx - 1);
}

/* Ackley's  */
void cec2014::ackley_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int s_flag,
                          int r_flag) const
{

    unsigned i;
    double sum1, sum2;
    sum1 = 0.0;
    sum2 = 0.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        sum1 += m_z[i] * m_z[i];
        sum2 += std::cos(2.0 * PI * m_z[i]);
    }
    sum1 = -0.2 * std::sqrt(sum1 / nx);
    sum2 /= nx;
    f[0] = E - 20.0 * std::exp(sum1) - std::exp(sum2) + 20.0;
}

/* Weierstrass's  */
void cec2014::weierstrass_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                               int s_flag, int r_flag) const
{

    unsigned i, j, k_max;
    double sum, sum2, a, b;
    a = 0.5;
    b = 3.0;
    k_max = 20;
    f[0] = 0.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 0.5 / 100.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        sum = 0.0;
        sum2 = 0.0;
        for (j = 0; j <= k_max; j++) {
            sum += std::pow(a, j) * std::cos(2.0 * PI * std::pow(b, j) * (m_z[i] + 0.5));
            sum2 += std::pow(a, j) * std::cos(2.0 * PI * std::pow(b, j) * 0.5);
        }
        f[0] += sum;
    }
    f[0] -= nx * sum2;
}

/* Griewank's  */
void cec2014::griewank_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                            int s_flag, int r_flag) const
{

    unsigned i;
    double s, p;
    s = 0.0;
    p = 1.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 600.0 / 100.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        s += m_z[i] * m_z[i];
        p *= std::cos(m_z[i] / std::sqrt(1.0 + i));
    }
    f[0] = 1.0 + s / 4000.0 - p;
}

/* Rastrigin's  */
void cec2014::rastrigin_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                             int s_flag, int r_flag) const
{

    unsigned i;
    f[0] = 0.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        f[0] += (m_z[i] * m_z[i] - 10.0 * std::cos(2.0 * PI * m_z[i]) + 10.0);
    }
}

/* Noncontinuous Rastrigin's  */
void cec2014::step_rastrigin_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                                  int s_flag, int r_flag) const
{

    unsigned i;
    f[0] = 0.0;
    for (i = 0; i < nx; i++) {
        if (fabs(m_y[i] - Os[i]) > 0.5) m_y[i] = Os[i] + std::floor(2 * (m_y[i] - Os[i]) + 0.5) / 2;
    }

    sr_func(x, m_z.data(), nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        f[0] += (m_z[i] * m_z[i] - 10.0 * std::cos(2.0 * PI * m_z[i]) + 10.0);
    }
}

/* Schwefel's  */
void cec2014::schwefel_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                            int s_flag, int r_flag) const
{

    unsigned i;
    double tmp;
    f[0] = 0.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 1000.0 / 100.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        m_z[i] += 4.209687462275036e+002;
        if (m_z[i] > 500) {
            f[0] -= (500.0 - std::fmod(m_z[i], 500)) * std::sin(std::pow(500.0 - std::fmod(m_z[i], 500), 0.5));
            tmp = (m_z[i] - 500.0) / 100;
            f[0] += tmp * tmp / nx;
        } else if (m_z[i] < -500) {
            f[0] -= (-500.0 + std::fmod(std::fabs(m_z[i]), 500))
                    * std::sin(std::pow(500.0 - std::fmod(std::fabs(m_z[i]), 500), 0.5));
            tmp = (m_z[i] + 500.0) / 100;
            f[0] += tmp * tmp / nx;
        } else
            f[0] -= m_z[i] * std::sin(std::pow(std::fabs(m_z[i]), 0.5));
    }
    f[0] += 4.189828872724338e+002 * nx;
}

/* Katsuura  */
void cec2014::katsuura_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                            int s_flag, int r_flag) const
{

    unsigned i, j;
    double temp, tmp1, tmp2, tmp3;
    f[0] = 1.0;
    tmp3 = std::pow(1.0 * nx, 1.2);

    sr_func(x, m_z.data(), nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag); /* shift and rotate */

    for (i = 0; i < nx; i++) {
        temp = 0.0;
        for (j = 1; j <= 32; j++) {
            tmp1 = std::pow(2.0, j);
            tmp2 = tmp1 * m_z[i];
            temp += std::abs(tmp2 - std::floor(tmp2 + 0.5)) / tmp1;
        }
        f[0] *= std::pow(1.0 + (i + 1) * temp, 10.0 / tmp3);
    }
    tmp1 = 10.0 / nx / nx;
    f[0] = f[0] * tmp1 - tmp1;
}

/* Lunacek Bi_rastrigin Function */
void cec2014::bi_rastrigin_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                                int s_flag, int r_flag) const
{

    unsigned i;
    double mu0 = 2.5, d = 1.0, s, mu1, tmp, tmp1, tmp2;
    double *tmpx;
    tmpx = static_cast<double *>(std::malloc(sizeof(double) * nx));
    s = 1.0 - 1.0 / (2.0 * std::pow(nx + 20.0, 0.5) - 8.2);
    mu1 = -std::pow((mu0 * mu0 - d) / s, 0.5);

    if (s_flag == 1) {
        shiftfunc(x, m_y.data(), nx, Os);
    } else {
        // shrink to the orginal search range
        for (i = 0; i < nx; i++) {
            m_y[i] = x[i];
        }
    }
    // shrink to the orginal search range
    for (i = 0; i < nx; i++) {
        m_y[i] *= 10.0 / 100.0;
    }

    for (i = 0; i < nx; i++) {
        tmpx[i] = 2 * m_y[i];
        if (Os[i] < 0.0) {
            tmpx[i] *= -1.;
        }
    }
    for (i = 0; i < nx; i++) {
        m_z[i] = tmpx[i];
        tmpx[i] += mu0;
    }
    tmp1 = 0.0;
    tmp2 = 0.0;
    for (i = 0; i < nx; i++) {
        tmp = tmpx[i] - mu0;
        tmp1 += tmp * tmp;
        tmp = tmpx[i] - mu1;
        tmp2 += tmp * tmp;
    }

    tmp2 *= s;
    tmp2 += d * nx;
    tmp = 0.0;

    if (r_flag == 1) {
        rotatefunc(m_z.data(), m_y.data(), nx, Mr);
        for (i = 0; i < nx; i++) {
            tmp += std::cos(2.0 * PI * m_y[i]);
        }
        if (tmp1 < tmp2) {
            f[0] = tmp1;
        } else {
            f[0] = tmp2;
        }
        f[0] += 10.0 * (nx - tmp);
    } else {
        for (i = 0; i < nx; i++) {
            tmp += std::cos(2.0 * PI * m_z[i]);
        }
        if (tmp1 < tmp2) {
            f[0] = tmp1;
        } else {
            f[0] = tmp2;
        }
        f[0] += 10.0 * (nx - tmp);
    }

    std::free(tmpx);
}

/* Griewank-Rosenbrock  */
void cec2014::grie_rosen_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int s_flag, int r_flag) const
{

    unsigned i;
    double temp, tmp1, tmp2;
    f[0] = 0.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag); /* shift and rotate */

    m_z[0] += 1.0; // shift to orgin
    for (i = 0; i < nx - 1; i++) {
        m_z[i + 1] += 1.0; // shift to orgin
        tmp1 = m_z[i] * m_z[i] - m_z[i + 1];
        tmp2 = m_z[i] - 1.0;
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
        f[0] += (temp * temp) / 4000.0 - std::cos(temp) + 1.0;
    }
    tmp1 = m_z[nx - 1] * m_z[nx - 1] - m_z[0];
    tmp2 = m_z[nx - 1] - 1.0;
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
    f[0] += (temp * temp) / 4000.0 - std::cos(temp) + 1.0;
}

/* Expanded Scaffer??s F6  */
void cec2014::escaffer6_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                             int s_flag, int r_flag) const
{

    unsigned i;
    double temp1, temp2;

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    f[0] = 0.0;
    for (i = 0; i < nx - 1; i++) {
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

/* HappyCat, provdided by Hans-Georg Beyer (HGB) */
/* original global optimum: [-1,-1,...,-1] */
void cec2014::happycat_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                            int s_flag, int r_flag) const
{

    unsigned i;
    double alpha, r2, sum_z;
    alpha = 1.0 / 8.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag); /* shift and rotate */

    r2 = 0.0;
    sum_z = 0.0;
    for (i = 0; i < nx; i++) {
        m_z[i] = m_z[i] - 1.0; // shift to orgin
        r2 += m_z[i] * m_z[i];
        sum_z += m_z[i];
    }

    f[0] = std::pow(std::abs(r2 - nx), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5;
}

/* HGBat, provdided by Hans-Georg Beyer (HGB)*/
/* original global optimum: [-1,-1,...,-1] */
void cec2014::hgbat_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int s_flag,
                         int r_flag) const
{

    unsigned i;
    double alpha, r2, sum_z;
    alpha = 1.0 / 4.0;

    sr_func(x, m_z.data(), nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag); /* shift and rotate */

    r2 = 0.0;
    sum_z = 0.0;
    for (i = 0; i < nx; i++) {
        m_z[i] = m_z[i] - 1.0; // shift to orgin
        r2 += m_z[i] * m_z[i];
        sum_z += m_z[i];
    }

    f[0] = std::pow(std::abs(std::pow(r2, 2.0) - std::pow(sum_z, 2.0)), 2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5;
}

/* Hybrid Function 1 */
void cec2014::hf01(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *S,
                   int s_flag, int r_flag) const
{
    unsigned i, tmp, cf_num = 3;
    double fit[3];
    unsigned G[3], G_nx[3];
    double Gp[3] = {0.3, 0.3, 0.4};

    tmp = 0;
    for (i = 0; i < cf_num - 1; i++) {
        G_nx[i] = static_cast<unsigned>(std::ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;
    G[0] = 0;
    for (i = 1; i < cf_num; i++) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (auto j = 0u; j < nx; j++) {
        m_y[j] = m_z[static_cast<unsigned>(S[j] - 1)];
    }
    i = 0;
    schwefel_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 1;
    rastrigin_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 2;
    ellips_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    f[0] = 0.0;
    for (i = 0; i < cf_num; i++) {
        f[0] += fit[i];
    }
}

/* Hybrid Function 2 */
void cec2014::hf02(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *S,
                   int s_flag, int r_flag) const
{
    unsigned i, tmp, cf_num = 3;
    double fit[3];
    unsigned G[3], G_nx[3];
    double Gp[3] = {0.3, 0.3, 0.4};

    tmp = 0;
    for (i = 0; i < cf_num - 1; i++) {
        G_nx[i] = static_cast<unsigned>(std::ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;

    G[0] = 0;
    for (i = 1; i < cf_num; i++) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (auto j = 0u; j < nx; j++) {
        m_y[j] = m_z[static_cast<unsigned>(S[j] - 1)];
    }
    i = 0;
    bent_cigar_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 1;
    hgbat_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 2;
    rastrigin_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);

    f[0] = 0.0;
    for (i = 0; i < cf_num; i++) {
        f[0] += fit[i];
    }
}

/* Hybrid Function 3 */
void cec2014::hf03(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *S,
                   int s_flag, int r_flag) const
{

    unsigned i, tmp, cf_num = 4;
    double fit[4];
    unsigned G[4], G_nx[4];
    double Gp[4] = {0.2, 0.2, 0.3, 0.3};

    tmp = 0;
    for (i = 0; i < cf_num - 1; i++) {
        G_nx[i] = static_cast<unsigned>(std::ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;

    G[0] = 0;
    for (i = 1; i < cf_num; i++) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (auto j = 0u; j < nx; j++) {
        m_y[j] = m_z[static_cast<unsigned>(S[j] - 1)];
    }
    i = 0;
    griewank_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 1;
    weierstrass_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 2;
    rosenbrock_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 3;
    escaffer6_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);

    f[0] = 0.0;
    for (i = 0; i < cf_num; i++) {
        f[0] += fit[i];
    }
}

/* Hybrid Function 4 */
void cec2014::hf04(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *S,
                   int s_flag, int r_flag) const
{

    unsigned i, tmp, cf_num = 4;
    double fit[4];
    unsigned G[4], G_nx[4];
    double Gp[4] = {0.2, 0.2, 0.3, 0.3};

    tmp = 0;
    for (i = 0; i < cf_num - 1; i++) {
        G_nx[i] = static_cast<unsigned>(std::ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }
    G_nx[cf_num - 1] = nx - tmp;

    G[0] = 0;
    for (i = 1; i < cf_num; i++) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (auto j = 0u; j < nx; j++) {
        m_y[j] = m_z[static_cast<unsigned>(S[j] - 1)];
    }
    i = 0;
    hgbat_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 1;
    discus_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 2;
    grie_rosen_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 3;
    rastrigin_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);

    f[0] = 0.0;
    for (i = 0; i < cf_num; i++) {
        f[0] += fit[i];
    }
}

/* Hybrid Function 5 */
void cec2014::hf05(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *S,
                   int s_flag, int r_flag) const
{

    unsigned i, tmp, cf_num = 5;
    double fit[5];
    unsigned G[5], G_nx[5];
    double Gp[5] = {0.1, 0.2, 0.2, 0.2, 0.3};

    tmp = 0;
    for (i = 0; i < cf_num - 1; i++) {
        G_nx[i] = static_cast<unsigned>(std::ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }

    G_nx[cf_num - 1] = nx - tmp;

    G[0] = 0;
    for (i = 1; i < cf_num; i++) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (auto j = 0u; j < nx; j++) {
        m_y[j] = m_z[static_cast<unsigned>(S[j] - 1)];
    }

    i = 0;
    escaffer6_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 1;
    hgbat_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 2;
    rosenbrock_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 3;
    schwefel_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 4;
    ellips_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);

    f[0] = 0.0;
    for (i = 0; i < cf_num; i++) {
        f[0] += fit[i];
    }
}

/* Hybrid Function 6 */
void cec2014::hf06(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *S,
                   int s_flag, int r_flag) const
{

    unsigned i, tmp, cf_num = 5;
    double fit[5];
    unsigned G[5], G_nx[5];
    double Gp[5] = {0.1, 0.2, 0.2, 0.2, 0.3};

    tmp = 0;
    for (i = 0; i < cf_num - 1; i++) {
        G_nx[i] = static_cast<unsigned>(std::ceil(Gp[i] * nx));
        tmp += G_nx[i];
    }

    G_nx[cf_num - 1] = nx - tmp;

    G[0] = 0;
    for (i = 1; i < cf_num; i++) {
        G[i] = G[i - 1] + G_nx[i - 1];
    }

    sr_func(x, m_z.data(), nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */

    for (auto j = 0u; j < nx; j++) {
        m_y[j] = m_z[static_cast<unsigned>(S[j] - 1)];
    }

    i = 0;
    katsuura_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 1;
    happycat_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 2;
    grie_rosen_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 3;
    schwefel_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    i = 4;
    ackley_func(&m_y[G[i]], &fit[i], G_nx[i], Os, Mr, 0, 0);
    f[0] = 0.0;
    for (i = 0; i < cf_num; i++) {
        f[0] += fit[i];
    }
}

/* Composition Function 1 */
void cec2014::cf01(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int r_flag) const
{
    unsigned i;
    int cf_num = 5;
    double fit[5];
    double delta[5] = {10, 20, 30, 40, 50};
    double bias[5] = {0, 100, 200, 300, 400};

    i = 0;
    rosenbrock_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+4;
    i = 1;
    ellips_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+10;
    i = 2;
    bent_cigar_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+30;
    i = 3;
    discus_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+10;
    i = 4;
    ellips_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, 0);
    fit[i] = 10000 * fit[i] / 1e+10;
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

/* Composition Function 2 */
void cec2014::cf02(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int r_flag) const
{
    unsigned i;
    int cf_num = 3;
    double fit[3];
    double delta[3] = {20, 20, 20};
    double bias[3] = {0, 100, 200};

    i = 0;
    schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, 0);
    i = 1;
    rastrigin_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    i = 2;
    hgbat_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

/* Composition Function 3 */
void cec2014::cf03(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int r_flag) const
{
    unsigned i;
    int cf_num = 3;
    double fit[3];
    double delta[3] = {10, 30, 50};
    double bias[3] = {0, 100, 200};
    i = 0;
    schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 4e+3;
    i = 1;
    rastrigin_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 1e+3;
    i = 2;
    ellips_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 1e+10;
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

/* Composition Function 4 */
void cec2014::cf04(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int r_flag) const
{
    unsigned i;
    int cf_num = 5;
    double fit[5];
    double delta[5] = {10, 10, 10, 10, 10};
    double bias[5] = {0, 100, 200, 300, 400};
    i = 0;
    schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 4e+3;
    i = 1;
    happycat_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 1e+3;
    i = 2;
    ellips_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 1e+10;
    i = 3;
    weierstrass_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 400;
    i = 4;
    griewank_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 1000 * fit[i] / 100;
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

/* Composition Function 4 */
void cec2014::cf05(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int r_flag) const
{
    unsigned i;
    int cf_num = 5;
    double fit[5];
    double delta[5] = {10, 10, 10, 20, 20};
    double bias[5] = {0, 100, 200, 300, 400};
    i = 0;
    hgbat_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1000;
    i = 1;
    rastrigin_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+3;
    i = 2;
    schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 4e+3;
    i = 3;
    weierstrass_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 400;
    i = 4;
    ellips_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+10;
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

/* Composition Function 6 */
void cec2014::cf06(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, int r_flag) const
{
    unsigned i;
    int cf_num = 5;
    double fit[5];
    double delta[5] = {10, 20, 30, 40, 50};
    double bias[5] = {0, 100, 200, 300, 400};
    i = 0;
    grie_rosen_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 4e+3;
    i = 1;
    happycat_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+3;
    i = 2;
    schwefel_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 4e+3;
    i = 3;
    escaffer6_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 2e+7;
    i = 4;
    ellips_func(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], 1, r_flag);
    fit[i] = 10000 * fit[i] / 1e+10;
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

/* Composition Function 7 */
void cec2014::cf07(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *SS,
                   int r_flag) const
{
    unsigned i;
    int cf_num = 3;
    double fit[3];
    double delta[3] = {10, 30, 50};
    double bias[3] = {0, 100, 200};
    i = 0;
    hf01(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], &SS[i * nx], 1, r_flag);
    i = 1;
    hf02(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], &SS[i * nx], 1, r_flag);
    i = 2;
    hf03(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], &SS[i * nx], 1, r_flag);
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

/* Composition Function 8 */
void cec2014::cf08(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr, const int *SS,
                   int r_flag) const
{
    unsigned i;
    int cf_num = 3;
    double fit[3];
    double delta[3] = {10, 30, 50};
    double bias[3] = {0, 100, 200};
    i = 0;
    hf04(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], &SS[i * nx], 1, r_flag);
    i = 1;
    hf05(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], &SS[i * nx], 1, r_flag);
    i = 2;
    hf06(x, &fit[i], nx, &Os[i * nx], &Mr[i * nx * nx], &SS[i * nx], 1, r_flag);
    cf_cal(x, f, nx, Os, delta, bias, fit, cf_num);
}

void cec2014::shiftfunc(const double *x, double *xshift, const unsigned nx, const double *Os) const
{

    unsigned i;
    for (i = 0; i < nx; i++) {
        xshift[i] = x[i] - Os[i];
    }
}

void cec2014::rotatefunc(const double *x, double *xrot, const unsigned nx, const double *Mr) const
{
    unsigned j;
    unsigned i;
    for (i = 0; i < nx; i++) {
        xrot[i] = 0;

        for (j = 0; j < nx; j++) {
            xrot[i] = xrot[i] + x[j] * Mr[i * nx + j];
        }
    }
}

/* shift and rotate */
void cec2014::sr_func(const double *x, double *sr_x, const unsigned nx, const double *Os, const double *Mr,
                      double sh_rate, int s_flag, int r_flag) const
{

    unsigned i;
    if (s_flag == 1) {
        if (r_flag == 1) {
            shiftfunc(x, m_y.data(), nx, Os);

            // shrink to the original search range
            for (i = 0; i < nx; i++) {
                m_y[i] = m_y[i] * sh_rate;
            }
            rotatefunc(m_y.data(), sr_x, nx, Mr);
        } else {
            shiftfunc(x, sr_x, nx, Os);

            // shrink to the original search range
            for (i = 0; i < nx; i++) {
                sr_x[i] = sr_x[i] * sh_rate;
            }
        }
    } else {
        if (r_flag == 1) {
            // shrink to the original search range
            for (i = 0; i < nx; i++) {
                m_y[i] = x[i] * sh_rate;
            }
            rotatefunc(m_y.data(), sr_x, nx, Mr);
        } else {
            // shrink to the original search range
            for (i = 0; i < nx; i++) {
                sr_x[i] = x[i] * sh_rate;
            }
        }
    }
}

void cec2014::asyfunc(const double *x, double *xasy, const unsigned nx, double beta) const
{

    unsigned i;
    for (i = 0; i < nx; i++) {
        if (x[i] > 0) {
            xasy[i] = std::pow(x[i], 1.0 + beta * i / (nx - 1) * std::pow(x[i], 0.5));
        }
    }
}

void cec2014::oszfunc(const double *x, double *xosz, const unsigned nx) const
{
    unsigned i;
    int sx;
    double c1, c2, xx;
    for (i = 0; i < nx; i++) {
        if (i == 0 || i == nx - 1) {
            if (x[i] != 0) {
                xx = std::log(std::abs(x[i]));
            }
            if (x[i] > 0) {
                c1 = 10;
                c2 = 7.9;
            } else {
                c1 = 5.5;
                c2 = 3.1;
            }
            if (x[i] > 0) {
                sx = 1;
            } else if (x[i] == 0) {
                sx = 0;
            } else {
                sx = -1;
            }
            xosz[i] = sx * std::exp(xx + 0.049 * (std::sin(c1 * xx) + std::sin(c2 * xx)));
        } else {
            xosz[i] = x[i];
        }
    }
}

void cec2014::cf_cal(const double *x, double *f, const unsigned nx, const double *Os, double *delta, double *bias,
                     double *fit, int cf_num) const
{
    unsigned j;
    unsigned i;
    double *w;
    double w_max = 0, w_sum = 0;
    w = static_cast<double *>(std::malloc(static_cast<unsigned>(cf_num) * sizeof(double)));
    for (i = 0; i < static_cast<unsigned>(cf_num); i++) {
        fit[i] += bias[i];
        w[i] = 0;
        for (j = 0; j < nx; j++) {
            w[i] += std::pow(x[j] - Os[i * nx + j], 2.0);
        }
        if (w[i] != 0)
            w[i] = std::pow(1.0 / w[i], 0.5) * std::exp(-w[i] / 2.0 / nx / std::pow(delta[i], 2.0));
        else
            w[i] = INF;
        if (w[i] > w_max) w_max = w[i];
    }

    for (i = 0; i < static_cast<unsigned>(cf_num); i++) {
        w_sum = w_sum + w[i];
    }
    if (w_max == 0) {
        for (i = 0; i < static_cast<unsigned>(cf_num); i++)
            w[i] = 1;
        w_sum = cf_num;
    }
    f[0] = 0.0;
    for (i = 0; i < static_cast<unsigned>(cf_num); i++) {
        f[0] = f[0] + w[i] / w_sum * fit[i];
    }
    std::free(w);
}
// LCOV_EXCL_STOP

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::cec2014)
