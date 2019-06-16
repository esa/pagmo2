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

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <pagmo/detail/constants.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/wfg.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#pragma GCC diagnostic ignored "-Wsuggest-attribute=const"
#endif

namespace pagmo
{

wfg::wfg(unsigned prob_id, vector_double::size_type dim_dvs, vector_double::size_type dim_obj,
         vector_double::size_type dim_k)
    : m_prob_id(prob_id), m_dim_dvs(dim_dvs), m_dim_obj(dim_obj), m_dim_k(dim_k)
{

    if (prob_id == 0u || prob_id > 9u) {
        pagmo_throw(std::invalid_argument, "WFG test suite contains nine (prob_id=[1 ... 9]) problems, prob_id="
                                               + std::to_string(prob_id) + " was detected");
    }
    if (dim_dvs < 1u) {
        pagmo_throw(std::invalid_argument, "WFG problem suite must have minimum 1 dimension for the decision vector, "
                                               + std::to_string(dim_dvs) + " requested");
    }

    if (dim_obj < 2u) {
        pagmo_throw(std::invalid_argument,
                    "WFG test problems must have a minimum value of 2 for the objective vector dimension, "
                        + std::to_string(dim_obj) + " requested");
    }

    if (dim_k >= dim_dvs || dim_k < 1 || dim_k % (dim_obj - 1) != 0) {
        pagmo_throw(std::invalid_argument,
                    "WFG test problems must have a dim_k parameter which is within [1,dim_dvs), and such that dim_k "
                    "mod(dim_obj-1) == 0 "
                        + std::to_string(dim_k) + " requested");
    }
    if (prob_id == 2u || prob_id == 3u) {
        if ((dim_dvs - dim_k) % 2 != 0) {
            pagmo_throw(std::invalid_argument,
                        "For problems WFG2 and WFG3 the dim_k parameter and the decision vector size must satisfy "
                        "(dim_dvs-dim_k) mod(2)=0"
                            + std::to_string((dim_dvs - dim_k) % 2) + " was detected");
        }
    }
}

// Fitness computation
vector_double wfg::fitness(const vector_double &x) const
{
    vector_double retval;
    switch (m_prob_id) {
        case 1u:
            retval = wfg1_fitness(x);
            break;
        case 2u:
            retval = wfg2_fitness(x);
            break;
        case 3u:
            retval = wfg3_fitness(x);
            break;
        case 4u:
            retval = wfg4_fitness(x);
            break;
        case 5u:
            retval = wfg5_fitness(x);
            break;
        case 6u:
            retval = wfg6_fitness(x);
            break;
        case 7u:
            retval = wfg7_fitness(x);
            break;
        case 8u:
            retval = wfg8_fitness(x);
            break;
        case 9u:
            retval = wfg9_fitness(x);
            break;
    }
    return retval;
}

// Number of objectives
vector_double::size_type wfg::get_nobj() const
{
    return m_dim_obj;
}
// Box-bounds
std::pair<vector_double, vector_double> wfg::get_bounds() const
{
    vector_double upper_bounds(m_dim_dvs);
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        upper_bounds[i] = 2.0 * (static_cast<double>(i) + 1);
    }

    return {vector_double(m_dim_dvs, 0.), upper_bounds};
}

// Problem name
std::string wfg::get_name() const
{
    return "WFG" + std::to_string(m_prob_id);
}

// Object serialization
template <typename Archive>
void wfg::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_prob_id, m_dim_dvs, m_dim_obj, m_dim_k);
}

// We first define the shape functions (we assume that m varies from 1 to m_dim_obj):
double wfg::linear(const vector_double &parameters, const vector_double::size_type m) const
{
    double g = 1.;
    if (m == 1u) {
        for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - 1; ++i) {
            g *= parameters[i];
        }
        return g;
    } else if (m > 1u && m < m_dim_obj) {
        for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - m; ++i) {
            g *= parameters[i];
        }
        return g * (1.0 - parameters[m_dim_obj - m]);
    } else {
        return 1.0 - parameters[0];
    }
}

double wfg::convex(const vector_double &parameters, const vector_double::size_type m) const
{
    double g = 1.;
    if (m == 1) {
        for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - 1; ++i) {
            g *= 1.0 - std::cos(parameters[i] * boost::math::constants::pi<double>() / 2.0);
        }
        return g;
    } else {
        for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - m; ++i) {
            g *= 1.0 - std::cos(parameters[i] * boost::math::constants::pi<double>() / 2.0);
        }
        return g * (1 - std::sin(parameters[m_dim_obj - m] * boost::math::constants::pi<double>() / 2.0));
    }
}

double wfg::concave(const vector_double &parameters, const vector_double::size_type m) const
{
    double g = 1.;
    if (m == 1) {
        for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - 1; ++i) {
            g *= std::sin(parameters[i] * boost::math::constants::pi<double>() / 2.0);
        }
        return g;
    } else if (m > 1 && m < m_dim_obj) {
        for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - m; ++i) {
            g *= std::sin(parameters[i] * boost::math::constants::pi<double>() / 2.0);
        }
        return g * std::cos(parameters[m_dim_obj - m] * boost::math::constants::pi<double>() / 2.0);
    } else {
        return std::cos(parameters[0] * boost::math::constants::pi<double>() / 2.0);
    }
}

double wfg::mixed(const double parameters_0, const double alpha, const double deg_const) const
{
    return std::pow((1.0 - parameters_0
                     - std::cos(2 * deg_const * boost::math::constants::pi<double>() * parameters_0
                                + boost::math::constants::pi<double>() / 2.0)
                           / (2.0 * deg_const * boost::math::constants::pi<double>())),
                    alpha);
}

double wfg::disconnected(const double parameters_0, const double alpha, const double beta, const double deg_const) const
{
    return 1.0
           - std::pow(parameters_0, alpha)
                 * std::pow(std::cos(deg_const * std::pow(parameters_0, beta) * boost::math::constants::pi<double>()),
                            2);
}

// We now define the transformation functions:
double wfg::b_poly(const double y, const double alpha) const
{
    return std::pow(y, alpha);
}

double wfg::b_flat(const double y, const double a_par, const double b_par, const double c_par) const
{
    return a_par
           + std::min(0.0, std::floor(y - b_par)) * a_par * (b_par - y) / (b_par)-std::min(0.0, std::floor(c_par - y))
                 * (1.0 - a_par) * (y - c_par) / (1 - c_par);
}

double wfg::b_param(const double y, const double u, const double a_par, const double b_par, const double c_par) const
{
    double v = a_par - (1.0 - 2 * u) * std::abs(std::floor(0.5 - u) + a_par);
    return std::pow(y, b_par + (c_par - b_par) * v);
}

double wfg::s_linear(const double y, const double a_par) const
{
    return std::abs(y - a_par) / (std::abs(std::floor(a_par - y) + a_par));
}

double wfg::s_decept(const double y, const double a_par, const double b_par, const double c_par) const
{
    return 1.0
           + (std::abs(y - a_par) - b_par)
                 * ((std::floor(y - a_par + b_par) * (1.0 - c_par + (a_par - b_par) / b_par)) / (a_par - b_par)
                    + (std::floor(a_par + b_par - y) * (1.0 - c_par + (1.0 - a_par - b_par) / b_par))
                          / (1.0 - a_par - b_par)
                    + 1.0 / b_par);
}

double wfg::s_multi(const double y, const double a_par, const double b_par, const double c_par) const
{
    return (1
            + std::cos((4.0 * a_par + 2.0) * boost::math::constants::pi<double>()
                       * (0.5 - (std::abs(y - c_par)) / (2.0 * (std::floor(c_par - y) + c_par))))
            + 4.0 * b_par * std::pow(std::abs(y - c_par) / (2 * (std::floor(c_par - y) + c_par)), 2))
           / (b_par + 2.0);
}

double wfg::r_sum(const vector_double &y_vec, const vector_double &weights) const
{
    double g_1 = 0.;
    double g_2 = 0.;
    for (decltype(y_vec.size()) i = 0u; i < y_vec.size(); ++i) {
        g_1 += weights[i] * y_vec[i];
        g_2 += weights[i];
    }
    return g_1 / g_2;
}

double wfg::r_nonsep(const vector_double &y_vec, const vector_double::size_type a_par) const
{
    if (a_par == 1) {
        vector_double weights(y_vec.size(), 1.0);
        return r_sum(y_vec, weights);
    } else {
        double g = 0.;
        auto len = y_vec.size();
        for (decltype(len) j = 0u; j < len; ++j) {
            g += y_vec[j];
            for (decltype(len) i = 0u; i <= a_par - 2; ++i) {
                g += std::abs(y_vec[j] - y_vec[(1 + j + i) % len]);
            }
        }
        return g
               / (static_cast<double>(len) / static_cast<double>(a_par) * std::ceil(a_par / 2)
                  * (1.0 + 2.0 * static_cast<double>(a_par) - 2.0 * std::ceil(static_cast<double>(a_par) / 2.0)));
    }
}

vector_double wfg::wfg1_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_dvs);
    vector_double t_3(m_dim_dvs);
    vector_double t_4(m_dim_obj);
    vector_double first_input_2(m_dim_dvs - m_dim_k);
    vector_double second_input_2(m_dim_dvs - m_dim_k);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }

    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_1[i] = x_norm[i];
        } else {
            t_1[i] = s_linear(x_norm[i], 0.35);
        }
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_2[i] = y[i];
        } else {
            t_2[i] = b_flat(y[i], 0.8, 0.75, 0.85);
        }
        y[i] = t_2[i];
    }
    // t_3:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        t_3[i] = b_poly(y[i], 0.02);
        y[i] = t_3[i];
    }
    // t_4:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_obj) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) tail = i * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) index = 0u;
        vector_double first_input(tail - head);
        vector_double second_input(tail - head);

        for (decltype(head) j = head; j < tail; ++j) {
            first_input[index] = y[j];
            second_input[index] = 2. * (static_cast<double>(j) + 1);
            ++index;
        }

        t_4[i - 1] = r_sum(first_input, second_input);
    }

    decltype(m_dim_obj) index_2 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        first_input_2[index_2] = y[i];
        second_input_2[index_2] = 2. * (static_cast<double>(i) + 1);
        ++index_2;
    }
    t_4[m_dim_obj - 1] = r_sum(first_input_2, second_input_2);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_4[m_dim_obj - 1], 1.0) * (t_4[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_4[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - 1; ++i) {
        shape_fun[i] = convex(parameters, i + 1);
    }
    shape_fun[m_dim_obj - 1] = mixed(parameters[0], 1.0, 5.0);

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg2_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    auto l = m_dim_dvs - m_dim_k;
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_k + l / 2);
    vector_double t_3(m_dim_obj);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }
    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_1[i] = x_norm[i];
        } else {
            t_1[i] = s_linear(x_norm[i], 0.35);
        }
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_k) i = 1u; i <= m_dim_k + l / 2; ++i) {
        if (i <= m_dim_k) {
            t_2[i - 1] = y[i - 1];
        } else {
            decltype(m_dim_k) head = m_dim_k + 2 * (i - m_dim_k) - 2;
            decltype(m_dim_k) tail = m_dim_k + 2 * (i - m_dim_k);
            decltype(m_dim_obj) index = 0u;
            vector_double first_input(tail - head);
            for (decltype(head) j = head; j < tail; ++j) {
                first_input[index] = y[j];
                ++index;
            }
            t_2[i - 1] = r_nonsep(first_input, 2);
        }
        y[i - 1] = t_2[i - 1];
    }
    // t_3:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_k) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_k) tail = i * m_dim_k / (m_dim_obj - 1);
        vector_double first_input_2(tail - head);
        vector_double weights(tail - head, 1.0);
        decltype(m_dim_obj) index_2 = 0u;
        for (decltype(head) j = head; j < tail; ++j) {
            first_input_2[index_2] = y[j];
            ++index_2;
        }
        t_3[i - 1] = r_sum(first_input_2, weights);
    }
    vector_double first_input_3(l / 2);
    vector_double weights_2(l / 2, 1.0);
    decltype(m_dim_obj) index_3 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_k + l / 2; ++i) {
        first_input_3[index_3] = y[i];
        ++index_3;
    }
    t_3[m_dim_obj - 1] = r_sum(first_input_3, weights_2);
    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_3[m_dim_obj - 1], 1.0) * (t_3[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_3[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj - 1; ++i) {
        shape_fun[i] = convex(parameters, i + 1);
    }
    shape_fun[m_dim_obj - 1] = disconnected(parameters[0], 1.0, 1.0, 5.0);

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg3_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    auto l = m_dim_dvs - m_dim_k;
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_k + l / 2);
    vector_double t_3(m_dim_obj);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }
    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_1[i] = x_norm[i];
        } else {
            t_1[i] = s_linear(x_norm[i], 0.35);
        }
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_k) i = 1u; i <= m_dim_k + l / 2; ++i) {
        if (i <= m_dim_k) {
            t_2[i - 1] = y[i - 1];
        } else {
            decltype(m_dim_k) head = m_dim_k + 2 * (i - m_dim_k) - 2;
            decltype(m_dim_k) tail = m_dim_k + 2 * (i - m_dim_k);
            decltype(m_dim_obj) index = 0u;
            vector_double first_input(tail - head);
            for (decltype(head) j = head; j < tail; ++j) {
                first_input[index] = y[j];
                ++index;
            }
            t_2[i - 1] = r_nonsep(first_input, 2);
        }
        y[i - 1] = t_2[i - 1];
    }
    // t_3:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_k) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_k) tail = i * m_dim_k / (m_dim_obj - 1);
        vector_double first_input_2(tail - head);
        vector_double weights(tail - head, 1.0);
        decltype(m_dim_obj) index_2 = 0u;
        for (decltype(head) j = head; j < tail; ++j) {
            first_input_2[index_2] = y[j];
            ++index_2;
        }
        t_3[i - 1] = r_sum(first_input_2, weights);
    }
    vector_double first_input_3(l / 2);
    vector_double weights_2(l / 2, 1.0);
    decltype(m_dim_obj) index_3 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_k + l / 2; ++i) {
        first_input_3[index_3] = y[i];
        ++index_3;
    }
    t_3[m_dim_obj - 1] = r_sum(first_input_3, weights_2);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_3[m_dim_obj - 1], 1.0) * (t_3[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_3[m_dim_obj - 1];

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        if (i == 0u) {
            parameters[i] = std::max(t_3[m_dim_obj - 1], 1.0) * (t_3[i] - 0.5) + 0.5;
        } else {
            parameters[i] = std::max(t_3[m_dim_obj - 1], 0.0) * (t_3[i] - 0.5) + 0.5;
        }
    }
    parameters[m_dim_obj - 1] = t_3[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        shape_fun[i] = linear(parameters, i + 1);
    }
    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg4_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_dvs);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }

    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        t_1[i] = s_multi(x_norm[i], 30.0, 10.0, 0.35);
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_k) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_k) tail = i * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_k) index = 0u;
        vector_double first_input(tail - head);
        vector_double weights(tail - head, 1.0);

        for (decltype(head) j = head; j < tail; ++j) {
            first_input[index] = y[j];
            ++index;
        }
        t_2[i - 1] = r_sum(first_input, weights);
    }
    vector_double first_input_2(m_dim_dvs - m_dim_k);
    vector_double weights_2(first_input_2.size(), 1.0);
    decltype(m_dim_obj) index_2 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        first_input_2[index_2] = y[i];
        ++index_2;
    }
    t_2[m_dim_obj - 1] = r_sum(first_input_2, weights_2);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_2[m_dim_obj - 1], 1.0) * (t_2[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_2[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        shape_fun[i] = concave(parameters, i + 1);
    }

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg5_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_dvs);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }

    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        t_1[i] = s_decept(x_norm[i], 0.35, 0.001, 0.05);
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_k) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_k) tail = i * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) index = 0u;
        vector_double first_input(tail - head);
        vector_double weights(tail - head, 1.0);

        for (decltype(head) j = head; j < tail; ++j) {
            first_input[index] = y[j];
            ++index;
        }
        t_2[i - 1] = r_sum(first_input, weights);
    }
    vector_double first_input_2(m_dim_dvs - m_dim_k);
    vector_double weights_2(first_input_2.size(), 1.0);
    decltype(m_dim_obj) index_2 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        first_input_2[index_2] = y[i];
        ++index_2;
    }
    t_2[m_dim_obj - 1] = r_sum(first_input_2, weights_2);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_2[m_dim_obj - 1], 1.0) * (t_2[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_2[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        shape_fun[i] = concave(parameters, i + 1);
    }

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg6_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    auto l = m_dim_dvs - m_dim_k;
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_obj);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }
    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_1[i] = x_norm[i];
        } else {
            t_1[i] = s_linear(x_norm[i], 0.35);
        }
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_k) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_k) tail = i * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_k) index = 0u;
        vector_double first_input(tail - head);

        for (decltype(head) j = head; j < tail; ++j) {
            first_input[index] = y[j];
            ++index;
        }
        t_2[i - 1] = r_nonsep(first_input, m_dim_k / (m_dim_obj - 1));
    }

    vector_double first_input_2(m_dim_dvs - m_dim_k);
    decltype(m_dim_obj) index_2 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        first_input_2[index_2] = y[i];
        ++index_2;
    }
    t_2[m_dim_obj - 1] = r_nonsep(first_input_2, l);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_2[m_dim_obj - 1], 1.0) * (t_2[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_2[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        shape_fun[i] = concave(parameters, i + 1);
    }

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg7_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_dvs);
    vector_double t_3(m_dim_obj);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }

    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_k) i = 1u; i <= m_dim_k; ++i) {
        vector_double first_input(m_dim_dvs - i);
        vector_double weights(m_dim_dvs - i, 1.0);
        decltype(m_dim_obj) index = 0u;
        for (decltype(m_dim_dvs) j = i; j < m_dim_dvs; ++j) {
            first_input[index] = x_norm[j];
            ++index;
        }
        t_1[i - 1] = b_param(x_norm[i - 1], r_sum(first_input, weights), 0.98 / 49.98, 0.02, 50);
        y[i - 1] = t_1[i - 1];
    }
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        t_1[i] = x_norm[i];
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_2[i] = y[i];
        } else {
            t_2[i] = s_linear(y[i], 0.35);
        }
        y[i] = t_2[i];
    }

    // t_3:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_obj) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) tail = i * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) index_2 = 0u;
        vector_double first_input_2(tail - head);
        vector_double weights_2(tail - head, 1.0);

        for (decltype(head) j = head; j < tail; ++j) {
            first_input_2[index_2] = y[j];
            ++index_2;
        }
        t_3[i - 1] = r_sum(first_input_2, weights_2);
    }
    vector_double first_input_3(m_dim_dvs - m_dim_k);
    vector_double weights_3(first_input_3.size(), 1.0);
    decltype(m_dim_obj) index_3 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        first_input_3[index_3] = y[i];
        ++index_3;
    }
    t_3[m_dim_obj - 1] = r_sum(first_input_3, weights_3);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_3[m_dim_obj - 1], 1.0) * (t_3[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_3[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        shape_fun[i] = concave(parameters, i + 1);
    }

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg8_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    vector_double y(m_dim_dvs);
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_dvs);
    vector_double t_3(m_dim_obj);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }

    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_k) i = 0u; i < m_dim_k; ++i) {
        t_1[i] = x_norm[i];
        y[i] = t_1[i];
    }
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        vector_double first_input(i);
        vector_double weights(i, 1.0);
        decltype(m_dim_obj) index = 0u;
        for (decltype(i) j = 0u; j < i; ++j) {
            first_input[j] = y[j];
            ++index;
        }
        t_1[i] = b_param(x_norm[i], r_sum(first_input, weights), 0.98 / 49.98, 0.02, 50);
        y[i] = t_1[i];
    }
    // t_2:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_2[i] = y[i];
        } else {
            t_2[i] = s_linear(y[i], 0.35);
        }
        y[i] = t_2[i];
    }

    // t_3:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_obj) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) tail = i * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) index_2 = 0u;
        vector_double first_input_2(tail - head);
        vector_double weights_2(tail - head, 1.0);

        for (decltype(head) j = head; j < tail; ++j) {
            first_input_2[index_2] = y[j];
            ++index_2;
        }
        t_3[i - 1] = r_sum(first_input_2, weights_2);
    }
    vector_double first_input_3(m_dim_dvs - m_dim_k);
    vector_double weights_3(first_input_3.size(), 1.0);
    decltype(m_dim_obj) index_3 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        first_input_3[index_3] = y[i];
        ++index_3;
    }
    t_3[m_dim_obj - 1] = r_sum(first_input_3, weights_3);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_3[m_dim_obj - 1], 1.0) * (t_3[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_3[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        shape_fun[i] = concave(parameters, i + 1);
    }

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

vector_double wfg::wfg9_fitness(const vector_double &x) const
{
    // I declare some useful variables:
    vector_double s_parameter(m_dim_obj);
    vector_double x_norm(m_dim_dvs);
    vector_double parameters(m_dim_obj);
    vector_double shape_fun(m_dim_obj);
    auto l = m_dim_dvs - m_dim_k;
    vector_double y(m_dim_dvs);
    vector_double t_1(m_dim_dvs);
    vector_double t_2(m_dim_dvs);
    vector_double t_3(m_dim_obj);

    // s_parameter (useful for the shape function computation):
    for (decltype(s_parameter.size()) i = 0u; i < s_parameter.size(); ++i) {
        s_parameter[i] = 2.0 * (static_cast<double>(i) + 1);
    }

    // I normalize the decision vector:
    for (decltype(x_norm.size()) i = 0u; i < x_norm.size(); ++i) {
        x_norm[i] = x[i] / get_bounds().second[i];
    }

    // We make use of the fact that y=x_norm for the t_1 vector, for the others y=t_{i-1}
    // t_1:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs - 1; ++i) {
        vector_double first_input((m_dim_dvs - 1) - i);
        vector_double weights((m_dim_dvs - 1) - i, 1.0);
        decltype(m_dim_obj) index = 0u;
        for (decltype(i) j = i + 1; j < m_dim_dvs; ++j) {
            first_input[index] = x_norm[j];
            ++index;
        }
        t_1[i] = b_param(x_norm[i], r_sum(first_input, weights), 0.98 / 49.98, 0.02, 50);
        y[i] = t_1[i];
    }
    t_1[m_dim_dvs - 1] = x_norm[m_dim_dvs - 1];
    y[m_dim_dvs - 1] = t_1[m_dim_dvs - 1];
    // t_2:
    for (decltype(m_dim_dvs) i = 0u; i < m_dim_dvs; ++i) {
        if (i < m_dim_k) {
            t_2[i] = s_decept(y[i], 0.35, 0.001, 0.05);
            y[i] = t_2[i];
        } else {
            t_2[i] = s_multi(y[i], 30.0, 95.0, 0.35);
            y[i] = t_2[i];
        }
    }
    // t_3:
    for (decltype(m_dim_obj) i = 1u; i <= m_dim_obj - 1; ++i) {
        decltype(m_dim_obj) head = (i - 1) * m_dim_k / (m_dim_obj - 1);
        decltype(m_dim_obj) tail = i * m_dim_k / (m_dim_obj - 1);
        vector_double first_input_2(tail - head);
        decltype(m_dim_obj) index_2 = 0u;
        for (decltype(head) j = head; j < tail; ++j) {
            first_input_2[index_2] = y[j];
            ++index_2;
        }
        t_3[i - 1] = r_nonsep(first_input_2, m_dim_k / (m_dim_obj - 1));
    }
    vector_double first_input_3(m_dim_dvs - m_dim_k);
    decltype(m_dim_obj) index_3 = 0u;
    for (decltype(m_dim_k) i = m_dim_k; i < m_dim_dvs; ++i) {
        first_input_3[index_3] = y[i];
        ++index_3;
    }
    t_3[m_dim_obj - 1] = r_nonsep(first_input_3, l);

    // parameters:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        parameters[i] = std::max(t_3[m_dim_obj - 1], 1.0) * (t_3[i] - 0.5) + 0.5;
    }
    parameters[m_dim_obj - 1] = t_3[m_dim_obj - 1];

    // shape functions:
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        shape_fun[i] = concave(parameters, i + 1);
    }

    // fitness computation:
    vector_double f(m_dim_obj);
    for (decltype(m_dim_obj) i = 0u; i < m_dim_obj; ++i) {
        f[i] = parameters[m_dim_obj - 1] + s_parameter[i] * shape_fun[i];
    }
    return f;
}

} // namespace pagmo

PAGMO_S11N_PROBLEM_IMPLEMENT(pagmo::wfg)
