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

#ifndef PAGMO_UTILS_GRADIENTS_AND_HESSIANS_HPP
#define PAGMO_UTILS_GRADIENTS_AND_HESSIANS_HPP

/** \file gradients_and_hessians.hpp
 * \brief Utilities of general interest for gradients and hessians related calculations
 *
 * This header contains utilities useful in general for gradients and hessians related calculations
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <pagmo/exceptions.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{
/// Heuristic to estimate the sparsity pattern
/**
 * A numerical estimation of the sparsity pattern of same callable object is made by numerically
 * computing it around a given input point and detecting the components that are changed.
 *
 * The callable function \p f must have the prototype:
 *
 * @code{.unparsed}
 * vector_double f(const vector_double &)
 * @endcode
 *
 * otherwise compiler errors will be generated.
 *
 * Note that estimate_sparsity may fail to detect the real sparsity as it only considers one variation around the input
 * point. It is of use, though, in tests or cases where its not possible to write the sparsity or where the user is
 * confident the estimate will be correct.
 *
 * @param f instance of the callable object.
 * @param x decision vector to test the sparisty around.
 * @param dx To detect the sparsity each component of the input decision vector \p x will be changed by \f$\max(|x_i|,
 * 1) * \f$ \p dx.
 * @return the sparsity_pattern of \p f as detected around \p x.
 *
 * @throw std::invalid_argument if \p f returns fitness vectors of different sizes when perturbing \p x.
 */
template <typename Func>
sparsity_pattern estimate_sparsity(Func f, const vector_double &x, double dx = 1e-8)
{
    vector_double f0 = f(x);
    vector_double x_new = x;
    sparsity_pattern retval;
    // We change one by one each variable by dx and detect changes in the f call
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        x_new[j] = x[j] + std::max(std::abs(x[j]), 1.0) * dx;
        auto f_new = f(x_new);
        if (f_new.size() != f0.size()) {
            pagmo_throw(
                std::invalid_argument,
                "Change in the size of the returned vector around the reference point. Cannot estimate a sparisty.");
        }
        for (decltype(f_new.size()) i = 0u; i < f_new.size(); ++i) {
            if (f_new[i] != f0[i]) {
                retval.push_back({i, j});
            }
        }
        x_new[j] = x[j];
    }
    // Restore the lexicographic order required by pagmo::problem::gradient_sparsity
    std::sort(retval.begin(), retval.end());
    return retval;
}

/// Numerical computation of the gradient (low-order)
/**
 * A numerical estimation of the gradient of same callable function is made numerically.
 *
 * The callable function \p f must have the prototype:
 *
 * @code{.unparsed}
 * vector_double f(const vector_double &)
 * @endcode
 *
 * otherwise compiler errors will be generated. The gradient returned will be dense and contain, in the lexicographic
 * order requested by pagmo::problem::gradient(), \f$\frac{df_i}{dx_j}\f$.
 *
 * The numerical approximation of each derivative is made by central difference, according to the formula:
 *
 * \f[
 * \frac{df}{dx} \approx \frac{f(x+dx) - f(x-dx)}{2dx} + O(dx^2)
 * \f]
 *
 * The overall cost, in terms of calls to \p f will thus be \f$2n\f$ where \f$n\f$ is the size of \p x.
 *
 * @param f instance of the callable object.
 * @param x decision vector to test the sparisty around.
 * @param dx To detect the numerical derivative each component of the input decision vector \p x will be varied by
 * \f$\max(|x_i|,1) * \f$ \p dx.
 * @return the gradient of \p f approximated around \p x in the format required by pagmo::problem::gradient().
 *
 * @throw std::invalid_argument if \p f returns vectors of different sizes when perturbing \p x.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The gradient returned is assumed as dense: elements equal to zero are not excluded.
 *
 * \endverbatim
 */
template <typename Func>
vector_double estimate_gradient(Func f, const vector_double &x, double dx = 1e-8)
{
    vector_double f0 = f(x);
    vector_double gradient(f0.size() * x.size(), 0.);
    vector_double x_r = x, x_l = x;
    // We change one by one each variable by dx and estimate the derivative
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        double h = std::max(std::abs(x[j]), 1.0) * dx;
        x_r[j] = x[j] + h;
        x_l[j] = x[j] - h;
        vector_double f_r = f(x_r);
        vector_double f_l = f(x_l);
        if (f_r.size() != f0.size() || f_l.size() != f0.size()) {
            pagmo_throw(std::invalid_argument, "Change in the size of the returned vector detected around the "
                                               "reference point. Cannot compute a gradient");
        }
        for (decltype(f_r.size()) i = 0u; i < f_r.size(); ++i) {
            gradient[j + i * x.size()] = (f_r[i] - f_l[i]) / 2. / h;
        }
        x_r[j] = x[j];
        x_l[j] = x[j];
    }
    return gradient;
}

/// Numerical computation of the gradient (high-order)
/**
 * A numerical estimation of the gradient of same callable function is made numerically.
 *
 * The callable function \p f must have the prototype:
 *
 * @code{.unparsed}
 * vector_double f(const vector_double &)
 * @endcode
 *
 * otherwise compiler errors will be generated. The gradient returned will be dense and contain, in the lexicographic
 * order requested by pagmo::problem::gradient(), \f$\frac{df_i}{dx_j}\f$.
 *
 * The numerical approximation of each derivative is made by central difference, according to the formula:
 *
 * \f[
 * \frac{df}{dx} \approx \frac 32 m_1 - \frac 35 m_2 +\frac 1{10} m_3 + O(dx^6)
 * \f]
 *
 * where:
 *
 * \f[
 * m_i = \frac{f(x + i dx) - f(x-i dx)}{2i dx}
 * \f]
 *
 * The overall cost, in terms of calls to \p f will thus be \f$6n\f$ where \f$n\f$ is the size of \p x.
 *
 * @param f instance of the callable object.
 * @param x decision vector to test the sparisty around.
 * @param dx To detect the numerical derivative each component of the input decision vector \p x will be varied by
 * \f$\max(|x_i|,1) * \f$ \p dx.
 * @return the gradient of \p f approximated around \p x in the format required by pagmo::problem::gradient().
 *
 * @throw std::invalid_argument if \p f returns vectors of different sizes when perturbing \p x.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The gradient returned is assumed as dense: elements equal to zero are not excluded.
 *
 * \endverbatim
 */
template <typename Func>
vector_double estimate_gradient_h(Func f, const vector_double &x, double dx = 1e-2)
{
    vector_double f0 = f(x);
    vector_double gradient(f0.size() * x.size(), 0.);
    vector_double x_r1 = x, x_l1 = x;
    vector_double x_r2 = x, x_l2 = x;
    vector_double x_r3 = x, x_l3 = x;
    // We change one by one each variable by dx and estimate the derivative
    for (decltype(x.size()) j = 0u; j < x.size(); ++j) {
        double h = std::max(std::abs(x[j]), 1.0) * dx;
        x_r1[j] = x[j] + h;
        x_l1[j] = x[j] - h;
        x_r2[j] = x[j] + 2. * h;
        x_l2[j] = x[j] - 2. * h;
        x_r3[j] = x[j] + 3. * h;
        x_l3[j] = x[j] - 3. * h;
        vector_double f_r1 = f(x_r1);
        vector_double f_l1 = f(x_l1);
        vector_double f_r2 = f(x_r2);
        vector_double f_l2 = f(x_l2);
        vector_double f_r3 = f(x_r3);
        vector_double f_l3 = f(x_l3);
        if (f_r1.size() != f0.size() || f_l1.size() != f0.size() || f_r2.size() != f0.size() || f_l2.size() != f0.size()
            || f_r3.size() != f0.size() || f_l3.size() != f0.size()) {
            pagmo_throw(std::invalid_argument, "Change in the size of the returned vector detected around the "
                                               "reference point. Cannot compute a gradient");
        }
        for (decltype(f_r1.size()) i = 0u; i < f_r1.size(); ++i) {
            double m1 = (f_r1[i] - f_l1[i]) / 2.;
            double m2 = (f_r2[i] - f_l2[i]) / 4.;
            double m3 = (f_r3[i] - f_l3[i]) / 6.;
            double fifteen_m1 = 15. * m1;
            double six_m2 = 6. * m2;
            double ten_h = 10. * h;
            gradient[j + i * x.size()] = ((fifteen_m1 - six_m2) + m3) / ten_h;
        }
        x_r1[j] = x[j];
        x_l1[j] = x[j];
        x_r2[j] = x[j];
        x_l2[j] = x[j];
        x_r3[j] = x[j];
        x_l3[j] = x[j];
    }
    return gradient;
}
} // namespace pagmo
// namespace pagmo

#endif
