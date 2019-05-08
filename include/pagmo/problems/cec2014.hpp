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

/*
  CEC14 Test Function Suite for Single Objective Optimization
  Jane Jing Liang (email: liangjing@zzu.edu.cn; liangjing@pmail.ntu.edu.cn)
*/

#ifndef PAGMO_PROBLEMS_CEC2014_HPP
#define PAGMO_PROBLEMS_CEC2014_HPP

#include <pagmo/config.hpp>

#if !defined(PAGMO_ENABLE_CEC2014)

#error The cec2014.hpp header was included but the CEC 2014 problem is not supported on the current platform.

#endif

#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{
/// The CEC 2014 problems: Real-Parameter Single Objective Optimization Competition
/**
 * \verbatim embed:rst:leading-asterisk
 *
 * .. versionadded:: 2.8
 *
 * The 30 problems of the competition on real-parameter single objective optimization problems that
 * was organized for the 2014 IEEE Congress on Evolutionary Computation.
 *
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
 *    http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC2014/CEC2014.htm
 *    http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC-2014/cec14-c-code.zip
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC cec2014
{

public:
    /// Constructor
    /**
     * Will construct one of the 30 CEC2014 problems
     *
     * @param prob_id The problem id. One of [1,2,...,30]
     * @param dim problem dimension. One of [2,10,20,30,50,100]
     *
     * @throws invalid_argument if \p prob_id is not in [1,30] or if \p dim is not one of
     * [2,10,20,30,50,100]
     */
    cec2014(unsigned prob_id = 1u, unsigned dim = 2u);

    // Box-bounds
    std::pair<vector_double, vector_double> get_bounds() const;

    // Fitness computation
    vector_double fitness(const vector_double &) const;

    // Problem name
    std::string get_name() const;

    /// Returns the origin shift
    /**
     * This method will return the origin shift.
     *
     * @return The origin shift.
     */
    const vector_double &get_origin_shift() const
    {
        return m_origin_shift;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    /* Sphere */
    PAGMO_DLL_LOCAL void sphere_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                                     int s_flag, int r_flag) const;
    /* Ellipsoidal */
    PAGMO_DLL_LOCAL void ellips_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                                     int s_flag, int r_flag) const;
    /* Bent_Cigar */
    PAGMO_DLL_LOCAL void bent_cigar_func(const double *x, double *f, const unsigned nx, const double *Os,
                                         const double *Mr, int s_flag, int r_flag) const;
    /* Discus */
    PAGMO_DLL_LOCAL void discus_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                                     int s_flag, int r_flag) const;
    /* Different Powers */
    PAGMO_DLL_LOCAL void dif_powers_func(const double *x, double *f, const unsigned nx, const double *Os,
                                         const double *Mr, int s_flag, int r_flag) const;
    /* Rosenbrock's */
    PAGMO_DLL_LOCAL void rosenbrock_func(const double *x, double *f, const unsigned nx, const double *Os,
                                         const double *Mr, int s_flag, int r_flag) const;
    /* Schwefel's 1.2  */
    PAGMO_DLL_LOCAL void schaffer_F7_func(const double *x, double *f, const unsigned nx, const double *Os,
                                          const double *Mr, int s_flag, int r_flag) const;
    /* Ackley's  */
    PAGMO_DLL_LOCAL void ackley_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                                     int s_flag, int r_flag) const;
    /* Weierstrass's  */
    PAGMO_DLL_LOCAL void weierstrass_func(const double *x, double *f, const unsigned nx, const double *Os,
                                          const double *Mr, int s_flag, int r_flag) const;
    /* Griewank's  */
    PAGMO_DLL_LOCAL void griewank_func(const double *x, double *f, const unsigned nx, const double *Os,
                                       const double *Mr, int s_flag, int r_flag) const;
    /* Rastrigin's  */
    PAGMO_DLL_LOCAL void rastrigin_func(const double *x, double *f, const unsigned nx, const double *Os,
                                        const double *Mr, int s_flag, int r_flag) const;
    /* Noncontinuous Rastrigin's  */
    PAGMO_DLL_LOCAL void step_rastrigin_func(const double *x, double *f, const unsigned nx, const double *Os,
                                             const double *Mr, int s_flag, int r_flag) const;
    /* Schwefel's  */
    PAGMO_DLL_LOCAL void schwefel_func(const double *x, double *f, const unsigned nx, const double *Os,
                                       const double *Mr, int s_flag, int r_flag) const;
    /* Katsuura  */
    PAGMO_DLL_LOCAL void katsuura_func(const double *x, double *f, const unsigned nx, const double *Os,
                                       const double *Mr, int s_flag, int r_flag) const;
    /* Lunacek Bi_rastrigin Function */
    PAGMO_DLL_LOCAL void bi_rastrigin_func(const double *x, double *f, const unsigned nx, const double *Os,
                                           const double *Mr, int s_flag, int r_flag) const;
    /* Griewank-Rosenbrock  */
    PAGMO_DLL_LOCAL void grie_rosen_func(const double *x, double *f, const unsigned nx, const double *Os,
                                         const double *Mr, int s_flag, int r_flag) const;
    /* Expanded Scaffer??s F6  */
    PAGMO_DLL_LOCAL void escaffer6_func(const double *x, double *f, const unsigned nx, const double *Os,
                                        const double *Mr, int s_flag, int r_flag) const;
    /* HappyCat, provdided by Hans-Georg Beyer (HGB) */
    /* original global optimum: [-1,-1,...,-1] */
    PAGMO_DLL_LOCAL void happycat_func(const double *x, double *f, const unsigned nx, const double *Os,
                                       const double *Mr, int s_flag, int r_flag) const;
    /* HGBat, provdided by Hans-Georg Beyer (HGB)*/
    /* original global optimum: [-1,-1,...,-1] */
    PAGMO_DLL_LOCAL void hgbat_func(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                                    int s_flag, int r_flag) const;
    /* Hybrid Function 1 */
    PAGMO_DLL_LOCAL void hf01(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *S, int s_flag, int r_flag) const;
    /* Hybrid Function 2 */
    PAGMO_DLL_LOCAL void hf02(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *S, int s_flag, int r_flag) const;
    /* Hybrid Function 3 */
    PAGMO_DLL_LOCAL void hf03(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *S, int s_flag, int r_flag) const;
    /* Hybrid Function 4 */
    PAGMO_DLL_LOCAL void hf04(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *S, int s_flag, int r_flag) const;
    /* Hybrid Function 5 */
    PAGMO_DLL_LOCAL void hf05(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *S, int s_flag, int r_flag) const;
    /* Hybrid Function 6 */
    PAGMO_DLL_LOCAL void hf06(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *S, int s_flag, int r_flag) const;
    /* Composition Function 1 */
    PAGMO_DLL_LOCAL void cf01(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int r_flag) const;
    /* Composition Function 2 */
    PAGMO_DLL_LOCAL void cf02(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int r_flag) const;
    /* Composition Function 3 */
    PAGMO_DLL_LOCAL void cf03(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int r_flag) const;
    /* Composition Function 4 */
    PAGMO_DLL_LOCAL void cf04(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int r_flag) const;
    /* Composition Function 4 */
    PAGMO_DLL_LOCAL void cf05(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int r_flag) const;
    /* Composition Function 6 */
    PAGMO_DLL_LOCAL void cf06(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              int r_flag) const;
    /* Composition Function 7 */
    PAGMO_DLL_LOCAL void cf07(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *SS, int r_flag) const;
    /* Composition Function 8 */
    PAGMO_DLL_LOCAL void cf08(const double *x, double *f, const unsigned nx, const double *Os, const double *Mr,
                              const int *SS, int r_flag) const;
    PAGMO_DLL_LOCAL void shiftfunc(const double *x, double *xshift, const unsigned nx, const double *Os) const;
    PAGMO_DLL_LOCAL void rotatefunc(const double *x, double *xrot, const unsigned nx, const double *Mr) const;
    /* shift and rotate */
    PAGMO_DLL_LOCAL void sr_func(const double *x, double *sr_x, const unsigned nx, const double *Os, const double *Mr,
                                 double sh_rate, int s_flag, int r_flag) const;
    PAGMO_DLL_LOCAL void asyfunc(const double *x, double *xasy, const unsigned nx, double beta) const;
    PAGMO_DLL_LOCAL void oszfunc(const double *x, double *xosz, const unsigned nx) const;
    PAGMO_DLL_LOCAL void cf_cal(const double *x, double *f, const unsigned nx, const double *Os, double *delta,
                                double *bias, double *fit, int cf_num) const;

    // problem data
    vector_double m_origin_shift;
    vector_double m_rotation_matrix;
    std::vector<int> m_shuffle;

    // auxiliary vectors
    mutable vector_double m_z;
    mutable vector_double m_y;

    // problem id
    unsigned func_num;
};

} // namespace pagmo

PAGMO_S11N_PROBLEM_EXPORT_KEY(pagmo::cec2014)

#endif
