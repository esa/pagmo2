#ifndef PAGMO_PROBLEM_HOCK_SCHITTKOWSKY_71
#define PAGMO_PROBLEM_HOCK_SCHITTKOWSKY_71

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{

/// Problem No.71 from the Hock Schittkowsky suite
/**
 * Mainly used for testing and debugging during PaGMO development, this
 * struct implements the problem No.71 from the Hock Schittkowsky suite:
 *
 * \f[
 *    \begin{array}{rl}
 *    \mbox{find: } & 1 \le \mathbf x \le 5 \\
 *    \mbox{to minimize: } & x_1x_4(x_1+x_2+x_3) + x_3 \\
 *    \mbox{subject to: } & x_1^2+x_2^2+x_3^2+x_4^2 - 40 = 0 \\
 *                        & 25 - x_1 x_2 x_3 x_4 \le 0
 *    \end{array}
 * \f]
 * 
 * @see W. Hock and K. Schittkowski. Test examples for nonlinear programming codes. 
 * Lecture Notes in Economics and Mathematical Systems, 187, 1981. doi: 10.1007/978-3-642-48320-2.
 *
 */
struct hock_schittkowsky_71
{
    /// Fitness
    vector_double fitness(const vector_double &x) const
    {
        return {
            x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2],          // objfun
            x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3] - 40.,  // equality con.
            25. - x[0]*x[1]*x[2]*x[3]                       // inequality con.
        };
    }

    /// Number of objectives (one)
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }

    /// Equality constraint dimension (one)
    vector_double::size_type get_nec() const
    {
        return 1u;
    }

    /// Inequality constraint dimension (one)
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{1.,1.,1.,1.},{5.,5.,5.,5.}};
    }

    /// Gradients (dense)
    vector_double gradient(const vector_double &x) const
    {
        return {
            x[0]*x[3] + x[3]*(x[0] + x[1] + x[2]), x[0]*x[3], x[0]*x[3]+1,x[0]*(x[0]+x[1]+x[2]),
            2*x[0], 2*x[1], 2*x[2], 2*x[3],
            -x[1]*x[2]*x[3], -x[0]*x[2]*x[3], -x[0]*x[1]*x[3], -x[0]*x[1]*x[2]
        };
    }

    /// Hessians (sparse)
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        return {
            {2*x[3],x[3],x[3],2*x[0]+x[1]+x[2],x[0],x[0]},
            {2.,2.,2.,2.},
            {-x[2]*x[3], -x[1]*x[3],-x[0]*x[3],-x[1]*x[2],-x[0]*x[2],-x[0]*x[1]}
        };
    }

    /// Hessian sparsity
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {
            {{0,0},{1,0},{2,0},{3,0},{3,1},{3,2}}, 
            {{0,0},{1,1},{2,2},{3,3}}, 
            {{1,0},{2,0},{2,1},{3,0},{3,1},{3,2}}
        };
    }

    /// Problem name
    std::string get_name() const
    {   
        return "Hock Schittkowsky 71";
    }

    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tProblem number 71 from the Hock-Schittkowsky test suite\n";
    }
    
    /// Optimal solution 
    vector_double best_known() const
    {
        return {1.,4.74299963,3.82114998,1.37940829};
    }

    /// Serialization
    template <typename Archive>
    void serialize(Archive &) {}
};

}

PAGMO_REGISTER_PROBLEM(pagmo::hock_schittkowsky_71)

#endif
