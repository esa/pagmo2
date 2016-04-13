#ifndef PAGMO_PROBLEM_HOCK_SCHITTKOWSKY_71
#define PAGMO_PROBLEM_HOCK_SCHITTKOWSKY_71

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{

namespace problem
{

struct hock_schittkowsky_71
{
    // fitness
    vector_double fitness(const vector_double &x) const
    {
        return {
            x[0]*x[3]*(x[0] + x[1] + x[2]),                 // objfun
            x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3] - 40.,  // equality con.
            25. - x[0]*x[1]*x[2]*x[3]                       // inequality con.
        };
    }

    // problem dimension
    vector_double::size_type get_n() const
    {
        return 4u;
    }

    // number of objectives (single objective)
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }

    // equality constraint dimension
    vector_double::size_type get_nec() const
    {
        return 1u;
    }

    // inequality constraint dimension
    vector_double::size_type get_nic() const
    {
        return 1u;
    }
    
    // problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{1.,1.,1.,1.},{5.,5.,5.,5.}};
    }

    // gradients (dense)
    vector_double gradient(const vector_double &x) const
    {
        return {
            x[0]*x[3] + x[3]*(x[0] + x[1] + x[2]), x[0]*x[3], x[0]*x[3]+1,x[0]*(x[0]+x[1]+x[2]),
            2*x[0], 2*x[1], 2*x[2], 2*x[3],
            -x[1]*x[2]*x[3], -x[0]*x[2]*x[3], -x[0]*x[1]*x[3], -x[0]*x[1]*x[2]
        };
    }

    // hessians
    std::vector<vector_double> hessians(const vector_double &x) const
    {
        return {
            {2*x[3],x[3],x[3],2*x[0]+x[1]+x[2],x[0],x[0]},
            {2.,2.,2.,2.},
            {x[2]*x[3], x[1]*x[3],x[0]*x[3],x[1]*x[2],x[0]*x[2],x[0]*x[1]}
        };
    }

    // hessian sparsity
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {
            {{0,0},{1,0},{2,0},{3,0},{3,1},{3,2}}, 
            {{0,0},{1,1},{2,2},{3,3}}, 
            {{1,0},{2,0},{2,1},{3,0},{3,1},{3,2}}
        };
    }

    // Problem name
    std::string get_name() const
    {   
        return std::string("Hock Schittkowsky 71");
    }

    // Extra informations
    std::string extra_info() const
    {
        std::ostringstream s;
        s << "\tProblem number 71 from the Hock-Schittkowsky test suite" << '\n';
        return s.str();
    }
    
    // Optimal solution 
    vector_double best_known() const
    {
        return {1.,4.74299963,3.82114998,1.37940829};
    }

    // Serialization
    template <typename Archive>
    void serialize(Archive &) {}
};

}}

PAGMO_REGISTER_PROBLEM(pagmo::problem::hock_schittkowsky_71)

#endif
