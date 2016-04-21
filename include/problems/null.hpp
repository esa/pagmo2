#ifndef PAGMO_PROBLEM_NULL
#define PAGMO_PROBLEM_NULL

#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{

/// Null problem
/**
 * This problem is used to test, develop and provide default values to e.g. meta-problems
 */
struct null
{
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0.,0.,0.};
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
        return {{0.},{1.}};
    }

    /// Gradients
    vector_double gradient(const vector_double &) const
    {
        return {0.,0.,0.};
    }

    /// Gradient sparsity
    sparsity_pattern gradient_sparsity(const vector_double &) const
    {
        return {{0,0}};
    }   

    /// Hessians
    std::vector<vector_double> hessians(const vector_double &) const
    {
        return {{0.}};
    }

    /// Hessian sparsity
    std::vector<sparsity_pattern> hessians_sparsity() const
    {
        return {{{0,0}},{{0,0}},{{0,0}}};
    }

    /// Problem name
    std::string get_name() const
    {   
        return "Null problem";
    }

    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tA ficticious problem useful to test, debug and initialize default constructors";
    }
    
    /// Optimal solution 
    vector_double best_known() const
    {
        return {0.};
    }

    /// Serialization
    template <typename Archive>
    void serialize(Archive &) {}
};

}

PAGMO_REGISTER_PROBLEM(pagmo::null)

#endif
