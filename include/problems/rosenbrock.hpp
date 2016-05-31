#ifndef PAGMO_PROBLEM_ROSENBROCK_HPP
#define PAGMO_PROBLEM_ROSENBROCK_HPP

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../exceptions.hpp"
#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{

/// The Rosenbrock problem.
/**
 * \image html rosenbrock.png "Two-dimensional Rosenbrock function." width=3cm
 *
 * This is a box-constrained continuous single-objective problem.
 * The objective function is the generalised n-dimensional Rosenbrock function:
 * \f[
 *  F\left(x_1,\ldots,x_n\right) =
 *  \sum_{i=1}^{n-1}\left[ 100\left(x_i^2-x_{i+1}\right)^2+\left(x_i-1\right)^2\right], \quad x_i \in \left[ -5,10 \right].
 * \f]
 * The global minimum is in \f$x_i=1\f$, where \f$ F\left( 1,\ldots,1 \right) = 0 \f$.
 */
struct rosenbrock
{
    /// Constructor from dimension
    /**
     * @param[in] dim problem dimension
     * @throw std::invalid_argument if \p dim is less than 2
     */
    rosenbrock(unsigned int dim = 2u) : m_dim(dim)
    {
        if (dim < 2u) {
            pagmo_throw(std::invalid_argument, "Rosenbrock Function must have minimum 2 dimensions, " + std::to_string(dim) + " requested");
        }
    };
    /// Fitness
    vector_double fitness(const vector_double &x) const
    {
        vector_double f(1,0.);
        for (decltype(m_dim) i=0u; i < m_dim-1u; ++i){
                f[0] += 100. * (x[i]*x[i] -x[i+1])*(x[i]*x[i] -x[i+1]) + (x[i]-1)*(x[i]-1);
        }
        return f;
    }

    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(m_dim,-5.);
        vector_double ub(m_dim, 10.);
        return {lb,ub};
    }
    /// Problem name
    std::string get_name() const
    {
        return "Multidimensional Rosenbrock Function";
    }
    /// Optimal solution
    vector_double best_known() const
    {
        return vector_double(m_dim,1.);
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_dim);
    }
    /// Problem dimensions
    unsigned int m_dim;
};

} // namespace pagmo

PAGMO_REGISTER_PROBLEM(pagmo::rosenbrock)

#endif
