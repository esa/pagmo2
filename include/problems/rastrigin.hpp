#ifndef PAGMO_PROBLEM_RASTRIGIN_HPP
#define PAGMO_PROBLEM_RASTRIGIN_HPP

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../detail/constants.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{

/// The Rastrigin problem.
/**
 *
 * \image html rastrigin.png "Two-dimensional Rastrigin function." width=3cm
 *
 * This is a scalable box-constrained continuous single-objective problem.
 * The objective function is the generalised n-dimensional Rastrigin function:
 * \f[
 * 	F\left(x_1,\ldots,x_n\right) = 10 \cdot n + \sum_{i=1}^n x_i^2 - 10\cdot\cos\left( 2\pi \cdot x_i \right), \quad x_i \in \left[ -5.12,5.12 \right].
 * \f]
 * The global minimum is in the origin, where \f$ F\left( 0,\ldots,0 \right) = 0 \f$.
 */
struct rastrigin
{
    /// Constructor from dimension
    rastrigin(unsigned int dim = 1u) : m_dim(dim)
    {
        if (dim < 1u) {
            pagmo_throw(std::invalid_argument, "Rosenbrock Function must have minimum 1 dimension, " + std::to_string(dim) + " requested");
        }
    };
    /// Fitness
    vector_double fitness(const vector_double &x) const
    {
        vector_double f(1, 0.);
        const auto omega = 2. * pagmo::detail::pi();
        const auto n = x.size();
        for (decltype(n) i = 0u; i < n; ++i) {
            f[0] += x[i] * x[i] - 10.0 * std::cos(omega * x[i]);
        }
        f[0] += 10. * n;
        return f;
    }

    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(m_dim,-5.12);
        vector_double ub(m_dim, 5.12);
        return {lb,ub};
    }
    /// Problem name
    std::string get_name() const
    {
        return "Rastrigin Function";
    }
    /// Optimal solution
    vector_double best_known() const
    {
        return vector_double(m_dim,0.);
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

PAGMO_REGISTER_PROBLEM(pagmo::rastrigin)

#endif
