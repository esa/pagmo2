#ifndef PAGMO_PROBLEM_ACKLEY_HPP
#define PAGMO_PROBLEM_ACKLEY_HPP

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../detail/constants.hpp"
#include "../exceptions.hpp"
#include "../problem.hpp" // needed for cereal registration macro
#include "../types.hpp"

namespace pagmo
{

/// The Ackley problem.
/**
 *
 * \image html ackley.png "Two-dimensional Ackley function." width=3cm
 *
 * This is a scalable box-constrained continuous single-objective problem.
 * The objective function is the generalised n-dimensional Ackley function:
 * \f[
 * 	F\left(x_1,\ldots,x_n\right) = 20 + e - 20e^{-\frac 15 \sqrt{\frac 1n \sum_{i=1}^n x_i^2}} - e^{\frac 1n \sum_{i=1}^n \cos(2\pi x_i)}, \quad x_i \in \left[ -15,30 \right].
 * \f]
 * The global minimum is in \f$x_i=0\f$, where \f$ F\left( 0,\ldots,0 \right) = 0 \f$.
 */
struct ackley
{
    /// Constructor from dimension
    ackley(unsigned int dim = 1u) : m_dim(dim)
    {
        if (dim < 1u) {
            pagmo_throw(std::invalid_argument, "Ackley Function must have minimum 1 dimension, " + std::to_string(dim) + " requested");
        }
    };
    /// Fitness
    vector_double fitness(const vector_double &x) const
    {
        vector_double f(1,0.);
        auto n = x.size();
        double omega = 2. * detail::pi();
        double s1 = 0., s2 = 0.;
        double nepero=std::exp(1.0);

        for (decltype(n) i = 0u; i < n; i++){
            s1 += x[i]*x[i];
            s2 += std::cos(omega*x[i]);
        }
        f[0] = -20 * std::exp(-0.2 * std::sqrt(1.0 / static_cast<double>(n) * s1)) - std::exp(1.0 / static_cast<double>(n) * s2) + 20 + nepero;
        return f;
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        vector_double lb(m_dim,-15);
        vector_double ub(m_dim, 30);
        return {lb,ub};
    }
    /// Problem name
    std::string get_name() const
    {
        return "Ackley Function";
    }
    /// Optimal solution
    vector_double best_known() const
    {
        return vector_double(m_dim, 0.);
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

PAGMO_REGISTER_PROBLEM(pagmo::ackley)

#endif
