#ifndef PAGMO_PROBLEM_ZDT
#define PAGMO_PROBLEM_ZDT

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../detail/constants.hpp"
#include "../io.hpp"
#include "../problem.hpp"
#include "../types.hpp"

namespace pagmo
{

/// ZDT problem test suite
/**
 *
 * This widespread test suite was conceived for two-objective problems and takes its name from its
 * authors Zitzler, Deb and Thiele. 
 * 
 * In their paper the authors propose a set of 6 different scalable problems all originating from a
 * well thought combination of functions allowing, by construction, to measure the distance of
 * any point to the Pareto front while creating interesting problems. They also suggest some
 * dimensions for instantiating the problems, namely \f$m = [30, 30, 30, 10, 11, 10]\f$, which are 
 * here used as default values.
 *
 * @note The ZDT5 problem is an integer problem, its chromosome is here represented with doubles floored
 * via std::floor
 *
 * @see Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary algorithms: 
 * Empirical results." Evolutionary computation 8.2 (2000): 173-195. doi: 10.1.1.30.5848
 *
 * ZDT1:
 *
 * This is a box-constrained continuous \f$n\f$-dimensional (\f$n\f$>1) multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *  g\left(x\right) = 1 + 9 \left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right) \\
 *  F_1 \left(x\right) = x_1 \\
 *  F_2 \left(x\right) = g(x) \left[ 1 - \sqrt{x_1 / g(x)} \right]  x \in \left[ 0,1 \right].
 * \end{array}
 * \f]
 *
 * ZDT2:
 *
 * This is a box-constrained continuous \f$n\f$-dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 1 + 9 \left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right) \\
 *      F_1 \left(x\right) = x_1 \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - \left(x_1 / g(x)\right)^2 \right]  x \in \left[ 0,1 \right].
 * \end{array}
 * \f]
 *
 * ZDT3:
 *
 * This is a box-constrained continuous \f$n\f$-dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 1 + 9 \left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right) \\
 *      F_1 \left(x\right) = x_1 \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - \sqrt{x_1 / g(x)} - x_1/g(x) \sin(10 \pi x_1) \right]  x \in \left[ 0,1 \right].
 * \end{array}
 * \f]
 *
 * ZDT4:
 *
 * This is a box-constrained continuous \f$n\f$-dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 91 + \sum_{i=2}^{n} \left[x_i^2 - 10 \cos \left(4\pi x_i \right) \right] \\
 *      F_1 \left(x\right) = x_1 \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - \sqrt{x_1 / g(x)} \right]  x_1 \in [0,1], x_i \in \left[ -5,5 \right] i=2, \cdots, 10.
 * \end{array}
 * \f]
 *
 * ZDT5
 *
 * This is a box-constrained integer \f$n\f$-dimension multi-objecive problem. The chromosome is
 * a bitstring so that \f$x_i \in \left\{0,1\right\}\f$. Refer to the original paper for the formal definition.
 *
 * ZDT6
 *
 * This is a box-constrained continuous \f$n\f$--dimension multi-objecive problem.
 * \f[
 * \begin{array}{l}
 *      g\left(x\right) = 1 + 9 \left[\left(\sum_{i=2}^{n} x_i \right) / \left( n-1 \right)\right]^{0.25} \\
 *      F_1 \left(x\right) = 1 - \exp(-4 x_1) \sin^6(6 \pi \ x_1) \\
 *      F_2 \left(x\right) = g(x) \left[ 1 - (f_1(x) / g(x))^2  \right]  x \in \left[ 0,1 \right].
 * \end{array}
 * \f]
 *
 */

class zdt
{
public:
    /** Constructor
     *
     * Will construct one problem from the ZDT test-suite.
     *
     * @param[in] id problem number. Must be in [1, .., 6]
     * @param[in] param problem parameter, representing the problem dimension
     * except for ZDT5 where it represents the number of binary strings
     *
     * @throws std::invalid_argument if \p id is not in [1,..,6]
     * @throws std::invalid_argument if \p param is not at least 2.
     */
    zdt(unsigned int id = 1u, unsigned int param = 30u) : m_id(id), m_param(param) 
    {
        if (param < 2) {
            pagmo_throw(std::invalid_argument, "ZDT test problems must have a minimum value of 2 for the constructing parameter (representing the dimension except for ZDT5), " + std::to_string(param) + " requested");
        }
        if (id == 0 || id > 6) {
            pagmo_throw(std::invalid_argument, "ZDT test suite contains six (id=[1 ... 6]) problems, id=" + std::to_string(id) + " requested");
        }
    };
    /// Fitness
    vector_double fitness(const vector_double &x) const
    {
        switch(m_id)
        {
        case 1u:
            return zdt1_fitness(x);
            break;
        case 2u:
            return zdt2_fitness(x);
            break;
        case 3u:
            return zdt3_fitness(x);
            break;
        case 4u:
            return zdt4_fitness(x);
            break;
        case 5u:
            return zdt5_fitness(x);
            break;
        case 6u:
            return zdt6_fitness(x);
            break;
        default:
            pagmo_throw(std::invalid_argument, "Error: There are only 6 test functions in the ZDT test suite!");
            break;
        }
    }
    /// Number of objectives
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }

    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        switch(m_id)
        {
        case 1u:
        case 2u:
        case 3u:
        case 6u:
        {
            return {vector_double(m_param, 0.), vector_double(m_param, 1.)};
        }
        case 4u:
        {
            vector_double lb(m_param,-5.);
            vector_double ub(m_param,5.);
            lb[0] = 0.0;
            ub[0] = 1.0;
            return {lb, ub};
        }
        case 5u:
        {
            auto dim = 30u + 5u * (m_param - 1u);
            return {vector_double(dim, 0.), vector_double(dim, 2.)}; // the bounds [0,2] guarantee that floor(x) will be in [0,1] as the rng generates in [0,2)
            break;
        }
        default:
            pagmo_throw(std::invalid_argument, "Error: There are only 6 test functions in the ZDT test suite!");
            break;
        }
    }
    /// Problem name
    std::string get_name() const
    {   
        return "ZDT" + std::to_string(m_id);
    }
    /* Convergence metric for a given decision_vector (0 = on the optimal front)
     *
     * Introduced by Martens and Izzo, this metric is able
     * to measure "a distance" of any point from the pareto front of any ZDT 
     * problem analytically without the need to precompute the front.
     *
     * @see Märtens, Marcus, and Dario Izzo. "The asynchronous island model
     * and NSGA-II: study of a new migration operator and its performance." 
     * Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013.
     *
     * @throws std::invalid_argument if the problem id is not 1-6
     *
     
    double p_distance(const vector_double &x) const
    {
        switch(m_id)
        {
        case 1u:
        case 2u:
        case 3u:
            return zdt123_p_distance(x);
        case 4u:
            return zdt4_p_distance(x);
        case 5u:
            return zdt5_p_distance(x);
        case 6u:
            return zdt6_p_distance(x);
        default:
            pagmo_throw(std::invalid_argument, "Error: There are only 6 test functions in this test suite!");
        }
    }*/

private:
    friend class cereal::access; 
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_id, m_param);
    }
    /// Problem dimensions
    unsigned int m_id;
    unsigned int m_param;

private:
    vector_double zdt1_fitness(const vector_double &x) const
    {
        double g = 0.;
        vector_double f(2,0.);
        f[0] = x[0];
        auto N = x.size();

        for(decltype(N) i = 1u; i < N; ++i) {
            g += x[i];
        }
        g = 1. + (9. * g) / static_cast<double>(N - 1u);

        f[1] = g * ( 1. - sqrt(x[0]/g));
        return f;
    }

    vector_double zdt2_fitness(const vector_double &x) const
    {
        double g = 0.;
        vector_double f(2,0.);
        f[0] = x[0];
        auto N = x.size();

        for(decltype(N) i = 1u; i < N; ++i) {
                g += x[i];
        }
        g = 1. + (9. * g) / static_cast<double>(N - 1u);
        f[1] = g * ( 1. - (x[0]/g)*(x[0]/g));

        return f;
    }

    vector_double zdt3_fitness(const vector_double &x) const
    {
        double g = 0.;
        vector_double f(2,0.);
        f[0] = x[0];
        auto N = x.size();

        for(decltype(N) i = 1u; i < N; ++i) {
                g += x[i];
        }
        g = 1. + (9. * g) / static_cast<double>(N - 1u);
        f[1] = g * ( 1. - sqrt(x[0]/g) - x[0]/g * sin(10. * pagmo::detail::pi() * x[0]));

        return f;
    }

    vector_double zdt4_fitness(const vector_double &x) const
    {
        double g = 0.;
        vector_double f(2,0.);
        f[0] = x[0];
        auto N = x.size();

        g = 1 + 10 * static_cast<double>(N - 1u);
        f[0] = x[0];
        for(decltype(N) i = 1u; i < N; ++i) {
            g += x[i]*x[i] - 10. * cos(4. * pagmo::detail::pi() * x[i]);
        }
        f[1] = g * ( 1. - sqrt(x[0]/g) );

        return f;
    }

    vector_double zdt5_fitness(const vector_double &x_double) const
    {
        double g = 0.;
        vector_double f(2,0.);
        auto size_x = x_double.size();
        auto n_vectors = ((size_x-30u) / 5u) + 1u;

        int k = 30;
        std::vector<vector_double::size_type> u(n_vectors, 0u);
        std::vector<vector_double::size_type> v(n_vectors);

        // Convert the input vector into floored values (integers)
        vector_double x(size_x);
        std::transform(x_double.begin(), x_double.end(), x.begin(), [](auto item) {return std::floor(item);});
        f[0] = x[0];

        // Counts how many 1s are there in the first (30 dim)
        u[0] = std::count(x.begin(), x.begin() + 30, 1.);

        for (decltype(n_vectors) i = 1u; i < n_vectors; ++i) {
            for (int j = 0; j < 5; ++j) {
                if (x[k] == 1.) {
                    ++u[i];
                }
                ++k;
            }
        }
        f[0] = 1.0 + static_cast<double>(u[0]);
        for (decltype(n_vectors) i = 1u; i<n_vectors; ++i) {
            if (u[i] < 5u) {
                v[i] = 2u + u[i];
            }
            else {
                v[i] = 1u;
            }
        }
        for (decltype(n_vectors) i = 1u; i<n_vectors; ++i) {
            g += static_cast<double>(v[i]);
        }
        f[1] = g * (1. / f[0]);
        return f;
    }

vector_double zdt6_fitness(const vector_double &x) const
{
        double g = 0.;
        vector_double f(2,0.);
        f[0] = x[0];
        auto N = x.size();


        f[0] = 1 - exp(-4*x[0])*pow(sin(6*pagmo::detail::pi()*x[0]),6);
        for(decltype(N) i = 1; i < N; ++i) {
                g += x[i];
        }
        g = 1 + 9 * pow((g / static_cast<double>(N - 1u)),0.25);
        f[1] = g * ( 1 - (f[0]/g)*(f[0]/g));

        return f;
}

};

}

PAGMO_REGISTER_PROBLEM(pagmo::zdt)

#endif
