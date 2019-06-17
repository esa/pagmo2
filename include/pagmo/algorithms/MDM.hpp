

#ifndef PAGMO_ALGORITHMS_MDM_HPP
#define PAGMO_ALGORITHMS_MDM_HPP

#include <algorithm> // std::shuffle, std::transform
#include <iomanip>
#include <numeric> // std::iota, std::inner_product
#include <random>
#include <string>
#include <tuple>

#include <pagmo/algorithm.hpp> // needed for the cereal macro
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/valuefunc.cpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/utils/generic.hpp>         // uniform_real_from_range, some_bound_is_equal
#include <pagmo/utils/multi_objective.hpp> // crowding_distance, etc..
#include <pagmo/problems/dtlz.hpp>

namespace pagmo
{
/// Machine Decision Maker
/**
This class contains all the functions related to preference function, noise, biases and etc.
 */
class MDM
{
public:
    /// Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned int, unsigned long long, vector_double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs the MDM user defined algorithm.
     *
     * @param[in] util indicates the type of utility function.

     */
    MDM(unsigned util = 1u)
        : m_util(util)
    {

    }

  double tchebycheff_utility(const vector<double> &f, const vector<double> &w, const vector<double> &ideal)
    {
        vector<double> x;
        x.reserve(w.size());
        for (size_t i = 0; i < w.size(); i++) {
            x[i] = w[i] * std::abs(f[i] - ideal[i]);
        }
        return max_of(x);
    }

    double linear_utility(const vector<double> &f, const vector<double> &w, const vector<double> &ideal)
    {
        double sum = 0.0;
        for (size_t i = 0; i < w.size(); i++) {
            sum += w[i] * std::abs(f[i] - ideal[i]);
        }
        return sum;
    }

    double quadratic_utility(const vector<double> &f, const vector<double> &w, const vector<double> &ideal)
    {
        double x;
        double sum = 0.0;
        for (size_t i = 0; i < w.size(); i++) {
            x = w[i] * (f[i] - ideal[i]);
            sum += (x * x);
        }
        return sqrt(sum);
    }

    /* iTEAX, IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, vol. 14,
       no. 5, October 2010. */

    double proportional_gaussian_error(double u, double alpha)
    {
        double sigma = alpha * u;
        double noise = Rand_normal(0.0, sigma * sigma);
        return u * (1 + noise);
    }

    /* Campigotto and Passerini (BC-EMO).

       Probability of incorrect feedback when comparing u and v is gamma.
       It returns an integer less than, equal to, or greater than zero if
       u is found, respectively, to be less than, to match, or be greater
       than v.  (default gamma: 0.3).
    */
    int cmp_utilities_noisy_1(double u, double v, double gamma)
    {
        int res = u - v;
        if (Rand() <= gamma) return -res;
        return res;
    }

    /* Campigotto and Passerini (BC-EMO).

       Probability of incorrect feedback when comparing u and v is gamma.
       It returns an integer less than, equal to, or greater than zero if
       u is found, respectively, to be less than, to match, or be greater
       than v.
    */
    int cmp_utilities_noisy_2(double u, vector<double> &vec_u, double v, vector<double> &vec_v, double gamma)
    {
        double dist = euclid_distance(vec_u, vec_v);
        gamma = gamma * dist / sqrt(dist * dist);

        return cmp_utilities_noisy_1(u, v, gamma);
    }

    /* Marginal value function (Eq. 2) used by

       T. J. Stewart. Robustness of additive value function methods in
       MCDM. Journal of Multi-Criteria Decision Analysis, 5(4):301–309,
       1996.

    */

    double valuefunc_stewart(double obj, double alpha, double beta, double lambda, double tau)
    {
        assert(obj >= 0);
        assert(obj <= 1);
        if (obj <= tau) {
            return lambda * (exp(alpha * obj) - 1.0) / (exp(alpha * tau) - 1.0);
        } else {
            return lambda + (1.0 - lambda) * (1.0 - exp(-beta * (obj - tau))) / (1.0 - exp(-beta * (1.0 - tau)));
        }
    }

    double valuefunc(int m, const vector<double> &obj, const double *alpha, const double *beta, const double *lambda,
                     const double *tau, const double *weights)
    {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += weights[i] * valuefunc_stewart(obj[i], alpha[i], beta[i], lambda[i], tau[i]);
        }
        return sum;
    }

    double estim_valuefunc(int m, const vector<double> &obj, const double *alpha, const double *beta,
                           const double *lambda, const double *tau, const double *weights, double gamma, double sigma,
                           double delta, int q)
    {
        vector<int> c(m);

        if (q < m) {
            select_criteria_subset(q, weights, m, c);
        } else {
            assert(q == m);
            for (size_t i = 0; i < m; i++) {
                c[i] = i;
            }
        }

        vector<double> z_mod(obj);
        modified_criteria(z_mod, c, q, gamma);

        /*
           (a) the m - q unmodelled criteria set at their
           reference levels tau_i (i.e. no perceived gains or
           losses)
        */
        for (int i = q; i < m; i++) {
            z_mod[c[i]] = tau[c[i]];
        }
        /*
          (b) the addition of a noise term, normally
          distributed with zero mean and a variance of
          sigma^2 (which will be a specified model parameter),
        */
        double noise = Rand_normal(0.0, sigma * sigma);

        /* (c) a shift in the reference levels tau_i from the ‘ideal’
           positions by an amount \delta, which may be
           positive or negative (and which is also a
           specified model parameter).
        */
        double *tau_mod = (double *)malloc(sizeof(double) * m);
        for (int i = 0; i < m; i++) {
            tau_mod[i] = tau[i] + delta;
        }

        double estim_v = noise + valuefunc(m, z_mod, alpha, beta, lambda, tau_mod, weights);
        free(tau_mod);
        return estim_v;
    }

    
    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned int seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    };
    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned int get_seed() const
    {
        return m_seed;
    }
  
    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }
    /// Algorithm name
    /**
     * Returns the name of the algorithm.
     *
     * @return <tt> std::string </tt> containing the algorithm name
     */
    std::string get_name() const
    {
        return "MDM";
    }
 

    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the UDP and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen, m_cr, m_eta_c, m_m, m_eta_m, m_e, m_seed, m_verbosity, m_log);
    }

private:
    

    unsigned int m_gen;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::MDM)

#endif
