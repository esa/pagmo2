#ifndef PAGMO_ALGORITHMS_CMAES_HPP
#define PAGMO_ALGORITHMS_CMAES_HPP

#include <iomanip>
#include <random>
#include <string>
#include <tuple>
#include <Eigen/Dense>

#include "../algorithm.hpp"
#include "../io.hpp"
#include "../exceptions.hpp"
#include "../population.hpp"
#include "../rng.hpp"
#include "../utils/generic.hpp"

namespace pagmo
{
/// Covariance Matrix Adaptation Evolutionary Strategy
class cmaes
{
public:
    /// Constructor.
    cmaes(unsigned int gen = 1, double cc = -1, double cs = -1, double c1 = -1, double cmu = -1, double sigma0 = 0.5, double ftol = 1e-6, double xtol = 1e-6, bool memory = false, unsigned int seed = pagmo::random_device::next()) :
        m_gen(gen), m_cc(cc), m_cs(cs), m_c1(c1),
        m_cmu(cmu), m_sigma0(sigma0), m_ftol(ftol),
        m_xtol(xtol), m_memory(memory),
        m_e(seed), m_seed(seed), m_verbosity(0u)
    {
        if ( ((cc < 0.) || (cc > 1.)) && !(cc==-1) ) {
            pagmo_throw(std::invalid_argument, "cc must be in [0,1] or -1 if its value has to be initialized automatically, a value of " + std::to_string(cc) + " was detected");
        }
        if ( ((cs < 0.) || (cs > 1.)) && !(cs==-1) ){
            pagmo_throw(std::invalid_argument, "cs needs to be in [0,1] or -1 if its value has to be initialized automatically, a value of " + std::to_string(cs) + " was detected");
        }
        if ( ((c1 < 0.) || (c1 > 1.)) && !(c1==-1) ){
            pagmo_throw(std::invalid_argument, "c1 needs to be in [0,1] or -1 if its value has to be initialized automatically, a value of " + std::to_string(c1) + " was detected");
        }
        if ( ((cmu < 0.) || (cmu > 1.)) && !(cmu==-1) ){
            pagmo_throw(std::invalid_argument, "cmu needs to be in [0,1] or -1 if its value has to be initialized automatically, a value of " + std::to_string(cmu) + " was detected");
        }

        // Initialize the algorithm memory
        sigma = m_sigma0;
        mean = Eigen::VectorXd::Zero(1);
        variation = Eigen::VectorXd::Zero(1);
        newpop = std::vector<Eigen::VectorXd>{};
        B = Eigen::MatrixXd::Identity(1,1);
        D = Eigen::MatrixXd::Identity(1,1);
        C = Eigen::MatrixXd::Identity(1,1);
        invsqrtC = Eigen::MatrixXd::Identity(1,1);
        pc = Eigen::VectorXd::Zero(1);
        ps = Eigen::VectorXd::Zero(1);
        counteval = 0u;
        eigeneval = 0u;
    }

    /// Algorithm evolve method (juice implementation of the algorithm)
    population evolve(population pop) const
    {
        // We store some useful variables
        const auto &prob = pop.get_problem();       // This is a const reference, so using set_seed for example will not be allowed (pop.set_problem_seed is)
        auto dim = prob.get_nx();                   // This getter does not return a const reference but a copy
        const auto bounds = prob.get_bounds();
        const auto &lb = bounds.first;
        const auto &ub = bounds.second;
        auto lam = pop.size();
        auto mu = lam / 2u;
        auto prob_f_dimension = prob.get_nf();
        //auto fevals0 = prob.get_fevals();           // discount for the already made fevals
        //auto count = 1u;                            // regulates the screen output

        // PREAMBLE--------------------------------------------------
        // particular algorithm.
        if (prob.get_nc() != 0u) {
            pagmo_throw(std::invalid_argument,"Non linear constraints detected in " + prob.get_name() + " instance. " + get_name() + " cannot deal with them");
        }
        if (prob_f_dimension != 1u) {
            pagmo_throw(std::invalid_argument,"Multiple objectives detected in " + prob.get_name() + " instance. " + get_name() + " cannot deal with them");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        if (lam < 5u) {
            pagmo_throw(std::invalid_argument, prob.get_name() + " needs at least 5 individuals in the population, " + std::to_string(lam) + " detected");
        }
        // -----------------------------------------------------------

        // No throws, all valid: we clear the logs
        // m_log.clear();

        // Initializing the random number generators
        std::uniform_real_distribution<double> randomly_distributed_number(0.,1.); // to generate a number in [0, 1)
        std::normal_distribution<double> normally_distributed_number(0.,1.);           // to generate a normally distributed number

        // Setting coefficients for Selection
        Eigen::VectorXd weights(_(mu));
        for (decltype(weights.rows()) i = 0u; i < weights.rows(); ++i){
            weights(i) = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i) + 1.);
        }
        weights /= weights.sum();                                   // weights for the weighted recombination
        double mueff = 1.0 / (weights.transpose() * weights);       // variance-effectiveness of sum w_i x_i

        // Setting coefficients for Adaptation automatically or to user defined data
        double cc(m_cc), cs(m_cs), c1(m_c1), cmu(m_cmu);
        double N = static_cast<double>(dim);
        if (cc == -1) {
            cc = (4. + mueff / N) / (N + 4. + 2. * mueff / N);                                // t-const for cumulation for C
        }
        if (cs == -1) {
            cs = (mueff + 2.) / (N + mueff + 5.);                                           // t-const for cumulation for sigma control
        }
        if (c1 == -1) {
            c1 = 2. / ((N + 1.3) * (N + 1.3) + mueff);                                      // learning rate for rank-one update of C
        }
        if (cmu == -1) {
            cmu = 2. * (mueff - 2 + 1 / mueff) / ((N + 2) * (N + 2) + mueff);               // and for rank-mu update
        }

        double damps = 1. + 2. * std::max(0., std::sqrt((mueff - 1.)/(N + 1.)) - 1.) + cs;  // damping coefficient for sigma
        double chiN = std::sqrt(N) * (1. - 1. / (4. * N) + 1. / (21 * N * N));              // expectation of ||N(0,I)|| == norm(randn(N,1))


        // Some buffers
        Eigen::VectorXd meanold = Eigen::VectorXd::Zero(_(dim));
        Eigen::MatrixXd Dinv = Eigen::MatrixXd::Identity(_(dim), _(dim));
        Eigen::MatrixXd Cold = Eigen::MatrixXd::Identity(_(dim), _(dim));
        Eigen::VectorXd tmp = Eigen::VectorXd::Zero(_(dim));
        std::vector<Eigen::VectorXd> elite(mu, tmp);
        vector_double dumb(dim, 0.);
        double var_norm = 0.;

        // If the algorithm is called for the first time on this problem dimension / pop size or if m_memory is false we erease the memory of past calls
        if ( (newpop.size() != lam) || ((unsigned int)newpop[0].rows() != dim) || (m_memory==false) ) {
            sigma = m_sigma0;
            mean.resize(_(dim));
            auto idx_b = pop.best_idx();
            for (decltype(dim) i = 0u; i < dim; ++i){
                mean(_(i)) = pop.get_x()[idx_b][i];
            }
            newpop = std::vector<Eigen::VectorXd>(lam, tmp);
            variation.resize(_(dim));

            //We define the starting B,D,C
            B = Eigen::MatrixXd::Identity(_(dim), _(dim));                 //B defines the coordinate system
            D = Eigen::MatrixXd::Identity(_(dim), _(dim));                 //diagonal D defines the scaling. By default this is the witdh of the box bounds.
                                                                     //If this is too small... then 1e-6 is used
            // TODO: here if the problem is unbounded what happens?
            for (decltype(dim) j = 0u; j < dim; ++j) {
                D(_(j),_(j)) = std::max((ub[j]-lb[j]), 1e-6);
            }
            C = Eigen::MatrixXd::Identity(_(dim), _(dim));                 //covariance matrix C
            C = D * D;
            invsqrtC = Eigen::MatrixXd::Identity(_(dim), _(dim));   //inverse of sqrt(C)
            for (decltype(dim) j = 0; j < dim; ++j) {
                invsqrtC(_(j),_(j)) = 1. / D(_(j),_(j));
            }
            pc = Eigen::VectorXd::Zero(_(dim));
            ps = Eigen::VectorXd::Zero(_(dim));
            counteval = 0u;
            eigeneval = 0u;
        }

        // ----------------------------------------------//
        // HERE WE START THE JUICE OF THE ALGORITHM      //
        // ----------------------------------------------//
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(_(dim));
        for (decltype(m_gen) g = 1u; g < m_gen; ++g) {
            // 1 - We generate and evaluate lam new individuals
            for (decltype(lam) i = 0u; i < lam; ++i ) {
                // 1a - we create a randomly normal distributed vector
                for (decltype(dim) j = 0u; j < dim; ++j){
                    tmp(_(j)) = normally_distributed_number(m_e);
                }
                // 1b - and store its transformed value in the newpop
                newpop[i] = mean + (sigma * B * D * tmp);
            }
            //This is evaluated here on the last generated tmp and will be used only as
            //a stopping criteria
            var_norm = (sigma * B * D * tmp).norm();

            // 1bis - Check the exit conditions (every 5 generations) // we need to do it here as
            //termination is defined on tmp
            if (g % 5u == 0u) {
                // Exit condition on xtol
                if  ( (sigma*B*D*tmp).norm() < m_xtol ) {
                    if (m_verbosity > 0u) {
                        std::cout << "Exit condition -- xtol < " <<  m_xtol << std::endl;
                    }
                    return pop;
                }
                // Exit condition on ftol
                auto idx_b = pop.best_idx();
                auto idx_w = pop.worst_idx();
                double delta_f = std::abs(pop.get_f()[idx_b][0] - pop.get_f()[idx_w][0]);
                if (delta_f < m_ftol) {
                    if (m_verbosity) {
                        std::cout << "Exit condition -- ftol < " <<  m_ftol << std::endl;
                    }
                    return pop;
                }
            }

            // 1c - we fix the bounds. We cannot use the utils::generic::force_bounds_random as we here represent a chromosome
            // via an Eigen matrix. Maybe iterators could be used to generalize that util
            for (decltype(lam) i = 0u; i < lam; ++i) {
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    if ( (newpop[i](_(j)) < lb[j]) || (newpop[i](_(j)) > ub[j]) ) {
                        newpop[i](_(j)) = lb[j] + randomly_distributed_number(m_e) * (ub[j] - lb[j]);
                    }
                }
            }

            // 2 - We Evaluate the new population (if the problem is stochastic change seed first)
            if(prob.is_stochastic()) {
                // change the problem seed. This is done via the population_set_seed method as prob.set_seed
                // is forbidden being prob a const ref.
                pop.set_problem_seed(std::uniform_int_distribution<unsigned int>()(m_e));
                // re-evaluate the whole population w.r.t. the new seed
                for (decltype(lam) j = 0u; j < lam; ++j) {
                    pop.set_xf(j, pop.get_x()[j], prob.fitness(pop.get_x()[j]));
                }
            }
            // Reinsertion
            for (decltype(lam) i = 0u; i < lam; ++i) {
                for (decltype(dim) j = 0u; j < dim; ++j ) {
                    dumb[j] = newpop[i](_(j));
                }
                pop.set_x(i,dumb);
            }
            counteval += lam;

            // 2 - We extract the elite from this generation.
            std::vector<population::size_type> best_idx(lam);
            std::iota(best_idx.begin(), best_idx.end(), population::size_type(0));
            std::sort(best_idx.begin(), best_idx.end(), [&pop](auto idx1, auto idx2) {return pop.get_f()[idx1][0] < pop.get_f()[idx2][0];});
            best_idx.resize(mu); // not needed?
            for (decltype(mu) i = 0u; i < mu; ++i ) {
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    elite[i](_(j)) = pop.get_x()[best_idx[i]][j];
                }
            }

            // 3 - Compute the new mean of the elite storing the old one
            meanold = mean;
            mean = elite[0]*weights(0);
            for (decltype(mu) i = 1u; i < mu; ++i) {
                mean += elite[i] * weights(_(i));
            }

            // 4 - Update evolution paths
            ps = (1. - cs) * ps + std::sqrt(cs*(2. - cs)*mueff) * invsqrtC * (mean-meanold) / sigma;
            double hsig = 0.;
            hsig = (ps.squaredNorm() / N / (1. - std::pow((1. - cs),(2. * static_cast<double>(counteval / lam) ))) ) < (2. + 4. / (N + 1.));
            pc = (1. - cc) * pc + hsig * std::sqrt(cc * (2. - cc) * mueff) * (mean - meanold) / sigma;

            // 5 - Adapt Covariance Matrix
            Cold = C;
            C = (elite[0]-meanold)*(elite[0]-meanold).transpose()*weights(0);
            for (decltype(mu) i = 1u; i < mu; ++i ) {
                C += (elite[i]-meanold)*(elite[i]-meanold).transpose()*weights(_(i));
            }
            C /= sigma*sigma;
            C = (1. - c1-cmu) * Cold +
                cmu * C +
                c1 * ((pc * pc.transpose()) + (1. - hsig) * cc * (2. - cc) * Cold);

            //6 - Adapt sigma
            sigma *= std::exp( std::min( 0.6, (cs / damps) * (ps.norm() / chiN - 1.) ) );
            if ( !std::isfinite(sigma) || !std::isfinite(var_norm)) { //debug info
                std::cout << "eigen: " << es.info() << std::endl;
                std::cout << "B: " << B << std::endl;
                std::cout << "D: " << D << std::endl;
                std::cout << "Dinv: " << D << std::endl;
                std::cout << "invsqrtC: " << invsqrtC << std::endl;
                pagmo_throw(std::invalid_argument, "NaN!!!!! in CMAES");
            }

            //7 - Perform eigen-decomposition of C
            if ( static_cast<double>(counteval - eigeneval) > (static_cast<double>(lam) / (c1 + cmu) / N / 10u) ) {   //achieve O(N^2)
                eigeneval = counteval;
                C = (C + C.transpose()) / 2.;                            //enforce symmetry
                es.compute(C);                                          //eigen decomposition
                if (es.info()==Eigen::Success) {
                    B = es.eigenvectors();
                    D = es.eigenvalues().asDiagonal();
                    for (decltype(dim) j = 0u; j < dim; ++j ) {
                        D(_(j),_(j)) = std::sqrt( std::max(1e-20, D(_(j),_(j))) );       //D contains standard deviations now
                    }
                    for (decltype(dim) j = 0u; j < dim; ++j ) {
                        Dinv(_(j),_(j)) = 1. / D(_(j),_(j));
                    }
                    invsqrtC = B*Dinv*B.transpose();
                } //if eigendecomposition fails just skip it and keep pevious successful one.
            }
        } // end of generation loop
        return pop;
    }

    /// Sets the algorithm seed
    void set_seed(unsigned int seed)
    {
        m_seed = seed;
    };
    /// Gets the seed
    unsigned int get_seed() const
    {
        return m_seed;
    }
    /// Sets the algorithm verbosity
    void set_verbosity(unsigned int level)
    {
        m_verbosity = level;
    };
    /// Gets the verbosity level
    unsigned int get_verbosity() const
    {
        return m_verbosity;
    }
    /// Get generations
    unsigned int get_gen() const
    {
        return m_gen;
    }
    /// Algorithm name
    std::string get_name() const
    {
        return "CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) +
            "\n\tcc: " + ( (m_cc == -1) ? "auto" : std::to_string(m_cc) ) +
            "\n\tcs: " + ( (m_cs == -1) ? "auto" : std::to_string(m_cs) )+
            "\n\tc1: " + ( (m_c1 == -1) ? "auto" : std::to_string(m_c1) )+
            "\n\tcmu: " + ( (m_cmu == -1) ? "auto" : std::to_string(m_cmu) ) +
            "\n\tsigma0: " + std::to_string(m_sigma0) +
            "\n\tStopping xtol: " + std::to_string(m_xtol) +
            "\n\tStopping ftol: " + std::to_string(m_ftol) +
            "\n\tMemory: " + std::to_string(m_memory) +
            "\n\tVerbosity: " + std::to_string(m_verbosity) +
            "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen,m_cc,m_cs,m_c1,m_cmu,m_sigma0,m_ftol,m_xtol,m_memory,
        sigma, mean, variation, newpop,B,D,C,invsqrtC,pc,ps,counteval,eigeneval,
        m_e,m_seed,m_verbosity);
    }

private:
    // Eigen stores indexes and sizes as signed types, while PaGMO
    // uses STL containers thus sizes and indexes are unsigned. To
    // make the conversion as painless as possible this template is provided
    // allowing, for example, syntax of the type D(_(i),_(j)) to adress an Eigen matrix
    // when i and j are unsigned
    template <typename I>
    static Eigen::DenseIndex _(I n)
    {
        return static_cast<Eigen::DenseIndex>(n);
    }
    // "Real" data members
    unsigned int m_gen;
    double m_cc;
    double m_cs;
    double m_c1;
    double m_cmu;
    double m_sigma0;
    double m_ftol;
    double m_xtol;
    bool m_memory;

    // "Memory" data members (these are adapted during each evolve call and may be remembered if m_memory is true)
    mutable double sigma;
    mutable Eigen::VectorXd mean;
    mutable Eigen::VectorXd variation;
    mutable std::vector<Eigen::VectorXd> newpop;
    mutable Eigen::MatrixXd B;
    mutable Eigen::MatrixXd D;
    mutable Eigen::MatrixXd C;
    mutable Eigen::MatrixXd invsqrtC;
    mutable Eigen::VectorXd pc;
    mutable Eigen::VectorXd ps;
    mutable population::size_type counteval;
    mutable population::size_type eigeneval;

    // "Common" data members
    mutable detail::random_engine_type  m_e;
    unsigned int                        m_seed;
    unsigned int                        m_verbosity;
};

} //namespace pagmo

// PAGMO_REGISTER_ALGORITHM(pagmo::cmaes)

#endif
