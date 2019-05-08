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

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/xnes.hpp>
#include <pagmo/detail/eigen.hpp>
#include <pagmo/detail/eigen_s11n.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

xnes::xnes(unsigned gen, double eta_mu, double eta_sigma, double eta_b, double sigma0, double ftol, double xtol,
           bool memory, bool force_bounds, unsigned seed)
    : m_gen(gen), m_eta_mu(eta_mu), m_eta_sigma(eta_sigma), m_eta_b(eta_b), m_sigma0(sigma0), m_ftol(ftol),
      m_xtol(xtol), m_memory(memory), m_force_bounds(force_bounds), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
{
    if (((eta_mu <= 0.) || (eta_mu > 1.)) && !(eta_mu == -1)) {
        pagmo_throw(std::invalid_argument,
                    "eta_mu must be in ]0,1] or -1 if its value has to be initialized automatically, a value of "
                        + std::to_string(eta_mu) + " was detected");
    }
    if (((eta_sigma <= 0.) || (eta_sigma > 1.)) && !(eta_sigma == -1)) {
        pagmo_throw(std::invalid_argument,
                    "eta_sigma needs to be in ]0,1] or -1 if its value has to be initialized automatically, a value of "
                        + std::to_string(eta_sigma) + " was detected");
    }
    if (((eta_b <= 0.) || (eta_b > 1.)) && !(eta_b == -1)) {
        pagmo_throw(std::invalid_argument,
                    "eta_b needs to be in ]0,1] or -1 if its value has to be initialized automatically, a value of "
                        + std::to_string(eta_b) + " was detected");
    }
    if (((sigma0 <= 0.) || (sigma0 > 1.)) && !(sigma0 == -1)) {
        pagmo_throw(std::invalid_argument,
                    "sigma0 needs to be in ]0,1] or -1 if its value has to be initialized automatically, a value of "
                        + std::to_string(sigma0) + " was detected");
    }
    // Initialize explicitly the algorithm memory
    sigma = m_sigma0;
    mean = Eigen::VectorXd::Zero(1);
    A = Eigen::MatrixXd::Identity(1, 1);
}

/// Algorithm evolve method
/**
 *
 * Evolves the population for a maximum number of generations, until one of
 * tolerances set on the population flatness (x_tol, f_tol) are met.
 *
 * @param pop population to be evolved
 * @return evolved population
 * @throws std::invalid_argument if the problem is multi-objective or constrained
 * @throws std::invalid_argument if the problem is unbounded
 * @throws std::invalid_argument if the population size is not at least 4
 */
population xnes::evolve(population pop) const
{
    // We store some useful variables
    const auto &prob = pop.get_problem(); // This is a const reference, so using set_seed for example will not be
                                          // allowed.
    auto dim = prob.get_nx();
    const auto bounds = prob.get_bounds();
    const auto &lb = bounds.first;
    const auto &ub = bounds.second;
    auto lam = pop.size();
    auto prob_f_dimension = prob.get_nf();
    auto fevals0 = prob.get_fevals(); // discount for the already made fevals
    auto count = 1u;                  // regulates the screen output

    // PREAMBLE--------------------------------------------------
    // Checks on the problem type
    if (prob.get_nc() != 0u) {
        pagmo_throw(std::invalid_argument, "Non linear constraints detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (prob_f_dimension != 1u) {
        pagmo_throw(std::invalid_argument, "Multiple objectives detected in " + prob.get_name() + " instance. "
                                               + get_name() + " cannot deal with them");
    }
    if (lam < 4u) {
        pagmo_throw(std::invalid_argument, get_name() + " needs at least 5 individuals in the population, "
                                               + std::to_string(lam) + " detected");
    }
    for (auto num : lb) {
        if (!std::isfinite(num)) {
            pagmo_throw(std::invalid_argument, "A " + std::to_string(num) + " is detected in the lower bounds, "
                                                   + get_name() + " cannot deal with it.");
        }
    }
    for (auto num : ub) {
        if (!std::isfinite(num)) {
            pagmo_throw(std::invalid_argument, "A " + std::to_string(num) + " is detected in the upper bounds, "
                                                   + get_name() + " cannot deal with it.");
        }
    }
    // Get out if there is nothing to do.
    if (m_gen == 0u) {
        return pop;
    }
    // -----------------------------------------------------------

    // No throws, all valid: we clear the logs
    m_log.clear();

    // -------------------------------------------------------//
    // HERE WE PREPARE AND DEFINE VARIOUS PARAMETERS          //
    // -------------------------------------------------------//
    // Initializing the random number generators
    std::uniform_real_distribution<double> randomly_distributed_number(0., 1.); // to generate a number in [0, 1)
    std::normal_distribution<double> normally_distributed_number(0., 1.);
    // Initialize default values for the learning rates
    double dim_d = static_cast<double>(dim);
    double lam_d = static_cast<double>(lam);

    double eta_mu(m_eta_mu), eta_sigma(m_eta_sigma), eta_b(m_eta_b);
    if (eta_mu == -1) {
        eta_mu = 1.;
    }
    double common_default = 0.6 * (3. + std::log(dim_d)) / (dim_d * std::sqrt(dim_d));
    if (eta_sigma == -1) {
        eta_sigma = common_default;
    }
    if (eta_b == -1) {
        eta_b = common_default;
    }
    // Initialize the utility function u
    std::vector<double> u(lam);
    for (decltype(u.size()) i = 0u; i < u.size(); ++i) {
        u[i] = std::max(0., std::log(lam_d / 2. + 1.) - std::log(i + 1));
    }
    double sum = 0.;
    for (decltype(u.size()) i = 0u; i < u.size(); ++i) {
        sum += u[i];
    }
    for (decltype(u.size()) i = 0u; i < u.size(); ++i) {
        u[i] = u[i] / sum - 1. / lam_d; // Give an option to turn off the unifrm baseline (i.e. -1/lam_d) ?
    }
    // If m_memory is false we redefine mutable members erasing the memory of past calls.
    // This is also done if the problem dimension has changed
    if ((mean.size() != _(dim)) || (m_memory == false)) {
        if (m_sigma0 == -1) {
            sigma = 0.5;
        } else {
            sigma = m_sigma0;
        }
        A = Eigen::MatrixXd::Identity(_(dim), _(dim));
        // The diagonal of the initial covariance matrix A defines the search width in all directions.
        // By default we set this to be sigma times the witdh of the box bounds or 1e-6 if too small.
        for (decltype(dim) j = 0u; j < dim; ++j) {
            A(_(j), _(j)) = std::max((ub[j] - lb[j]), 1e-6) * sigma;
        }
        mean.resize(_(dim));
        auto idx_b = pop.best_idx();
        for (decltype(dim) i = 0u; i < dim; ++i) {
            mean(_(i)) = pop.get_x()[idx_b][i];
        }
    }
    // This will hold in the eigen data structure the sampled population
    Eigen::VectorXd tmp = Eigen::VectorXd::Zero(_(dim));
    auto z = std::vector<Eigen::VectorXd>(lam, tmp);
    auto x = std::vector<Eigen::VectorXd>(lam, tmp);
    // Temporary container
    vector_double dumb(dim, 0.);

    if (m_verbosity > 0u) {
        std::cout << "xNES 4 PaGMO: " << std::endl;
        print("eta_mu: ", eta_mu, " - eta_sigma: ", eta_sigma, " - eta_b: ", eta_b, " - sigma0: ", sigma, "\n");
        print("utilities: ", u, "\n");
    }

    // ----------------------------------------------//
    // HERE WE START THE JUICE OF THE ALGORITHM      //
    // ----------------------------------------------//
    for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
        // 0 -If the problem is stochastic change seed first
        if (prob.is_stochastic()) {
            // change the problem seed. This is done via the population_set_seed method as prob.set_seed
            // is forbidden being prob a const ref.
            pop.get_problem().set_seed(std::uniform_int_distribution<unsigned>()(m_e));
        }
        // 1 - We generate lam new individuals using the current probability distribution
        for (decltype(lam) i = 0u; i < lam; ++i) {
            // 1a - we create a randomly normal distributed vector
            for (decltype(dim) j = 0u; j < dim; ++j) {
                z[i](_(j)) = normally_distributed_number(m_e);
            }
            // 1b - and store its transformed value in the new chromosomes
            x[i] = mean + A * z[i];
            if (m_force_bounds) {
                // We fix the bounds. Note that this screws up the whole covariance matrix machinery and worsen
                // performances considerably.
                for (decltype(dim) j = 0u; j < dim; ++j) {
                    if (x[i](_(j)) < lb[j]) {
                        x[i](_(j)) = lb[j];
                    } else if (x[i](_(j)) > ub[j]) {
                        x[i](_(j)) = ub[j];
                    }
                }
            }
            for (decltype(dim) j = 0u; j < dim; ++j) {
                dumb[j] = x[i](_(j));
            }
            pop.set_x(i, dumb);
        }

        // 2 - Check the exit conditions and logs
        // Exit condition on xtol
        {
            if ((A * z[0]).norm() < m_xtol) {
                if (m_verbosity > 0u) {
                    std::cout << "Exit condition -- xtol < " << m_xtol << std::endl;
                }
                return pop;
            }
            // Exit condition on ftol
            auto idx_b = pop.best_idx();
            auto idx_w = pop.worst_idx();
            double delta_f = std::abs(pop.get_f()[idx_b][0] - pop.get_f()[idx_w][0]);
            if (delta_f < m_ftol) {
                if (m_verbosity) {
                    std::cout << "Exit condition -- ftol < " << m_ftol << std::endl;
                }
                return pop;
            }
        }

        // 2bis - Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
        if (m_verbosity > 0u) {
            // Every m_verbosity generations print a log line
            if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                // The population flattness in chromosome
                auto dx = (A * z[0]).norm();
                // The population flattness in fitness
                auto idx_b = pop.best_idx();
                auto idx_w = pop.worst_idx();
                auto df = std::abs(pop.get_f()[idx_b][0] - pop.get_f()[idx_w][0]);
                // Every 50 lines print the column names
                if (count % 50u == 1u) {
                    print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15), "Best:", std::setw(15),
                          "dx:", std::setw(15), "df:", std::setw(15), "sigma:", '\n');
                }
                print(std::setw(7), gen, std::setw(15), prob.get_fevals() - fevals0, std::setw(15),
                      pop.get_f()[idx_b][0], std::setw(15), dx, std::setw(15), df, std::setw(15), sigma, '\n');
                ++count;
                // Logs
                m_log.emplace_back(gen, prob.get_fevals() - fevals0, pop.get_f()[idx_b][0], dx, df, sigma);
            }
        }

        // 3 - We sort the population
        std::vector<vector_double::size_type> s_idx(lam);
        std::iota(s_idx.begin(), s_idx.end(), vector_double::size_type(0u));
        std::sort(s_idx.begin(), s_idx.end(), [&pop](vector_double::size_type a, vector_double::size_type b) {
            return pop.get_f()[a][0] < pop.get_f()[b][0];
        });
        // 4 - We update the distribution parameters mu, sigma and B following the xnes rules
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(_(dim), _(dim));
        Eigen::VectorXd d_center = u[0] * z[s_idx[0]];
        for (decltype(u.size()) i = 1u; i < u.size(); ++i) {
            d_center += u[i] * z[s_idx[i]];
        }
        Eigen::MatrixXd cov_grad = u[0] * (z[s_idx[0]] * z[s_idx[0]].transpose() - I);
        for (decltype(u.size()) i = 1u; i < u.size(); ++i) {
            cov_grad += u[i] * (z[s_idx[i]] * z[s_idx[i]].transpose() - I);
        }
        double cov_trace = cov_grad.trace();
        cov_grad = cov_grad - cov_trace / dim_d * I;
        Eigen::MatrixXd d_A = 0.5 * (eta_sigma * cov_trace / dim_d * I + eta_b * cov_grad);
        mean = mean + eta_mu * A * d_center;
        A = A * d_A.exp();
        sigma = sigma * std::exp(eta_sigma / 2. * cov_trace / dim_d); // used only for cmaes comparisons
    }
    if (m_verbosity) {
        std::cout << "Exit condition -- generations = " << m_gen << std::endl;
    }
    return pop;
}

/// Sets the seed
/**
 * @param seed the seed controlling the algorithm stochastic behaviour
 */
void xnes::set_seed(unsigned seed)
{
    m_e.seed(seed);
    m_seed = seed;
}

/// Extra info
/**
 * One of the optional methods of any user-defined algorithm (UDA).
 *
 * @return a string containing extra info on the algorithm
 */
std::string xnes::get_extra_info() const
{
    std::ostringstream ss;
    stream(ss, "\tGenerations: ", m_gen);
    stream(ss, "\n\teta_mu: ");
    if (m_eta_mu == -1) {
        stream(ss, "auto");
    } else {
        stream(ss, m_eta_mu);
    }
    stream(ss, "\n\teta_sigma: ");
    if (m_eta_sigma == -1) {
        stream(ss, "auto");
    } else {
        stream(ss, m_eta_sigma);
    }
    stream(ss, "\n\teta_b: ");
    if (m_eta_b == -1) {
        stream(ss, "auto");
    } else {
        stream(ss, m_eta_b);
    }
    stream(ss, "\n\tsigma0: ");
    if (m_sigma0 == -1) {
        stream(ss, "auto");
    } else {
        stream(ss, m_sigma0);
    }
    stream(ss, "\n\tStopping xtol: ", m_xtol);
    stream(ss, "\n\tStopping ftol: ", m_ftol);
    stream(ss, "\n\tMemory: ", m_memory);
    stream(ss, "\n\tForce bounds: ", m_force_bounds);
    stream(ss, "\n\tVerbosity: ", m_verbosity);
    stream(ss, "\n\tSeed: ", m_seed);
    return ss.str();
}

/// Object serialization
/**
 * This method will save/load \p this into the archive \p ar.
 *
 * @param ar target archive.
 *
 * @throws unspecified any exception thrown by the serialization of primitive types.
 */
template <typename Archive>
void xnes::serialize(Archive &ar, unsigned)
{
    detail::archive(ar, m_gen, m_eta_mu, m_eta_sigma, m_eta_b, m_sigma0, m_ftol, m_xtol, m_memory, m_force_bounds,
                    sigma, mean, A, m_e, m_seed, m_verbosity, m_log);
}

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pagmo::xnes)
