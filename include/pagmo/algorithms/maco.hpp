/* Copyright 2017-2020 PaGMO development team

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

#ifndef PAGMO_ALGORITHMS_MACO_HPP
#define PAGMO_ALGORITHMS_MACO_HPP

#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <boost/optional.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>

namespace pagmo
{
// Multi-objective Hypervolume-based Ant Colony Opitmization (MHACO)
class PAGMO_DLL_PUBLIC maco
{
public:
    // Single entry of the log (gen, fevals, ideal_point)
    typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
    // The log
    typedef std::vector<log_line_type> log_type;

    // Constructs the MACO user defined algorithm for multi-objective optimization.
    maco(unsigned gen = 1u, unsigned ker = 63u, double q = 1.0, unsigned threshold = 1u, unsigned n_gen_mark = 7u,
         unsigned evalstop = 100000u, double focus = 0., bool memory = false,
         unsigned seed = pagmo::random_device::next());

    // Algorithm evolve method
    population evolve(population) const;

    // Sets the seed
    void set_seed(unsigned);

    // Gets the seed
    unsigned get_seed() const
    {
        return m_seed;
    }

    // Sets the algorithm verbosity
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    }

    // Gets the verbosity level
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }

    // Gets the generations
    unsigned get_gen() const
    {
        return m_gen;
    }

    // Sets the bfe
    void set_bfe(const bfe &b);

    // Algorithm name
    std::string get_name() const
    {
        return "MHACO: Multi-objective Hypervolume-based Ant Colony Optimization";
    }

    // Extra info
    std::string get_extra_info() const;

    // Get log
    const log_type &get_log() const
    {
        return m_log;
    }

    // Object serialization
    template <typename Archive>
    void serialize(Archive &, unsigned);

private:
    PAGMO_DLL_LOCAL void pheromone_computation(const unsigned gen, vector_double &prob_cumulative,
                                               vector_double &omega_vec, vector_double &sigma_vec,
                                               const population &popul, std::vector<vector_double> &sol_archive) const;
    PAGMO_DLL_LOCAL void generate_new_ants(const population &pop, std::uniform_real_distribution<> dist,
                                           std::normal_distribution<double> gauss_pdf, vector_double prob_cumulative,
                                           vector_double sigma, std::vector<vector_double> &dvs_new,
                                           std::vector<vector_double> &sol_archive) const;

    unsigned m_gen;
    double m_focus;
    unsigned m_ker;
    unsigned m_evalstop;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    unsigned m_threshold;
    mutable double m_q;
    unsigned m_n_gen_mark;
    bool m_memory;
    mutable unsigned m_counter;
    mutable std::vector<vector_double> m_sol_archive;
    mutable unsigned m_n_evalstop;
    mutable unsigned m_gen_mark;
    boost::optional<bfe> m_bfe;
    mutable population m_pop;
};

} // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::maco)

#endif
