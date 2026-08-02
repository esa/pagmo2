/* Copyright 2017-2021 PaGMO development team

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

/*
 *  Implements the NSGA-III multi-objective evolutionary algorithm
 *  as described in http://dx.doi.org/10.1109/TEVC.2013.2281535
 *
 *  Paul Slavin <paul.slavin@manchester.ac.uk>
 */

#ifndef PAGMO_ALGORITHMS_NSGA3_HPP
#define PAGMO_ALGORITHMS_NSGA3_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/detail/visibility.hpp>  // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>  // population
#include <pagmo/rng.hpp>  // random_device, random_engine_type
#include <pagmo/s11n.hpp>  // detail::archive
#include <pagmo/types.hpp>  // vector_double


namespace pagmo{

class PAGMO_DLL_PUBLIC nsga3{
    public:
        // Log line format: (gen, fevals, ideal_point)
        typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
        typedef std::vector<log_line_type> log_type;
        // Defaults from IEEE ToEC Vol.18 Iss.4 Aug, 2014
        nsga3(unsigned gen = 1u, double cr = 1.0, double eta_c = 30.0,
              double mut = 0.10, double eta_mut = 20.0, size_t divisions = 12u,
              unsigned seed = pagmo::random_device::next(), bool use_memory = false);
        std::string get_name() const{ return "NSGA-III:"; }
        std::string get_extra_info() const;
        population evolve(population) const;
        const log_type &get_log() const { return m_log; }
        void set_verbosity(unsigned level) { m_verbosity = level; }
        unsigned get_verbosity() const { return m_verbosity; }
        void set_seed(unsigned seed) { m_reng.seed(seed); m_seed = seed; }
        unsigned get_seed() const { return m_seed; }
    private:
        /*  State retained across generations when memory is enabled.
         *  Both members are expressed in the *original* objective coordinates,
         *  so that they remain meaningful as the ideal point moves.
         */
        struct nsga3_memory{
            std::vector<std::vector<double>> v_extreme;
            std::vector<double> v_ideal;
            template <typename Archive>
            void serialize(Archive &ar, unsigned){
                detail::archive(ar, v_extreme, v_ideal);
            }
        };
        // Survival selection over the combined parent and offspring populations
        std::vector<size_t> selection(population &, size_t) const;

        unsigned m_gen;
        double m_cr;        // crossover
        double m_eta_c;     // eta crossover
        double m_mut;       // mutation
        double m_eta_mut;   // eta mutation
        size_t m_divisions; // Reference Point hyperplane subdivisions
        unsigned m_seed;    // Seed for PRNG initialisation
        bool m_use_memory;  // Preserve extremes and ideal across generations
        mutable nsga3_memory m_memory{};
        mutable detail::random_engine_type m_reng;  // Defaults to std::mt19937
        mutable log_type m_log;
        unsigned m_verbosity {0};
        // Serialisation support
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &, unsigned int);
};

}  // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nsga3)
#endif
