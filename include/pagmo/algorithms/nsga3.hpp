#ifndef PAGMO_ALGORITHMS_NSGA3_HPP
#define PAGMO_ALGORITHMS_NSGA3_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/rng.hpp>  // random_device, random_engine_type
#include <pagmo/detail/visibility.hpp>  // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>  // population
#include <pagmo/s11n.hpp>  // detail::archive
#include <pagmo/utils/reference_point.hpp>  // ReferencePoint


namespace pagmo{

class PAGMO_DLL_PUBLIC nsga3{
    public:
        /*  State retained across generations when memory is enabled.
         *  Both members are expressed in the *original* objective coordinates,
         *  so that they remain meaningful as the ideal point moves.
         */
        struct NSGA3Memory{
            std::vector<std::vector<double>> v_extreme;
            std::vector<double> v_ideal;
            template <typename Archive>
            void serialize(Archive &ar, unsigned){
                detail::archive(ar, v_extreme, v_ideal);
            }
        };
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
        std::vector<size_t> selection(population &, size_t) const;
        std::vector<ReferencePoint> generate_uniform_reference_points(size_t nobjs, size_t divisions) const;
        /*  The normalisation helpers below are expressed in terms of objective vectors
         *  and an explicit ideal point rather than a population, so that the coordinate
         *  system in use is never implicit. They are public to allow direct testing.
         */
        std::vector<double> compute_ideal(const std::vector<vector_double> &) const;
        std::vector<std::vector<double>> translate_objectives(const std::vector<vector_double> &,
                                                              const std::vector<double> &) const;
        std::vector<std::vector<double>> find_extreme_points(const std::vector<std::vector<pop_size_t>> &,
                                                             const std::vector<std::vector<double>> &,
                                                             const std::vector<double> &) const;
        std::vector<double> find_intercepts(const std::vector<std::vector<double>> &,
                                            const std::vector<std::vector<double>> &) const;
        std::vector<std::vector<double>> normalize_objectives(const std::vector<std::vector<double>> &,
                                                              const std::vector<double> &) const;
        const log_type &get_log() const { return m_log; }
        void set_verbosity(unsigned level) { m_verbosity = level; }
        unsigned get_verbosity() const { return m_verbosity; }
        void set_seed(unsigned seed) { m_reng.seed(seed); m_seed = seed; }
        unsigned get_seed() const { return m_seed; }
        bool has_memory() const {return m_use_memory; }
    private:
        unsigned m_gen;
        double m_cr;        // crossover
        double m_eta_c;     // eta crossover
        double m_mut;       // mutation
        double m_eta_mut;   // eta mutation
        size_t m_divisions; // Reference Point hyperplane subdivisions
        unsigned m_seed;    // Seed for PRNG initialisation
        bool m_use_memory;  // Preserve extremes and ideal across generations
        mutable NSGA3Memory m_memory{};
        mutable detail::random_engine_type m_reng;  // Defaults to std::mt19937
        mutable log_type m_log;
        unsigned m_verbosity {0};
        mutable std::vector<ReferencePoint> m_refpoints;
        // Serialisation support
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &, unsigned int);
};

}  // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nsga3)
#endif
