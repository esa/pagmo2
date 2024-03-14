#ifndef PAGMO_ALGORITHMS_NSGA3_HPP
#define PAGMO_ALGORITHMS_NSGA3_HPP

#include <string>
#include <tuple>
#include <vector>

#include <pagmo/rng.hpp>  // random_device, random_engine_type
#include <pagmo/detail/visibility.hpp>  // PAGMO_DLL_PUBLIC
#include <pagmo/population.hpp>  // population
#include <pagmo/utils/reference_point.hpp>  // ReferencePoint


namespace pagmo{

class PAGMO_DLL_PUBLIC nsga3{
    public:
        typedef struct{
            std::vector<std::vector<double>> v_extreme;
            std::vector<double> v_ideal;
            std::vector<double> v_nadir;
        } NSGA3Memory;
        // Log line format: (gen, fevals, ideal_point)
        typedef std::tuple<unsigned, unsigned long long, vector_double> log_line_type;
        typedef std::vector<log_line_type> log_type;
        // Defaults from IEEE ToEC Vol.18 Iss.4 Aug, 2014
        nsga3(unsigned gen = 1u, double cr = 1.0, double eta_c = 30.0,
              double mut = 0.10, double eta_mut = 20.0, size_t divisions = 12u,
              unsigned seed = pagmo::random_device::next(), bool use_memory = false);
        std::string get_name() const{ return "NSGA-III:"; }
        population evolve(population) const;
        std::vector<size_t> selection(population &, size_t) const;
        std::vector<ReferencePoint> generate_uniform_reference_points(size_t nobjs, size_t divisions) const;
        std::vector<std::vector<double>> translate_objectives(population) const;
        std::vector<std::vector<double>> find_extreme_points(population, std::vector<std::vector<pop_size_t>> &, std::vector<std::vector<double>> &) const;
        std::vector<double> find_intercepts(population, std::vector<std::vector<double>> &) const;
        std::vector<std::vector<double>> normalize_objectives(std::vector<std::vector<double>> &, std::vector<double> &) const;
        const log_type &get_log() const { return m_log; }
        void set_verbosity(unsigned level) { m_verbosity = level; }
        unsigned get_verbosity() const { return m_verbosity; }
        void set_seed(unsigned seed) { reng.seed(seed); seed = seed; }
        unsigned get_seed() const { return seed; }
        bool has_memory() const {return use_memory; }
    private:
        unsigned ngen;
        double cr;        // crossover
        double eta_c;     // eta crossover
        double mut;       // mutation
        double eta_mut;   // eta mutation
        size_t divisions; // Reference Point hyperplane subdivisions
        unsigned seed;    // Seed for PRNG initialisation
        bool use_memory;  // Preserve extremes, ideal, nadir across generations
        mutable NSGA3Memory memory{};
        mutable detail::random_engine_type reng;  // Defaults to std::mt19937
        mutable log_type m_log;
        unsigned m_verbosity;
        mutable std::vector<ReferencePoint> refpoints;
        // Serialisation support
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize(Archive &, unsigned int);
};

}  // namespace pagmo

PAGMO_S11N_ALGORITHM_EXPORT_KEY(pagmo::nsga3)
#endif
