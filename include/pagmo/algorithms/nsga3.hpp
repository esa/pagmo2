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
        // Defaults from IEEE ToEC Vol.18 Iss.4 Aug, 2014
        nsga3(unsigned gen = 1u, double cr = 1.0,
              double eta_c = 30.0, double m = 0.01,
              double eta_m = 20.0, unsigned seed = pagmo::random_device::next());
        population evolve(population) const;
        std::vector<ReferencePoint> generate_uniform_reference_points(size_t nobjs, size_t divisions);
        std::vector<std::vector<double>> translate_objectives(population);
        std::vector<size_t> extreme_points(population, std::vector<std::vector<pop_size_t>>);
    private:
        unsigned ngen;
        double cr;      // crossover
        double eta_c;   // eta crossover
        double m;       // mutation
        double eta_m;   // eta mutation
        unsigned seed;  // Seed for PRNG initialisation
        mutable detail::random_engine_type reng;  // Defaults to std::mt19937
        std::vector<ReferencePoint> refpoints;
};

}  // namespace pagmo


#endif
