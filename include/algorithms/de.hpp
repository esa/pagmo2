#ifndef PAGMO_ALGORITHMS_DE_HPP
#define PAGMO_ALGORITHMS_DE_HPP

#include <iomanip>
#include <random>
#include <string>
#include <tuple>

#include "../io.hpp"
#include "../exceptions.hpp"
#include "../population.hpp"
#include "../rng.hpp"

namespace pagmo
{
class de
{
public:
    de(unsigned int gen = 1u, double F = 0.8, double CR = 0.2, unsigned int strategy = 2, double ftol = 1e-6, double xtol = 1e-6, unsigned int seed = pagmo::random_device::next()) :
        m_gen(gen), m_F(F), m_CR(CR), m_strategy(strategy), m_ftol(ftol), m_xtol(xtol), m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
        if (strategy < 1u || strategy > 10u) {
            pagmo_throw(std::invalid_argument, "The Differential Evolution strategy must be in [1, .., 10], while a value of " + std::to_string(strategy) + " was detected.");
        }
        if (CR < 0. || F < 0. || CR > 1. || F > 1.) {
            pagmo_throw(std::invalid_argument, "The F and CR parameters must be in the [0,1] range");
        }
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
        return "Differential Evolution";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) +
            "\n\tParameter F: " + std::to_string(m_F) +
            "\n\tParameter CR: " + std::to_string(m_CR) +
            "\n\tStrategy: " + std::to_string(m_strategy) +
            "\n\tStopping xtol: " + std::to_string(m_xtol) +
            "\n\tStopping ftol: " + std::to_string(m_ftol) +
            "\n\tVerbosity: " + std::to_string(m_verbosity) +
            "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Get log
    const log_type& get_log() const {
        return m_log;
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen,m_F,m_CR,m_strategy,m_ftol,m_xtol,m_e,m_seed,m_verbosity,m_log);
    }
private:
    unsigned int                        m_gen;
    double                              m_F;
    double                              m_CR;
    unsigned int                        m_strategy;
    double                              m_ftol;
    double                              m_xtol;
    mutable detail::random_engine_type  m_e;
    unsigned int                        m_seed;
    unsigned int                        m_verbosity;
    mutable log_type                    m_log;
};

} //namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::de)

#endif
