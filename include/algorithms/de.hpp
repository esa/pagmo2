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
        m_gen(gen), m_F(F), m_CR(CR), m_strategy(strategy), m_ftol(ftol), m_xtol(xtol), m_seed(seed), m_verbosity(0u), m_log()
    {
        if (strategy < 1u || strategy > 10u) {
            pagmo_throw(std::invalid_argument, "The Differential Evolution strategy must be in [1, .., 10], while a value of " + std::to_string(strategy) + " was detected.");
        }
        if (CR < 0. || F < 0. || CR > 1. || F > 1.) {
            pagmo_throw(std::invalid_argument, "The F and CR parameters must be in the [0,1] range");
        }
    }
private:
    unsigned int        m_gen;
    double              m_F;
    double              m_CR;
    unsigned int        m_strategy;
    double              m_ftol;
    double              m_xtol;
    unsigned int        m_seed;
    unsigned int        m_verbosity;
    mutable log_type    m_log;
};

} //namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::de)

#endif
