#ifndef PAGMO_ALGORITHMS_ABC_HPP
#define PAGMO_ALGORITHMS_ABC_HPP

#include <random>
#include <string>

#include "../algorithm.hpp"
#include "../exceptions.hpp"
#include "../io.hpp"
#include "../population.hpp"
#include "../rng.hpp"
#include "../utils/generic.hpp"

namespace pagmo
{
/// Artificial Bee Colony Algorithm
/**
 *
 */
class abc
{
public:
#if defined(DOXYGEN_INVOKED)
    /// Single entry of the log (gen, fevals, best, dx, df)
    typedef std::tuple<unsigned int, unsigned long long, double, double, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;
#else
    using log_line_type = std::tuple<unsigned int, unsigned long long, double, double, double>;
    using log_type = std::vector<log_line_type>;
#endif

    /// Constructor.
    /**
     * Constructs an abc algorithm
     */
    abc(unsigned int seed = pagmo::random_device::next()) : m_e(seed), m_seed(seed), m_verbosity(0u), m_log()
    {
    }

    /// Algorithm evolve method
    /**
     * Evolves the population
     */
    population evolve(population pop) const
    {
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
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 100):
     * @code
     * @endcode
     *
     * @param level verbosity level
     */
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
        return "Artificial Bee Colony";
    }
    /// Extra informations
    std::string get_extra_info() const
    {
        return "\tGenerations: " + std::to_string(m_gen) + "\n\tVerbosity: " + std::to_string(m_verbosity)
               + "\n\tSeed: " + std::to_string(m_seed);
    }
    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve.
     */
    const log_type &get_log() const
    {
        return m_log;
    }
    /// Serialization
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(m_gen, m_e, m_seed, m_verbosity, m_log);
    }

private:
    unsigned int m_gen;
    mutable detail::random_engine_type m_e;
    unsigned int m_seed;
    unsigned int m_verbosity;
    mutable log_type m_log;
};

} // namespace pagmo

PAGMO_REGISTER_ALGORITHM(pagmo::abc)

#endif
