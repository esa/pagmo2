#ifndef PAGMO_RNG_HPP
#define PAGMO_RNG_HPP

#include <mutex>
#include <random>

namespace pagmo
{
namespace details {

// PaGMO makes use of the 32-bit Mersenne Twister by Matsumoto and Nishimura, 1998.
using random_engine_type = std::mt19937;

template <typename dummy>
struct random_device_statics
{
    static random_engine_type m_e;
    static std::mutex  m_mutex;
};

template<typename dummy>
random_engine_type random_device_statics<dummy>::m_e{static_cast<random_engine_type::result_type>(std::random_device{}())};

template<typename dummy>
std::mutex random_device_statics<dummy>::m_mutex{};

} // end namespace details

/// Thread-safe random device
/**
 * This intends to be a thread-safe substitute for std::random_device, allowing for precise
 * global seed control in PaGMO. All classes that contain a random engine (thus that generate
 * random numbers from variates), by default should contain something like:
 * @code
 * #include "rng.hpp"
 * class class_using_random {
 * explicit class_using_random(args ...... , unsigned int seed = pagmo::random_device::next()) : m_e(seed), m_seed(seed);
 * private:
 *    // Random engine
 *    random_engine_type               m_e;
 *    // Seed
 *    unsigned int                     m_seed;
 * }
 * @endcode
 */
class random_device : public details::random_device_statics<void>
{
public:
    /// Describe!
    /**
     * Blah blah blah.
     * 
     * @returns dsdlsadslad
     */
    static unsigned int next() 
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return static_cast<unsigned int>(m_e());
    }
    /// Describe!
    /**
     * Blah blah blah.
     * 
     * @param[in] seed boo baa
     */
    static void set_seed(unsigned int seed) 
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_e.seed(static_cast<details::random_engine_type::result_type>(seed));
    }

};

} // end namespace pagmo

#endif
