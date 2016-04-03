#include <random>
#include <iostream>

// PaGMO makes use of the 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000.


namespace pagmo
{

namespace details {
	using random_engine_type = std::mt19937;

	template <typename dummy>
	struct random_device_statics
	{
		static random_engine_type m_e;
		static  std::mutex  m_mutex;
	};

	template<typename dummy>
	random_engine_type random_device_statics<dummy>::m_e{std::random_device{}()};

	template<typename dummy>
	std::mutex random_device_statics<dummy>::m_mutex{std::mutex{}};
}

/// This intends to be a thread-safe substitute for std::random_device, allowing for precise
/// global seed control in PaGMO. All classes that contain a random engine (thus that generate
/// random numbers from variates), by default should contain something like:
//#include "rng.hpp"

///class class_using_random {
///    explicit class_using_random(args ...... , unsigned int seed = pagmo::random_device::next()) : m_e(seed), m_seed(seed);
///private:
///    // Random engine
///    random_engine_type               m_e;
///
///    unsigned int 					m_seed;
///}


class random_device : public details::random_device_statics<void>
{
public:
	static unsigned int next() {
		std::lock_guard<std::mutex> lock(m_mutex);
		return static_cast<unsigned int>(m_e());
	}
	static void set_seed(unsigned int seed) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_e.seed(static_cast<random_engine_type::result_type>(seed));
	}

};

} // namespace