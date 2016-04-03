#include <random>
#include <iostream>

// PaGMO makes use of the 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000.


namespace pagmo
{

namespace details {
	using random_engine_type = std::mt19937_64;

	template <typename dummy>
	struct random_device_statics
	{
		static random_engine_type m_e;
	};

	template<typename dummy>
	random_engine_type random_device_statics<dummy>::m_e{std::random_device{}()};
}

class random_device : public details::random_device_statics<void>
{
public:
	unsigned int operator()() {
		return m_e();
	}
	static void set_seed(unsigned int seed) {
		m_e.seed(seed);
	}

};

} // namespace