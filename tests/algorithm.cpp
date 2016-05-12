#define BOOST_TEST_MODULE pagmo_problem_test
#include <boost/test/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <sstream>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "../include/algorithm.hpp"
#include "../include/problems/rosenbrock.hpp"
#include "../include/population.hpp"
#include "../include/serialization.hpp"
#include "../include/types.hpp"

using namespace pagmo;

struct al_01 
{
	al_01() {};
	population evolve(const population& pop) const {return pop;};
	std::string get_name() const {return "name";};
	std::string get_extra_info() const {return "\tSeed: " + std::to_string(m_seed) + "\n\tVerbosity: " + std::to_string(m_verbosity);};
	void set_seed(unsigned int seed) {m_seed = seed;};
	void set_verbosity(unsigned int level) {m_verbosity = level;};
	unsigned int m_seed = 0u;
	unsigned int m_verbosity = 0u;
};

BOOST_AUTO_TEST_CASE(algorithm_construction_test)
{
    algorithm algo{al_01{}};
    print(has_set_seed<al_01>::value, '\n');
    problem prob{rosenbrock{5}};
    population pop{prob, 10u};
    auto pop2 = algo.evolve(pop);
    algo.set_seed(32u);
    print(algo,'\n');
    algo.set_verbosity(1u);
    print(algo);
}
