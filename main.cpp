#include "include/algorithm.hpp"
#include "include/algorithms/null.hpp"
#include "include/problem.hpp"
#include "include/problems/base.hpp"

using namespace pagmo;

int main()
{
    algorithm a{algorithms::null{}};
    std::stringstream ss;
    {
    cereal::JSONOutputArchive oarchive(ss);
    oarchive(a);
    }
    std::cout << ss.str() << '\n';
    {
    cereal::JSONInputArchive iarchive(ss);
    iarchive(a);
    }    
    std::cout << a.extract<algorithms::null>()->get_a() << std::endl;
    a.evolve();

    problem p{problems::base{}};
    std::cout << p.extract<problems::base>()->objfun(std::vector<double>(3), std::vector<long long>(0))[0] << std::endl;
}
