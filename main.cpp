#include "include/algorithm.hpp"
#include "include/algorithms/null.hpp"

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

}
