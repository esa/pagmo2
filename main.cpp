#include "include/algorithm.hpp"
#include "include/algorithms/de.hpp"

using namespace pagmo;

int main()
{
    algorithm a{algorithms::de{}};
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
    std::cout << a.extract<algorithms::de>()->get_a() << std::endl;
}
