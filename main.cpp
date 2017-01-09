#include "include/algorithm.hpp"
#include "include/algorithms/null.hpp"
#include "include/problem.hpp"
#include "include/problems/hock_schittkowsky_71.hpp"

using namespace pagmo;
int main()
{
    //     {
    //     algorithm a{algorithms::null{}};
    //     std::stringstream ss;
    //     {
    //     cereal::JSONOutputArchive oarchive(ss);
    //     oarchive(a);
    //     }
    //     std::cout << ss.str() << '\n';
    //     {
    //     cereal::JSONInputArchive iarchive(ss);
    //     iarchive(a);
    //     }
    //     a.evolve();
    //     }

    problem p{hock_schittkowsky_71{}};
    std::stringstream ss;
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(p);
    }
    std::cout << ss.str() << '\n';
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(p);
    }
}
