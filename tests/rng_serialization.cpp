#define BOOST_TEST_MODULE pagmo_rng_serialization_test
#include <boost/test/unit_test.hpp>

#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "../include/serialization.hpp"

static std::mt19937 rng;

static const int ntrials = 100;

BOOST_AUTO_TEST_CASE(rng_serialization_test)
{
    using r_type = std::mt19937;
    using ia_type = cereal::JSONInputArchive;
    using oa_type = cereal::JSONOutputArchive;
    auto rng_save = [](const r_type &r) {
        std::stringstream ss;
        {
        oa_type oarchive(ss);
        oarchive(r);
        }
        return ss.str();
    };
    auto rng_load = [](const std::string &str, r_type &r) {
        std::stringstream ss;
        ss.str(str);
        {
        ia_type iarchive(ss);
        iarchive(r);
        }
    };
    std::uniform_int_distribution<r_type::result_type> dist;
    for (auto i = 0; i < ntrials; ++i) {
        auto seed = dist(rng);
        r_type r;
        r.seed(seed);
        auto str = rng_save(r);
        std::vector<r_type::result_type> v1;
        std::generate_n(std::back_inserter(v1),100,r);
        auto r_copy(r);
        rng_load(str,r);
        std::vector<r_type::result_type> v2;
        std::generate_n(std::back_inserter(v2),100,r);
        BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(),v1.end(),v2.begin(),v2.end());
        BOOST_CHECK(r_copy == r);
    }
}
