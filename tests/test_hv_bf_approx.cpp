#define BOOST_TEST_MODULE test_hv_polar_approx
#include <boost/test/unit_test.hpp>
#include <pagmo/utils/hv_algos/hv_polar_approx.hpp>
#include <cmath>

BOOST_TEST_CASE(test_generate_random_ray)
{
    pagmo::polar_approx algo; 
    unsigned dim = 5;
    auto ray = algo.generate_random_ray(dim);

    BOOST_CHECK(ray.size() == dim);

    double norm_sq = 0;
    for (auto val : ray) {
        norm_sq += val * val;
        BOOST_CHECK(val >= 0); 
    }
    BOOST_CHECK_CLOSE(std::sqrt(norm_sq), 1.0, 1e-9);
}
