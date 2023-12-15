#define BOOST_TEST_MODULE nsga3_test
#define BOOST_TEST_DYN_LINK
#include <iostream>

#include <boost/test/unit_test.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/reference_point.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nsga3_instance){
    BOOST_CHECK_NO_THROW(nsga3{});
};

BOOST_AUTO_TEST_CASE(nsga3_evolve_population){
    dtlz udp{1u, 10u, 3u};

    population pop1{udp, 52u, 23u};

    nsga3 user_algo1{10u, 0.95, 10., 0.01, 50., 32u};
    pop1 = user_algo1.evolve(pop1);
};

BOOST_AUTO_TEST_CASE(nsga3_reference_point_type){
    ReferencePoint rp3(3);
    BOOST_CHECK_EQUAL(rp3.dim(), 3);
    BOOST_CHECK_EQUAL(rp3[0], 0.0);
    BOOST_CHECK_EQUAL(rp3[1], 0.0);
    BOOST_CHECK_EQUAL(rp3[2], 0.0);
}

BOOST_AUTO_TEST_CASE(nsga3_verify_uniform_reference_points){
    /*  1. Verify cardinality of ref point set
     *  2. Verify coefficients sum to 1.0
     */

    nsga3 n = nsga3();
    auto rp_3_12 = n.generate_uniform_reference_points(3, 12);
    BOOST_CHECK_EQUAL(rp_3_12.size(), 91);
    for(auto& p: rp_3_12){
        double p_sum = 0.0;
        for(size_t idx=0; idx<p.dim(); idx++){
            p_sum += p[idx];
        }
        BOOST_CHECK_CLOSE(p_sum, 1.0, 1e-8);
    }

    auto rp_8_12 = n.generate_uniform_reference_points(8, 12);
    BOOST_CHECK_EQUAL(rp_8_12.size(), 50388);
    for(auto& p: rp_8_12){
        double p_sum = 0.0;
        for(size_t idx=0; idx<p.dim(); idx++){
            p_sum += p[idx];
        }
        BOOST_CHECK_CLOSE(p_sum, 1.0, 1e-8);
    }
}
