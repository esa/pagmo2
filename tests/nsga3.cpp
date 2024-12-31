#define BOOST_TEST_MODULE nsga3_test
#define BOOST_TEST_DYN_LINK
#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/reference_point.hpp>
#include <pagmo/utils/multi_objective.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(nsga3_instance){
    BOOST_CHECK_NO_THROW(nsga3{});
};

BOOST_AUTO_TEST_CASE(nsga3_evolve_population){
    dtlz udp{1u, 10u, 3u};

    population pop1{udp, 92u, 23u /*seed*/};

    nsga3 user_algo1{10u, 1.0, 30., 0.10, 20., 12u, 32u, false};
    BOOST_CHECK(user_algo1.get_seed() == 32u);
    user_algo1.set_verbosity(10u);
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

    double close_distance = 1e-8;
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
        BOOST_CHECK_CLOSE(p_sum, 1.0, close_distance);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_translate_objectives){
    double tolerance = 1e-6;
    std::vector<double> t_first = {0.92084430016240049, 0.16973405319857038, 290.74413330194784};
    std::vector<double> t_last = {1.7178358364136896, 109.71043974773266, 52.177816158337897};

    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto p0_obj = pop.get_f();
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    size_t t_size = translated_objectives.size();
    for(size_t i=0; i < translated_objectives[0].size(); i++){
        BOOST_CHECK_CLOSE(translated_objectives[0][i], t_first[i], tolerance);
        BOOST_CHECK_CLOSE(translated_objectives[t_size-1][i], t_last[i], tolerance);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_gaussian_elimination){
    // Verify correctness of simple system
    std::vector<std::vector<double>> A(3);
    std::vector<double> b = {1.0, 1.0, 1.0};

    A[0] = {-1, 1,  2};
    A[1] = { 2, 0, -3};
    A[2] = { 5, 1, -2};

    std::vector<double> x = gaussian_elimination(A, b);
    BOOST_CHECK_CLOSE(x[0], -0.4, 1e-8);
    BOOST_CHECK_CLOSE(x[1],  1.8, 1e-8);
    BOOST_CHECK_CLOSE(x[2], -0.6, 1e-8);

    // Verify graceful error on ill-condition
    A[0][0] = 0.0;
    BOOST_CHECK_THROW(gaussian_elimination(A, b), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_extreme_points){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<double> ep_first = {228.71584572959793, 0.92448959747508574, 0.61400521336079161};
    std::vector<double> ep_last = {0.092287013229137627, 0.0, 299.85225007963135};
    double tolerance = 1e-6;

    pop = nsga3_alg.evolve(pop);
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    auto fnds_res = fast_non_dominated_sorting(pop.get_f());
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = nsga3_alg.find_extreme_points(pop, fronts, translated_objectives);
    size_t point_count = ext_points.size();
    size_t point_sz = ext_points[0].size();

    for(size_t idx=0; idx < point_sz; idx++){
        BOOST_CHECK_CLOSE(ext_points[0][idx], ep_first[idx], tolerance);
        BOOST_CHECK_CLOSE(ext_points[point_count-1][idx], ep_last[idx], tolerance);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_intercepts){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<double> intercept_values = {230.13712800033696, 223.90511605342394, 299.97254170821623};
    double tolerance = 1e-6;

    pop = nsga3_alg.evolve(pop);
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    auto fnds_res = fast_non_dominated_sorting(pop.get_f());
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = nsga3_alg.find_extreme_points(pop, fronts, translated_objectives);

    auto intercepts = nsga3_alg.find_intercepts(pop, ext_points);
    for(size_t idx=0; idx < intercepts.size(); idx++){
        BOOST_CHECK_CLOSE(intercepts[idx], intercept_values[idx], tolerance);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_objectives){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<double> norm_first = {0.0040012852691941663, 0.00075806241585865187, 0.96923582287326526};
    std::vector<double> norm_last = {0.0074644011218006267, 0.48998630170449375, 0.173941974359411};
    double tolerance = 1e-6;

    pop = nsga3_alg.evolve(pop);
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    auto fnds_res = fast_non_dominated_sorting(pop.get_f());
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = nsga3_alg.find_extreme_points(pop, fronts, translated_objectives);
    auto intercepts = nsga3_alg.find_intercepts(pop, ext_points);
    auto norm_objs = nsga3_alg.normalize_objectives(translated_objectives, intercepts);
    size_t obj_count = norm_objs.size();
    size_t obj_len = norm_objs[0].size();
    for(size_t idx=0; idx < obj_len; idx++){
        BOOST_CHECK_CLOSE(norm_objs[0][idx], norm_first[idx], tolerance);
        BOOST_CHECK_CLOSE(norm_objs[obj_count-1][idx], norm_last[idx], tolerance);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_serialization_test){
    double close_distance = 1e-8;
    problem prob{zdt{1u, 30u}};
    population pop{prob, 40u, 23u};
    algorithm algo{nsga3{10u, 1.00, 30., 0.10, 20, 5u, 32u, false}};
    algo.set_verbosity(1u);
    algo.set_seed(1234u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<nsga3>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    // Reset the algorithm instance before deserialization
    algo = algorithm{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> algo;
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<nsga3>()->get_log();

    BOOST_CHECK_EQUAL(before_text, after_text);
    BOOST_CHECK(before_log == after_log);
    BOOST_CHECK(before_log.size() > 0u);

    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        for (auto j = 0u; j < 2u; ++j) {
            BOOST_CHECK_CLOSE(std::get<2>(before_log[i])[j], std::get<2>(after_log[i])[j], close_distance);
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_zdt5_test)
{
    algorithm algo{nsga3(100u, 1.00, 30., 0.10, 20., 4u, 32u, false)};
    algo.set_verbosity(10u);
    algo.set_seed(23456u);
    population pop{zdt(5u, 10u), 20u, 32u};
    pop = algo.evolve(pop);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        auto x = pop.get_x()[i];
        BOOST_CHECK(std::all_of(x.begin(), x.end(), [](double el) { return (el == std::floor(el)); }));
    }
}
