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
#include <pagmo/utils/multi_objective.hpp>  // gaussian_elimination

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
    std::cout << "-==: nsga3_test_translate_objectives :==-\n";
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto p0_obj = pop.get_f();
    std::cout << "-==: nsga3_test_translate_objectives  pre translation :==-\n";
    for(size_t i=0; i < p0_obj.size(); i++){
        std::for_each(p0_obj[i].begin(), p0_obj[i].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;
    }
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    std::cout << "-==: nsga3_test_translate_objectives  post translation :==-\n";
    for(size_t i=0; i < translated_objectives.size(); i++){
        std::for_each(translated_objectives[i].begin(), translated_objectives[i].end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_gaussian_elimination){
    // Verify correctness of simple system
    std::vector<std::vector<double>> A(3);
    std::vector<double> b = {1.0, 1.0, 1.0};

    A[0] = {-1, 1, 2};
    A[1] = {2, 0, -3};
    A[2] = {5, 1, -2};

    std::vector<double> x = gaussian_elimination(A, b);
    BOOST_CHECK_CLOSE(x[0], -0.4, 1e-8);
    BOOST_CHECK_CLOSE(x[1],  1.8, 1e-8);
    BOOST_CHECK_CLOSE(x[2], -0.6, 1e-8);
    std::for_each(x.begin(), x.end(), [](const auto& i){std::cout << i << " ";});
    std::cout << std::endl;

    // Verify graceful error on ill-condition
    A[0][0] = 0.0;
    BOOST_CHECK_THROW(gaussian_elimination(A, b), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_extreme_points){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    auto fnds_res = fast_non_dominated_sorting(pop.get_f());
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = nsga3_alg.find_extreme_points(pop, fronts, translated_objectives);

    std::cout << "-==: extreme points :==-\n";
    //std::for_each(ext_points.begin(), ext_points.end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_intercepts){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    auto fnds_res = fast_non_dominated_sorting(pop.get_f());
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = nsga3_alg.find_extreme_points(pop, fronts, translated_objectives);

    auto intercepts = nsga3_alg.find_intercepts(pop, ext_points);
    std::cout << "-==: intercepts :==-\n";
    std::for_each(intercepts.begin(), intercepts.end(), [](const auto& elem){std::cout << elem << " "; });
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_objectives){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};

    pop = nsga3_alg.evolve(pop);
    auto translated_objectives = nsga3_alg.translate_objectives(pop);
    auto fnds_res = fast_non_dominated_sorting(pop.get_f());
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = nsga3_alg.find_extreme_points(pop, fronts, translated_objectives);
    auto intercepts = nsga3_alg.find_intercepts(pop, ext_points);
    auto norm_objs = nsga3_alg.normalize_objectives(translated_objectives, intercepts);
    for(const auto &obj_f: norm_objs){
        std::for_each(obj_f.begin(), obj_f.end(), [](const auto& elem){std::cout << elem << " "; });
        std::cout << std::endl;
    }
}

//BOOST_AUTO_TEST_CASE(nsga3_test_associate_reference_points){
//    nsga3 n{10u, 0.95, 10., 0.01, 50., 32u, false};
//}

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
