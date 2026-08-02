#define BOOST_TEST_MODULE nsga3_test
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>

#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nsga3.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/detail/nsga3_impl.hpp>
#include <pagmo/detail/reference_point.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/multi_objective.hpp>

using namespace pagmo;


BOOST_AUTO_TEST_CASE(nsga3_algorithm_construction)
{
    BOOST_CHECK_NO_THROW(nsga3{});
    nsga3 user_algo{1u, 1.00, 30.0, 0.10, 20.0, 12u, 32u, false};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 32u);
    BOOST_CHECK(user_algo.get_extra_info().find("Seed: 32") != std::string::npos);

    // Verify throw on invalid arguments
    // Invalid cr
    BOOST_CHECK_THROW((nsga3{1u, 2.00, 30.0, 0.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, -1.00, 30.0, 0.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    // Invalid mut
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 1.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, -0.10, 20.0, 12u, 32u, false}), std::invalid_argument);
    // Invalid eta_mut
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, 100.1, 12u, 32u, false}), std::invalid_argument);
    BOOST_CHECK_THROW((nsga3{1u, 1.00, 30.0, 0.10, -0.1, 12u, 32u, false}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_evolve_population){
    dtlz udp{1u, 10u, 3u};

    population pop1{udp, 92u, 23u /*seed*/};

    nsga3 user_algo1{10u, 1.0, 30., 0.10, 20., 12u, 32u, false};
    BOOST_CHECK(user_algo1.get_seed() == 32u);
    user_algo1.set_verbosity(10u);
    pop1 = user_algo1.evolve(pop1);
};

BOOST_AUTO_TEST_CASE(nsga3_reference_point_type){
    detail::reference_point rp3(3);
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
    auto rp_3_12 = detail::generate_uniform_reference_points(3, 12);
    BOOST_CHECK_EQUAL(rp_3_12.size(), 91);
    for(auto& p: rp_3_12){
        double p_sum = 0.0;
        for(size_t idx=0; idx<p.dim(); idx++){
            p_sum += p[idx];
        }
        BOOST_CHECK_CLOSE(p_sum, 1.0, 1e-8);
    }

    auto rp_8_12 = detail::generate_uniform_reference_points(8, 12);
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
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<std::vector<double>> translated_objectives{};

    pop = nsga3_alg.evolve(pop);
    auto p0_obj = pop.get_f();
    auto ideal_point = detail::nsga3_compute_ideal(p0_obj, nullptr);
    BOOST_CHECK_NO_THROW((translated_objectives = detail::nsga3_translate_objectives(p0_obj, ideal_point)));
    size_t t_size = translated_objectives.size();
    BOOST_CHECK_EQUAL(t_size, p0_obj.size());
}

BOOST_AUTO_TEST_CASE(nsga3_test_gaussian_elimination){
    // Verify correctness of simple system
    std::vector<std::vector<double>> A(3);
    std::vector<double> b = {1.0, 1.0, 1.0};

    A[0] = {-1, 1,  2};
    A[1] = { 2, 0, -3};
    A[2] = { 5, 1, -2};

    auto x = detail::gaussian_elimination(A, b);
    BOOST_REQUIRE(x.has_value());
    BOOST_CHECK_CLOSE((*x)[0], -0.4, 1e-8);
    BOOST_CHECK_CLOSE((*x)[1],  1.8, 1e-8);
    BOOST_CHECK_CLOSE((*x)[2], -0.6, 1e-8);

    /*  A zero leading pivot is not an error: partial pivoting selects the largest
     *  available pivot in each column, so this non-singular system is solvable.
     */
    std::vector<std::vector<double>> pivoted{{0.0, 2.0, 1.0}, {1.0, 0.0, 3.0}, {2.0, 1.0, 0.0}};
    auto xp = detail::gaussian_elimination(pivoted, b);
    BOOST_REQUIRE(xp.has_value());
    for(size_t i=0; i<pivoted.size(); i++){
        double residual = 0.0;
        for(size_t j=0; j<pivoted[i].size(); j++){
            residual += pivoted[i][j]*(*xp)[j];
        }
        BOOST_CHECK_CLOSE(residual, b[i], 1e-8);
    }

    // An exactly singular system is reported, not thrown: the third row is row0 + row1
    std::vector<std::vector<double>> singular{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {5.0, 7.0, 9.0}};
    BOOST_CHECK_NO_THROW((detail::gaussian_elimination(singular, b)));
    BOOST_CHECK(!detail::gaussian_elimination(singular, b).has_value());

    /*  A nearly singular system is caught by the scale-aware tolerance: the first
     *  two rows differ by an amount well below eps*N*max|A_ij|.
     */
    std::vector<std::vector<double>> near_singular{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0 + 1e-15}, {1.0, 2.0, 3.0}};
    BOOST_CHECK(!detail::gaussian_elimination(near_singular, b).has_value());

    // The same system, perturbed well above the tolerance, remains solvable
    std::vector<std::vector<double>> conditioned{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0 + 1e-6}, {1.0, 2.0, 3.0}};
    BOOST_CHECK(detail::gaussian_elimination(conditioned, b).has_value());

    // Dimensions are validated
    std::vector<std::vector<double>> empty_matrix;
    BOOST_CHECK_THROW((detail::gaussian_elimination(empty_matrix, b)), std::invalid_argument);
    std::vector<std::vector<double>> non_square{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    BOOST_CHECK_THROW((detail::gaussian_elimination(non_square, b)), std::invalid_argument);
    std::vector<double> short_b{1.0, 1.0};
    BOOST_CHECK_THROW((detail::gaussian_elimination(A, short_b)), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(nsga3_test_extreme_point_duplicates){
    // Translated objectives whose nadir point, (1, 1, 1), differs from the solved intercepts
    const std::vector<std::vector<double>> translated{{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};

    /*  These extreme points are pairwise distinct, but every pair shares at least
     *  one coordinate. Comparing coordinates individually misclassifies them as
     *  duplicates and skips the solver; comparing complete vectors does not.
     */
    std::vector<std::vector<double>> distinct{{2.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 8.0}};
    auto intercepts = detail::nsga3_find_intercepts(distinct, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 2.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 8.0, 1e-8);

    // Two identical extreme points do fall back to the nadir point
    std::vector<std::vector<double>> duplicated{{2.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 0.0, 8.0}};
    intercepts = detail::nsga3_find_intercepts(duplicated, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 1.0, 1e-8);

    // As do points which differ only within the numerical tolerance
    std::vector<std::vector<double>> near_duplicated{{2.0, 0.0, 0.0}, {2.0 + 1e-15, 1e-16, -1e-16},
                                                     {0.0, 0.0, 8.0}};
    intercepts = detail::nsga3_find_intercepts(near_duplicated, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_intercepts_fallback){
    // Front 0 is {(1,3), (3,1)}; (4,4) is dominated, so the nadir point is (3, 3)
    const std::vector<std::vector<double>> translated{{1.0, 3.0}, {3.0, 1.0}, {4.0, 4.0}};

    // A negative solution component has no usable reciprocal
    std::vector<std::vector<double>> negative{{-1.0, 0.0}, {0.0, 2.0}};
    auto intercepts = detail::nsga3_find_intercepts(negative, translated);
    BOOST_CHECK_CLOSE(intercepts[0], 3.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 3.0, 1e-8);

    // A singular system falls back rather than aborting the evolution
    std::vector<std::vector<double>> singular{{0.0, 0.0}, {0.0, 2.0}};
    BOOST_CHECK_NO_THROW((intercepts = detail::nsga3_find_intercepts(singular, translated)));
    BOOST_CHECK_CLOSE(intercepts[0], 3.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 3.0, 1e-8);

    /*  A degenerate objective, identical across the population, has zero extent.
     *  The intercept is sanitised to 1.0 so that normalization leaves the
     *  coordinate at zero instead of producing an infinity or a NaN.
     */
    const std::vector<std::vector<double>> degenerate{{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}};
    std::vector<std::vector<double>> degenerate_ext{{0.0, 1.0}, {0.0, 1.0}};
    auto degenerate_intercepts = detail::nsga3_find_intercepts(degenerate_ext, degenerate);
    BOOST_CHECK_CLOSE(degenerate_intercepts[0], 1.0, 1e-8);
    auto norm_objs = detail::nsga3_normalize_objectives(degenerate, degenerate_intercepts);
    for(const auto &row: norm_objs){
        BOOST_CHECK_EQUAL(row[0], 0.0);
        for(double value: row){
            BOOST_CHECK(std::isfinite(value));
        }
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_nonzero_ideal){
    /*  Ideal point (2, 5, -1), well away from the origin. The first three points
     *  are mutually non-dominated and the fourth is dominated by all of them.
     */
    const std::vector<vector_double> objs{{2.0, 9.0, 3.0}, {6.0, 5.0, 3.0}, {6.0, 9.0, -1.0}, {6.0, 9.0, 3.0}};

    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    BOOST_CHECK_CLOSE(ideal_point[0],  2.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal_point[1],  5.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal_point[2], -1.0, 1e-8);

    auto translated = detail::nsga3_translate_objectives(objs, ideal_point);
    for(size_t obj=0; obj<ideal_point.size(); obj++){
        double col_min = std::numeric_limits<double>::max();
        for(const auto &row: translated){
            col_min = std::min(col_min, row[obj]);
        }
        BOOST_CHECK_SMALL(col_min, 1e-12);
    }

    const std::vector<std::vector<pop_size_t>> fronts{{0u, 1u, 2u}};
    auto ext_points = detail::nsga3_find_extreme_points(fronts, translated, ideal_point, nullptr);
    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated);

    /*  The extreme points coincide here, so the nadir fallback is taken. The
     *  intercepts must be the nadir of the *translated* objectives, (4, 4, 4),
     *  and not the nadir of the original objectives, (6, 9, 3).
     */
    BOOST_CHECK_CLOSE(intercepts[0], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[1], 4.0, 1e-8);
    BOOST_CHECK_CLOSE(intercepts[2], 4.0, 1e-8);

    // The individual sitting on the intercept vector normalizes to (1, 1, 1)
    auto norm_objs = detail::nsga3_normalize_objectives(translated, intercepts);
    BOOST_CHECK_CLOSE(norm_objs[3][0], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(norm_objs[3][1], 1.0, 1e-8);
    BOOST_CHECK_CLOSE(norm_objs[3][2], 1.0, 1e-8);

    /*  The same ideal point with well separated extreme points, so that the
     *  solver path is exercised: translated extremes (9,0,0), (0,5,0), (0,0,1)
     *  give x = (1/9, 1/5, 1) and therefore intercepts (9, 5, 1).
     */
    const std::vector<vector_double> spread_objs{{11.0, 5.0, -1.0}, {2.0, 10.0, -1.0}, {2.0, 5.0, 0.0}};
    auto spread_ideal = detail::nsga3_compute_ideal(spread_objs, nullptr);
    BOOST_CHECK_CLOSE(spread_ideal[0],  2.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_ideal[1],  5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_ideal[2], -1.0, 1e-8);
    auto spread_translated = detail::nsga3_translate_objectives(spread_objs, spread_ideal);
    auto spread_ext = detail::nsga3_find_extreme_points(fronts, spread_translated, spread_ideal, nullptr);
    auto spread_intercepts = detail::nsga3_find_intercepts(spread_ext, spread_translated);
    BOOST_CHECK_CLOSE(spread_intercepts[0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(spread_intercepts[2], 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_memory_ideal_shift){
    const std::vector<std::vector<pop_size_t>> fronts{{0u, 1u, 2u}};

    // Generation 1: the ideal point is (10, 10, 10)
    const std::vector<vector_double> objs1{{19.0, 10.0, 10.0}, {10.0, 15.0, 10.0}, {10.0, 10.0, 11.0}};
    // Generation 2: the ideal point improves to (8, 9, 10)
    const std::vector<vector_double> objs2{{8.0, 30.0, 30.0}, {30.0, 9.0, 30.0}, {30.0, 30.0, 10.0}};

    /*  With memory enabled the running ideal point and the retained extreme
     *  points persist across generations; nsga3 owns these two buffers.
     */
    std::vector<double> running_ideal;
    std::vector<std::vector<double>> retained_extremes;

    auto ideal1 = detail::nsga3_compute_ideal(objs1, &running_ideal);
    BOOST_CHECK_CLOSE(ideal1[0], 10.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal1[1], 10.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal1[2], 10.0, 1e-8);
    auto translated1 = detail::nsga3_translate_objectives(objs1, ideal1);
    auto ext1 = detail::nsga3_find_extreme_points(fronts, translated1, ideal1, &retained_extremes);
    // Translated extremes of generation 1: (9,0,0), (0,5,0), (0,0,1)
    BOOST_CHECK_CLOSE(ext1[0][0], 9.0, 1e-8);
    BOOST_CHECK_CLOSE(ext1[1][1], 5.0, 1e-8);
    BOOST_CHECK_CLOSE(ext1[2][2], 1.0, 1e-8);

    // The running ideal point is the elementwise minimum over both generations
    auto ideal2 = detail::nsga3_compute_ideal(objs2, &running_ideal);
    BOOST_CHECK_CLOSE(ideal2[0],  8.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal2[1],  9.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal2[2], 10.0, 1e-8);

    auto translated2 = detail::nsga3_translate_objectives(objs2, ideal2);
    auto ext2 = detail::nsga3_find_extreme_points(fronts, translated2, ideal2, &retained_extremes);

    /*  Generation 2 has no candidate better than the retained extreme points, so
     *  those are returned, re-expressed in the *current* translated coordinates:
     *  (19,10,10) - (8,9,10) = (11,1,0), and so on. Had the extreme points been
     *  retained in the translated coordinates of generation 1 they would still
     *  read (9,0,0), (0,5,0), (0,0,1) here.
     */
    const std::vector<std::vector<double>> expected_ext2{{11.0, 1.0, 0.0}, {2.0, 6.0, 0.0}, {2.0, 1.0, 1.0}};
    for(size_t i=0; i<expected_ext2.size(); i++){
        for(size_t j=0; j<expected_ext2[i].size(); j++){
            BOOST_CHECK_SMALL(ext2[i][j] - expected_ext2[i][j], 1e-9);
        }
    }

    // A worse generation does not degrade the retained ideal point
    const std::vector<vector_double> objs3{{20.0, 20.0, 20.0}, {21.0, 21.0, 21.0}, {22.0, 22.0, 22.0}};
    auto ideal3 = detail::nsga3_compute_ideal(objs3, &running_ideal);
    BOOST_CHECK_CLOSE(ideal3[0],  8.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal3[1],  9.0, 1e-8);
    BOOST_CHECK_CLOSE(ideal3[2], 10.0, 1e-8);

    /*  Without memory the same generation-2 input depends only on the current
     *  objectives: the extreme point for the first objective is (22, 0, 20).
     */
    auto plain_ideal = detail::nsga3_compute_ideal(objs2, nullptr);
    auto plain_translated = detail::nsga3_translate_objectives(objs2, plain_ideal);
    auto plain_ext = detail::nsga3_find_extreme_points(fronts, plain_translated, plain_ideal, nullptr);
    BOOST_CHECK_CLOSE(plain_ext[0][0], 22.0, 1e-8);
    BOOST_CHECK_SMALL(plain_ext[0][1], 1e-12);
    BOOST_CHECK_CLOSE(plain_ext[0][2], 20.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_extreme_points){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<std::vector<double>> ext_points{};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto fnds_res = fast_non_dominated_sorting(objs);
    auto fronts = std::get<0>(fnds_res);
    BOOST_CHECK_NO_THROW((ext_points = detail::nsga3_find_extreme_points(fronts, translated_objectives, ideal_point, nullptr)));
    size_t point_count = ext_points.size();
    BOOST_CHECK_EQUAL(point_count, udp.get_nobj());
}

BOOST_AUTO_TEST_CASE(nsga3_test_find_intercepts){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<double> intercepts{};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto fnds_res = fast_non_dominated_sorting(objs);
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = detail::nsga3_find_extreme_points(fronts, translated_objectives, ideal_point, nullptr);

    BOOST_CHECK_NO_THROW((intercepts = detail::nsga3_find_intercepts(ext_points, translated_objectives)));
    BOOST_CHECK_EQUAL(intercepts.size(), udp.get_nobj());
    // Intercepts are always usable divisors
    for(double intercept: intercepts){
        BOOST_CHECK(std::isfinite(intercept));
        BOOST_CHECK(intercept > 0.0);
    }
}

BOOST_AUTO_TEST_CASE(nsga3_test_normalize_objectives){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};
    nsga3 nsga3_alg{10u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    std::vector<std::vector<double>> norm_objs{};

    pop = nsga3_alg.evolve(pop);
    auto objs = pop.get_f();
    auto ideal_point = detail::nsga3_compute_ideal(objs, nullptr);
    auto translated_objectives = detail::nsga3_translate_objectives(objs, ideal_point);
    auto fnds_res = fast_non_dominated_sorting(objs);
    auto fronts = std::get<0>(fnds_res);
    auto ext_points = detail::nsga3_find_extreme_points(fronts, translated_objectives, ideal_point, nullptr);
    auto intercepts = detail::nsga3_find_intercepts(ext_points, translated_objectives);
    BOOST_CHECK_NO_THROW((norm_objs = detail::nsga3_normalize_objectives(translated_objectives, intercepts)));
    size_t obj_count = norm_objs.size();
    BOOST_CHECK_EQUAL(obj_count, translated_objectives.size());
}

BOOST_AUTO_TEST_CASE(nsga3_reproducibility_same_seed){
    dtlz udp{1u, 10u, 3u};

    population pop_a{udp, 52u, 23u};
    population pop_b{udp, 52u, 23u};
    nsga3 alg_a{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    nsga3 alg_b{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    alg_a.set_verbosity(1u);
    alg_b.set_verbosity(1u);

    pop_a = alg_a.evolve(pop_a);
    pop_b = alg_b.evolve(pop_b);

    BOOST_CHECK(pop_a.get_x() == pop_b.get_x());
    BOOST_CHECK(pop_a.get_f() == pop_b.get_f());
    BOOST_CHECK(alg_a.get_log() == alg_b.get_log());
}

BOOST_AUTO_TEST_CASE(nsga3_instance_independence){
    dtlz udp{1u, 10u, 3u};

    // Baseline run
    population pop_ref{udp, 52u, 23u};
    nsga3 alg_ref{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    pop_ref = alg_ref.evolve(pop_ref);

    /*  A differently seeded instance evolved in between, and a reseeded global
     *  random device, must leave an identically seeded run unchanged.
     */
    population pop_other{udp, 52u, 23u};
    nsga3 alg_other{5u, 1.00, 30., 0.10, 20., 5u, 7u, false};
    pop_other = alg_other.evolve(pop_other);
    random_device::set_seed(987654u);

    population pop_test{udp, 52u, 23u};
    nsga3 alg_test{5u, 1.00, 30., 0.10, 20., 5u, 42u, false};
    pop_test = alg_test.evolve(pop_test);

    BOOST_CHECK(pop_ref.get_x() == pop_test.get_x());
    BOOST_CHECK(pop_ref.get_f() == pop_test.get_f());
    // A differently seeded instance really does explore differently
    BOOST_CHECK(pop_ref.get_f() != pop_other.get_f());

    // Constructing an nsga3 must not disturb the global random device
    random_device::set_seed(4242u);
    unsigned expected = random_device::next();
    random_device::set_seed(4242u);
    nsga3 constructed{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    BOOST_CHECK_EQUAL(constructed.get_seed(), 32u);
    BOOST_CHECK_EQUAL(expected, random_device::next());
}

BOOST_AUTO_TEST_CASE(nsga3_log_generation_numbers){
    dtlz udp{1u, 10u, 3u};

    // Verbosity 1 logs every generation, numbered from 1
    population pop1{udp, 52u, 23u};
    nsga3 alg1{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    alg1.set_verbosity(1u);
    pop1 = alg1.evolve(pop1);
    const auto &log1 = alg1.get_log();
    BOOST_REQUIRE_EQUAL(log1.size(), 5u);
    for(unsigned i=0; i<log1.size(); i++){
        BOOST_CHECK_EQUAL(std::get<0>(log1[i]), i + 1u);
    }
    // Function evaluations accumulate across generations
    for(size_t i=1; i<log1.size(); i++){
        BOOST_CHECK(std::get<1>(log1[i]) > std::get<1>(log1[i-1]));
    }

    // Verbosity 2 logs generations 1, 3 and 5
    population pop2{udp, 52u, 23u};
    nsga3 alg2{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    alg2.set_verbosity(2u);
    pop2 = alg2.evolve(pop2);
    const auto &log2 = alg2.get_log();
    BOOST_REQUIRE_EQUAL(log2.size(), 3u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[0]), 1u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[1]), 3u);
    BOOST_CHECK_EQUAL(std::get<0>(log2[2]), 5u);

    // Verbosity 0 logs nothing
    population pop3{udp, 52u, 23u};
    nsga3 alg3{5u, 1.00, 30., 0.10, 20., 5u, 32u, false};
    pop3 = alg3.evolve(pop3);
    BOOST_CHECK(alg3.get_log().empty());
}

static void nsga3_verify_serialization_continuation(bool use_memory){
    dtlz udp{1u, 10u, 3u};
    population pop{udp, 52u, 23u};

    algorithm algo{nsga3{3u, 1.00, 30., 0.10, 20., 5u, 32u, use_memory}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    /*  Round-trip the *evolved* algorithm. Continuing the evolution from the
     *  restored copy requires the engine state, the inter-generational memory
     *  and the constructor arguments to have all been archived.
     */
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << algo;
    }
    algorithm restored{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> restored;
    }

    BOOST_CHECK_EQUAL(algo.get_extra_info(), restored.get_extra_info());
    BOOST_CHECK(algo.extract<nsga3>()->get_log() == restored.extract<nsga3>()->get_log());

    population continued_direct{pop};
    population continued_restored{pop};
    continued_direct = algo.evolve(continued_direct);
    continued_restored = restored.evolve(continued_restored);

    BOOST_CHECK(continued_direct.get_x() == continued_restored.get_x());
    BOOST_CHECK(continued_direct.get_f() == continued_restored.get_f());
    BOOST_CHECK(algo.extract<nsga3>()->get_log() == restored.extract<nsga3>()->get_log());
}

BOOST_AUTO_TEST_CASE(nsga3_serialization_continuation){
    nsga3_verify_serialization_continuation(false);
    nsga3_verify_serialization_continuation(true);
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
