
#define BOOST_TEST_MODULE gaco_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <string>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(construction_test)
{
    g_aco user_algo{100u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 2u, 1000u, 1000u, 0.0, 10u, 0.9, 23u};
    BOOST_CHECK(user_algo.get_verbosity() == 0u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK((user_algo.get_log() == g_aco::log_type{}));

    BOOST_CHECK_THROW((g_aco{100u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 2u, 1000u, 1000u, -0.1, 10u, 0.9, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((g_aco{100u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 2u, 1000u, 1000u, 0.0, 10u, -0.1, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((g_aco{100u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 2u, 1000u, 1000u, 0.0, 10u, 1.1, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((g_aco{100u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 3u, 1000u, 1000u, 0.0, 10u, 0.9, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((g_aco{100u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 0u, 1000u, 1000u, 0.0, 10u, 0.9, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((g_aco{100u, 13u, 1.0, 0.0, 0.01, 101u, 1e-7, 7u, 2u, 1000u, 1000u, 0.0, 10u, 0.9, 23u}),
                      std::invalid_argument);
    BOOST_CHECK_THROW((g_aco{100u, 13u, 1.0, 0.0, 0.01, 0u, 1e-7, 7u, 2u, 1000u, 1000u, 0.0, 10u, 0.9, 23u}),
                      std::invalid_argument);
}

struct gaco_sto {
    /// Fitness
    vector_double fitness(const vector_double &) const
    {
        return {0., 0.};
    }
    vector_double::size_type get_nobj() const
    {
        return 1u;
    }
    /// Problem bounds
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{0., 0.}, {1., 1.}};
    }
    void set_seed(unsigned int) {}
};

BOOST_AUTO_TEST_CASE(evolve_test)
{
    // Here we only test that evolution is deterministic if the
    // seed is controlled for all variants
    {
        problem prob{rosenbrock{25u}};
        population pop1{prob, 200u, 23u};
        population pop2{prob, 200u, 23u};
        population pop3{prob, 200u, 23u};

        for (unsigned int j = 1u; j <= 2u; ++j) {
            for (unsigned int i = 1u; i < 10u; ++i) {
                std::cout << "qui1" << std::endl;
                g_aco user_algo1{10u, 13u, 1.0, 0.0, 0.01, i, 1e-7, 7u, j, 1000u, 1000u, 0.0, 10u, 0.9, 23u};
                std::cout << "qui2" << std::endl;
                user_algo1.set_verbosity(1u);
                pop1 = user_algo1.evolve(pop1);

                BOOST_CHECK(user_algo1.get_log().size() > 0u);

                g_aco user_algo2{10u, 13u, 1.0, 0.0, 0.01, i, 1e-7, 7u, j, 1000u, 1000u, 0.0, 10u, 0.9, 23u};
                user_algo2.set_verbosity(1u);
                pop2 = user_algo2.evolve(pop2);

                BOOST_CHECK(user_algo1.get_log() == user_algo2.get_log());

                g_aco user_algo3{10u, 13u, 1.0, 0.0, 0.01, i, 1e-7, 7u, j, 1000u, 1000u, 0.0, 10u, 0.9, 23u};
                user_algo3.set_verbosity(1u);
                pop3 = user_algo3.evolve(pop3);

                BOOST_CHECK(user_algo2.get_log() == user_algo3.get_log());
            }
        }
    }
    // Here we check that the exit condition of impstop, evalstop and fstop actually provoke an exit within 300u gen
    // (rosenbrock{10} and rosenbrock{2} are used)
    {
        g_aco user_algo{2000u, 150u, 1.0, 0.0, 0.01, 1500u, 1.0, 7u, 2u, 3u, 1000u, 0.0, 10u, 0.9, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 200u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 2000u);
    }
    {
        g_aco user_algo{2000u, 150u, 1.0, 0.0, 0.01, 1500u, 1.0, 7u, 2u, 1000u, 3u, 0.0, 10u, 0.9, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 200u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 2000u);
    }
    {
        g_aco user_algo{2000u, 150u, 1.0, 0.0, 0.01, 1500u, 1.0, 7u, 2u, 1000u, 1000u, 0.0, 10u, 0.9, 23u};
        user_algo.set_verbosity(1u);
        problem prob{rosenbrock{2u}};
        population pop{prob, 200u, 23u};
        pop = user_algo.evolve(pop);
        BOOST_CHECK(user_algo.get_log().size() < 2000u);
    }

    // We then check that the evolve throws if called on unsuitable problems
    // Integer variables problem
    BOOST_CHECK_THROW(g_aco{100u}.evolve(population{problem{minlp_rastrigin{}}, 64u}), std::invalid_argument);
    // Multi-objective problem
    BOOST_CHECK_THROW(g_aco{100u}.evolve(population{problem{zdt{}}, 64u}), std::invalid_argument);
    // Empty population
    BOOST_CHECK_THROW(g_aco{100u}.evolve(population{problem{rosenbrock{}}, 0u}), std::invalid_argument);
    // Population size smaller than ker size
    BOOST_CHECK_THROW(g_aco{100u}.evolve(population{problem{rosenbrock{}}, 60u}), std::invalid_argument);
    // Stochastic problem
    BOOST_CHECK_THROW(g_aco{100u}.evolve(population{problem{gaco_sto{}}, 64u}), std::invalid_argument);
    // And a clean exit for 0 generations
    population pop{rosenbrock{25u}, 10u};
    BOOST_CHECK(g_aco{0u}.evolve(pop).get_x()[0] == pop.get_x()[0]);
}

BOOST_AUTO_TEST_CASE(setters_getters_test)
{
    g_aco user_algo{10000000u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 2u, 1000u, 1000u, 0.0, 10u, 0.9, 23u};
    user_algo.set_verbosity(23u);
    BOOST_CHECK(user_algo.get_verbosity() == 23u);
    user_algo.set_seed(23u);
    BOOST_CHECK(user_algo.get_seed() == 23u);
    BOOST_CHECK(user_algo.get_name().find("GACO: Ant Colony Optimization") != std::string::npos);
    BOOST_CHECK(user_algo.get_extra_info().find("Oracle parameter") != std::string::npos);
    BOOST_CHECK_NO_THROW(user_algo.get_log());
}

BOOST_AUTO_TEST_CASE(serialization_test)
{
    // Make one evolution
    problem prob{rosenbrock{2u}};
    population pop{prob, 15u, 23u};
    algorithm algo{g_aco{10000000u, 13u, 1.0, 0.0, 0.01, 90u, 1e-7, 7u, 2u, 1000u, 1000u, 0.0, 10u, 0.9, 23u}};
    algo.set_verbosity(1u);
    pop = algo.evolve(pop);

    // Store the string representation of p.
    std::stringstream ss;
    auto before_text = boost::lexical_cast<std::string>(algo);
    auto before_log = algo.extract<g_aco>()->get_log();
    // Now serialize, deserialize and compare the result.
    {
        cereal::JSONOutputArchive oarchive(ss);
        oarchive(algo);
    }
    // Change the content of p before deserializing.
    algo = algorithm{null_algorithm{}};
    {
        cereal::JSONInputArchive iarchive(ss);
        iarchive(algo);
    }
    auto after_text = boost::lexical_cast<std::string>(algo);
    auto after_log = algo.extract<g_aco>()->get_log();
    BOOST_CHECK_EQUAL(before_text, after_text);
    // BOOST_CHECK(before_log == after_log); // This fails because of floating point problems when using JSON and cereal
    // so we implement a close check
    BOOST_CHECK(before_log.size() > 0u);
    for (auto i = 0u; i < before_log.size(); ++i) {
        BOOST_CHECK_EQUAL(std::get<0>(before_log[i]), std::get<0>(after_log[i]));
        BOOST_CHECK_EQUAL(std::get<1>(before_log[i]), std::get<1>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<2>(before_log[i]), std::get<2>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<3>(before_log[i]), std::get<3>(after_log[i]), 1e-8);
        BOOST_CHECK_EQUAL(std::get<4>(before_log[i]), std::get<4>(after_log[i]));
        BOOST_CHECK_CLOSE(std::get<5>(before_log[i]), std::get<5>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<6>(before_log[i]), std::get<6>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<7>(before_log[i]), std::get<7>(after_log[i]), 1e-8);
        BOOST_CHECK_CLOSE(std::get<8>(before_log[i]), std::get<8>(after_log[i]), 1e-8);
    }
}
