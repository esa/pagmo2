#define BOOST_TEST_MODULE algorithm_type_traits_test
#include <boost/test/included/unit_test.hpp>

#include <utility>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>

using namespace pagmo;

struct hsv_00 {
};

// The good one
struct hsv_01 {
    void set_verbosity(unsigned int);
};

// also good
struct hsv_02 {
    void set_verbosity(unsigned int) const;
};

// also good
struct hsv_03 {
    void set_verbosity(int);
};

struct hsv_04 {
    double set_verbosity(unsigned int);
};

BOOST_AUTO_TEST_CASE(has_set_verbose_test)
{
    BOOST_CHECK((!has_set_verbosity<hsv_00>::value));
    BOOST_CHECK((has_set_verbosity<hsv_01>::value));
    BOOST_CHECK((has_set_verbosity<hsv_02>::value));
    BOOST_CHECK((has_set_verbosity<hsv_03>::value));
    BOOST_CHECK((!has_set_verbosity<hsv_04>::value));
}

struct hev_00 {
};

// The good one
struct hev_01 {
    population evolve(population) const;
};

struct hev_02 {
    population evolve(const population &);
};

struct hev_03 {
    population evolve(population &) const;
};

struct hev_04 {
    double evolve(const population &) const;
};

struct hev_05 {
    population evolve(const double &) const;
};

BOOST_AUTO_TEST_CASE(has_evolve_test)
{
    BOOST_CHECK((!has_evolve<hev_00>::value));
    BOOST_CHECK((has_evolve<hev_01>::value));
    BOOST_CHECK((!has_evolve<hev_02>::value));
    BOOST_CHECK((!has_evolve<hev_03>::value));
    BOOST_CHECK((!has_evolve<hev_04>::value));
    BOOST_CHECK((!has_evolve<hev_05>::value));
}
