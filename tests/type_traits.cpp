#include "../include/type_traits.hpp"

#define BOOST_TEST_MODULE pagmo_type_traits_test
#include <boost/test/unit_test.hpp>

#include "../include/types.hpp"

using namespace pagmo;

// No fitness.
struct f_00 {};

// Various types of wrong fitness.
struct f_01
{
    vector_double::size_type get_nobj() const;
    void fitness();
};

struct f_02
{
    vector_double::size_type get_nobj() const;
    void fitness(const vector_double &);
};

struct f_03
{
    vector_double::size_type get_nobj() const;
    vector_double fitness(const vector_double &);
};

struct f_04
{
    vector_double::size_type get_nobj() const;
    vector_double fitness(vector_double &) const;
};

// Good one.
struct f_05
{
    vector_double fitness(const vector_double &) const;
    vector_double::size_type get_nobj() const;
};

struct f_06
{
    vector_double fitness(const vector_double &) const;
    vector_double::size_type get_nobj();
};

struct f_07
{
    vector_double fitness(const vector_double &) const;
    int get_nobj() const;
};

BOOST_AUTO_TEST_CASE(has_fitness_test)
{
    BOOST_CHECK((!has_fitness<f_00>::value));
    BOOST_CHECK((!has_fitness<f_01>::value));
    BOOST_CHECK((!has_fitness<f_02>::value));
    BOOST_CHECK((!has_fitness<f_03>::value));
    BOOST_CHECK((!has_fitness<f_04>::value));
    BOOST_CHECK((has_fitness<f_05>::value));
    BOOST_CHECK((!has_fitness<f_06>::value));
    BOOST_CHECK((!has_fitness<f_07>::value));
}
