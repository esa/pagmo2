#include "../include/type_traits.hpp"

#define BOOST_TEST_MODULE pagmo_type_traits_test
#include <boost/test/unit_test.hpp>

#include <string>
#include <utility>

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

struct db_00
{};

// The good one.
struct db_01
{
    vector_double::size_type get_n() const;
    std::pair<vector_double,vector_double> get_bounds() const;
};

struct db_02
{
    vector_double::size_type get_n();
    std::pair<vector_double,vector_double> get_bounds() const;
};

struct db_03
{
    int get_n() const;
    std::pair<vector_double,vector_double> get_bounds() const;
};

struct db_04
{
    vector_double::size_type get_n() const;
    std::pair<vector_double,vector_double> get_bounds();
};

struct db_05
{
    vector_double::size_type get_n() const;
    vector_double get_bounds() const;
};

struct db_06
{
    std::pair<vector_double,vector_double> get_bounds() const;
};

struct db_07
{
    vector_double::size_type get_n() const;
};

BOOST_AUTO_TEST_CASE(has_dimensions_bounds_test)
{
    BOOST_CHECK((!has_dimensions_bounds<db_00>::value));
    BOOST_CHECK((has_dimensions_bounds<db_01>::value));
    BOOST_CHECK((!has_dimensions_bounds<db_02>::value));
    BOOST_CHECK((!has_dimensions_bounds<db_03>::value));
    BOOST_CHECK((!has_dimensions_bounds<db_04>::value));
    BOOST_CHECK((!has_dimensions_bounds<db_05>::value));
    BOOST_CHECK((!has_dimensions_bounds<db_06>::value));
    BOOST_CHECK((!has_dimensions_bounds<db_07>::value));
}

struct c_00
{};

// The good one.
struct c_01
{
    vector_double::size_type get_nec() const;
    vector_double::size_type get_nic() const;
};

struct c_02
{
    vector_double::size_type get_nec();
    vector_double::size_type get_nic() const;
};

struct c_03
{
    int get_nec() const;
    vector_double::size_type get_nic() const;
};

struct c_04
{
    vector_double::size_type get_nec() const;
    vector_double::size_type get_nic();
};

struct c_05
{
    vector_double::size_type get_nec() const;
    void get_nic() const;
};

struct c_06
{
    vector_double::size_type get_nec() const;
};

struct c_07
{
    vector_double::size_type get_nic() const;
};

BOOST_AUTO_TEST_CASE(has_constraints_test)
{
    BOOST_CHECK((!has_constraints<c_00>::value));
    BOOST_CHECK((has_constraints<c_01>::value));
    BOOST_CHECK((!has_constraints<c_02>::value));
    BOOST_CHECK((!has_constraints<c_03>::value));
    BOOST_CHECK((!has_constraints<c_04>::value));
    BOOST_CHECK((!has_constraints<c_05>::value));
    BOOST_CHECK((!has_constraints<c_06>::value));
    BOOST_CHECK((!has_constraints<c_07>::value));
}

struct n_00 {};

// The good one.
struct n_01
{
    std::string get_name() const;
};

struct n_02
{
    std::string get_name();
};

struct n_03
{
    void get_name() const;
};

BOOST_AUTO_TEST_CASE(has_name_test)
{
    BOOST_CHECK((!has_name<n_00>::value));
    BOOST_CHECK((has_name<n_01>::value));
    BOOST_CHECK((!has_name<n_02>::value));
    BOOST_CHECK((!has_name<n_03>::value));
}

struct ei_00 {};

// The good one.
struct ei_01
{
    std::string get_extra_info() const;
};

struct ei_02
{
    std::string get_extra_info();
};

struct ei_03
{
    void get_extra_info() const;
};

BOOST_AUTO_TEST_CASE(has_extra_info_test)
{
    BOOST_CHECK((!has_extra_info<ei_00>::value));
    BOOST_CHECK((has_extra_info<ei_01>::value));
    BOOST_CHECK((!has_extra_info<ei_02>::value));
    BOOST_CHECK((!has_extra_info<ei_03>::value));
}
