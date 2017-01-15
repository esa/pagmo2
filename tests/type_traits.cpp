#define BOOST_TEST_MODULE type_traits_test
#include <boost/test/included/unit_test.hpp>

#include <type_traits>
#include <utility>

#include <pagmo/type_traits.hpp>

using namespace pagmo;

struct s1
{
    void foo();
};

struct s2
{
    void foo() const;
};

struct s3 {};

struct s4
{
    void foo(int) const;
};

template <typename T>
using foo_t = decltype(std::declval<const T &>().foo());

BOOST_AUTO_TEST_CASE(type_traits_test_00)
{
    BOOST_CHECK((!is_detected<foo_t,s1>::value));
    BOOST_CHECK((is_detected<foo_t,s2>::value));
    BOOST_CHECK((!is_detected<foo_t,s3>::value));
    BOOST_CHECK((!is_detected<foo_t,s4>::value));
    BOOST_CHECK((std::is_same<detected_t<foo_t,s4>,nonesuch>::value));
}
