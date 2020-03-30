/* Copyright 2017-2020 PaGMO development team

This file is part of the PaGMO library.

The PaGMO library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 3 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The PaGMO library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the PaGMO library.  If not,
see https://www.gnu.org/licenses/. */

#if defined(_MSC_VER)

// Disable warnings from MSVC.
#pragma warning(disable : 4822)

#endif

#define BOOST_TEST_MODULE r_policy_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

#include <pagmo/detail/type_name.hpp>
#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

BOOST_AUTO_TEST_CASE(type_traits_tests)
{
    BOOST_CHECK(!is_udrp<void>::value);
    BOOST_CHECK(!is_udrp<int>::value);
    BOOST_CHECK(!is_udrp<double>::value);

    struct udrp00 {
        individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const;
    };

    BOOST_CHECK(is_udrp<udrp00>::value);
    BOOST_CHECK(!is_udrp<const udrp00>::value);
    BOOST_CHECK(!is_udrp<const udrp00 &>::value);
    BOOST_CHECK(!is_udrp<udrp00 &>::value);

    struct no_udrp00 {
        void replace(const individuals_group_t &, const vector_double::size_type &, const vector_double::size_type &,
                     const vector_double::size_type &, const vector_double::size_type &,
                     const vector_double::size_type &, const vector_double &, const individuals_group_t &) const;
    };

    BOOST_CHECK(!is_udrp<no_udrp00>::value);

    struct no_udrp01 {
        individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &);
    };

    BOOST_CHECK(!is_udrp<no_udrp01>::value);

    struct no_udrp02 {
        no_udrp02() = delete;
        individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const;
    };

    BOOST_CHECK(!is_udrp<no_udrp02>::value);
}

struct udrp1 {
    individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double &, const individuals_group_t &) const
    {
        return inds;
    }
    std::string foo = "hello world";
};

struct udrp2 {
    udrp2() = default;
    udrp2(const udrp2 &other) : foo{new std::string{*other.foo}} {}
    udrp2(udrp2 &&) = default;
    individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double &, const individuals_group_t &) const
    {
        return inds;
    }
    std::string get_name() const
    {
        return "frobniz";
    }
    std::unique_ptr<std::string> foo = std::unique_ptr<std::string>{new std::string{"hello world"}};
};

BOOST_AUTO_TEST_CASE(basic_tests)
{
    r_policy r;

    BOOST_CHECK(r.is<fair_replace>());
    BOOST_CHECK(!r.is<udrp1>());

    BOOST_CHECK(r.extract<fair_replace>() != nullptr);
    BOOST_CHECK(r.extract<udrp1>() == nullptr);

    BOOST_CHECK(static_cast<const r_policy &>(r).extract<fair_replace>() != nullptr);
    BOOST_CHECK(static_cast<const r_policy &>(r).extract<udrp1>() == nullptr);

    BOOST_CHECK(r.get_name() == "Fair replace");
    BOOST_CHECK(!r.get_extra_info().empty());

    BOOST_CHECK(r_policy(udrp1{}).get_extra_info().empty());
    BOOST_CHECK(r_policy(udrp1{}).get_name() == detail::type_name<udrp1>());

    // Constructors, assignments.
    // Generic constructor with copy.
    udrp1 r1;
    r_policy r_pol1{r1};
    BOOST_CHECK(r1.foo == "hello world");
    BOOST_CHECK(r_pol1.extract<udrp1>()->foo == "hello world");
    // Generic constructor with move.
    udrp2 r2;
    r_policy r_pol2{std::move(r2)};
    BOOST_CHECK(r2.foo.get() == nullptr);
    BOOST_CHECK(r_pol2.extract<udrp2>()->foo.get() != nullptr);
    BOOST_CHECK(*r_pol2.extract<udrp2>()->foo == "hello world");
    // Copy constructor.
    udrp2 r3;
    r_policy r_pol3{r3}, r_pol4{r_pol3};
    BOOST_CHECK(*r_pol4.extract<udrp2>()->foo == "hello world");
    BOOST_CHECK(r_pol4.extract<udrp2>()->foo.get() != r_pol3.extract<udrp2>()->foo.get());
    BOOST_CHECK(r_pol4.get_name() == "frobniz");
    // Move constructor.
    r_policy r_pol5{std::move(r_pol4)};
    BOOST_CHECK(*r_pol5.extract<udrp2>()->foo == "hello world");
    BOOST_CHECK(r_pol5.get_name() == "frobniz");
    // Revive r_pol4 via copy assignment.
    r_pol4 = r_pol5;
    BOOST_CHECK(*r_pol4.extract<udrp2>()->foo == "hello world");
    BOOST_CHECK(r_pol4.get_name() == "frobniz");
    // Revive r_pol4 via move assignment.
    r_policy r_pol6{std::move(r_pol4)};
    r_pol4 = std::move(r_pol5);
    BOOST_CHECK(*r_pol4.extract<udrp2>()->foo == "hello world");
    BOOST_CHECK(r_pol4.get_name() == "frobniz");
    // Self move-assignment.
    r_pol4 = std::move(*&r_pol4);
    BOOST_CHECK(*r_pol4.extract<udrp2>()->foo == "hello world");
    BOOST_CHECK(r_pol4.get_name() == "frobniz");

    // Minimal iostream test.
    {
        std::ostringstream oss;
        oss << r;
        BOOST_CHECK(!oss.str().empty());
    }

    // Minimal serialization test.
    {
        std::string before;
        std::stringstream ss;
        {
            before = boost::lexical_cast<std::string>(r);
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << r;
        }
        r = r_policy{udrp1{}};
        BOOST_CHECK(r.is<udrp1>());
        BOOST_CHECK(before != boost::lexical_cast<std::string>(r));
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> r;
        }
        BOOST_CHECK(before == boost::lexical_cast<std::string>(r));
        BOOST_CHECK(r.is<fair_replace>());
    }

    std::cout << r_policy{} << '\n';
}

BOOST_AUTO_TEST_CASE(optional_tests)
{
    // get_name().
    struct udrp_00 {
        individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return inds;
        }
        std::string get_name() const
        {
            return "frobniz";
        }
    };
    BOOST_CHECK_EQUAL(r_policy{udrp_00{}}.get_name(), "frobniz");
    struct udrp_01 {
        individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return inds;
        }
        // Missing const.
        std::string get_name()
        {
            return "frobniz";
        }
    };
    BOOST_CHECK(r_policy{udrp_01{}}.get_name() != "frobniz");

    // get_extra_info().
    struct udrp_02 {
        individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return inds;
        }
        std::string get_extra_info() const
        {
            return "frobniz";
        }
    };
    BOOST_CHECK_EQUAL(r_policy{udrp_02{}}.get_extra_info(), "frobniz");
    struct udrp_03 {
        individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return inds;
        }
        // Missing const.
        std::string get_extra_info()
        {
            return "frobniz";
        }
    };
    BOOST_CHECK(r_policy{udrp_03{}}.get_extra_info().empty());
}

BOOST_AUTO_TEST_CASE(stream_operator)
{
    struct udrp_00 {
        individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return inds;
        }
    };
    {
        std::ostringstream oss;
        oss << r_policy{udrp_00{}};
        BOOST_CHECK(!oss.str().empty());
    }
    struct udrp_01 {
        individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return inds;
        }
        std::string get_extra_info() const
        {
            return "bartoppo";
        }
    };
    {
        std::ostringstream oss;
        oss << r_policy{udrp_01{}};
        const auto st = oss.str();
        BOOST_CHECK(boost::contains(st, "bartoppo"));
        BOOST_CHECK(boost::contains(st, "Extra info:"));
    }
}

BOOST_AUTO_TEST_CASE(replace)
{
    r_policy r0;

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0}, {}, {}}, 0, 0, 0, 0, 0, {}, individuals_group_t{}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(),
                                  "an invalid group of individuals was passed to a replacement policy of type 'Fair "
                                  "replace': the sets of individuals IDs, decision vectors and fitness vectors "
                                  "must all have the same sizes, but instead their sizes are 1, 0 and 0");
                          });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 0, 0, 0, 0, 0, {},
                                     individuals_group_t{{0}, {{1.}, {1.}}, {{1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(),
                                  "an invalid group of migrants was passed to a replacement policy of type 'Fair "
                                  "replace': the sets of migrants IDs, decision vectors and fitness vectors "
                                  "must all have the same sizes, but instead their sizes are 1, 2 and 1");
                          });

    BOOST_CHECK_EXCEPTION(
        r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 0, 0, 0, 0, 0, {},
                   individuals_group_t{{0}, {{1.}}, {{1.}}}),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(), "a problem dimension of zero was passed to a replacement policy of type 'Fair replace'");
        });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 1, 2, 0, 0, 0, {},
                                     individuals_group_t{{0}, {{1.}}, {{1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(), "the integer dimension (2) passed to a replacement policy of type "
                                             "'Fair replace' is larger than the supplied problem dimension (1)");
                          });

    BOOST_CHECK_EXCEPTION(
        r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 1, 0, 0, 0, 0, {},
                   individuals_group_t{{0}, {{1.}}, {{1.}}}),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(),
                "an invalid number of objectives (0) was passed to a replacement policy of type 'Fair replace'");
        });

    BOOST_CHECK_EXCEPTION(
        r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 1, 0, std::numeric_limits<vector_double::size_type>::max(),
                   0, 0, {}, individuals_group_t{{0}, {{1.}}, {{1.}}}),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(),
                                   "the number of objectives ("
                                       + std::to_string(std::numeric_limits<vector_double::size_type>::max())
                                       + ") passed to a replacement policy of type 'Fair replace' is too large");
        });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 1, 0, 1,
                                     std::numeric_limits<vector_double::size_type>::max(), 0, {},
                                     individuals_group_t{{0}, {{1.}}, {{1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(),
                                  "the number of equality constraints ("
                                      + std::to_string(std::numeric_limits<vector_double::size_type>::max())
                                      + ") passed to a replacement policy of type 'Fair replace' is too large");
                          });

    BOOST_CHECK_EXCEPTION(
        r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 1, 0, 1, 0,
                   std::numeric_limits<vector_double::size_type>::max(), {}, individuals_group_t{{0}, {{1.}}, {{1.}}}),
        std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(),
                                   "the number of inequality constraints ("
                                       + std::to_string(std::numeric_limits<vector_double::size_type>::max())
                                       + ") passed to a replacement policy of type 'Fair replace' is too large");
        });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0}, {{1.}}, {{1.}}}, 1, 0, 1, 1, 1, {},
                                     individuals_group_t{{0}, {{1.}}, {{1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(),
                                  "the vector of tolerances passed to a replacement policy of type 'Fair replace' has "
                                  "a dimension (0) which is inconsistent with the total number of constraints (2)");
                          });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0, 1}, {{1.}, {}}, {{1.}, {1.}}}, 1, 0, 1, 0, 0, {},
                                     individuals_group_t{{0}, {{1.}}, {{1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(), "not all the individuals passed to a replacement policy of type 'Fair "
                                             "replace' have the expected dimension (1)");
                          });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {}}}, 1, 0, 1, 0, 0, {},
                                     individuals_group_t{{0}, {{1.}}, {{1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(), "not all the individuals passed to a replacement policy of type 'Fair "
                                             "replace' have the expected fitness dimension (1)");
                          });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}, 1, 0, 1, 0, 0, {},
                                     individuals_group_t{{0, 1}, {{1.}, {}}, {{1.}, {1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "not all the migrants passed to a replacement policy of type "
                                                     "'Fair replace' have the expected dimension (1)");
                          });

    BOOST_CHECK_EXCEPTION(r0.replace(individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}, 1, 0, 1, 0, 0, {},
                                     individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "not all the migrants passed to a replacement policy of type "
                                                     "'Fair replace' have the expected fitness dimension (1)");
                          });

    struct fail_0 {
        individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return individuals_group_t{{0}, {}, {}};
        }
        std::string get_name() const
        {
            return "fail_0";
        }
    };

    BOOST_CHECK_EXCEPTION(r_policy{fail_0{}}.replace(individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}, 1, 0, 1,
                                                     0, 0, {}, individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(
                                  ia.what(),
                                  "an invalid group of individuals was returned by a replacement policy of type "
                                  "'fail_0': the sets of individuals IDs, decision vectors and fitness vectors "
                                  "must all have the same sizes, but instead their sizes are 1, 0 and 0");
                          });

    struct fail_1 {
        individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return individuals_group_t{{0, 1}, {{1}, {}}, {{1}, {1}}};
        }
        std::string get_name() const
        {
            return "fail_1";
        }
    };

    BOOST_CHECK_EXCEPTION(r_policy{fail_1{}}.replace(individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}, 1, 0, 1,
                                                     0, 0, {}, individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "not all the individuals returned by a replacement "
                                                     "policy of type 'fail_1' have the expected dimension (1)");
                          });

    struct fail_2 {
        individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double::size_type &, const vector_double::size_type &,
                                    const vector_double &, const individuals_group_t &) const
        {
            return individuals_group_t{{0, 1}, {{1}, {1}}, {{1}, {}}};
        }
        std::string get_name() const
        {
            return "fail_2";
        }
    };

    BOOST_CHECK_EXCEPTION(r_policy{fail_2{}}.replace(individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}, 1, 0, 1,
                                                     0, 0, {}, individuals_group_t{{0, 1}, {{1.}, {1.}}, {{1.}, {1.}}}),
                          std::invalid_argument, [](const std::invalid_argument &ia) {
                              return boost::contains(ia.what(),
                                                     "not all the individuals returned by a replacement policy of type "
                                                     "'fail_2' have the expected fitness dimension (1)");
                          });
}

struct udrp_a {
    individuals_group_t replace(const individuals_group_t &inds, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double::size_type &, const vector_double::size_type &,
                                const vector_double &, const individuals_group_t &) const
    {
        return inds;
    }
    std::string get_name() const
    {
        return "abba";
    }
    std::string get_extra_info() const
    {
        return "dabba";
    }
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &state;
    }
    int state = 42;
};

PAGMO_S11N_R_POLICY_EXPORT(udrp_a)

// Serialization tests.
BOOST_AUTO_TEST_CASE(s11n)
{
    r_policy r_pol0{udrp_a{}};
    BOOST_CHECK(r_pol0.extract<udrp_a>()->state == 42);
    r_pol0.extract<udrp_a>()->state = -42;
    // Store the string representation.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(r_pol0);
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << r_pol0;
    }
    // Change the content of p before deserializing.
    r_pol0 = r_policy{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> r_pol0;
    }
    auto after = boost::lexical_cast<std::string>(r_pol0);
    BOOST_CHECK_EQUAL(before, after);
    BOOST_CHECK(r_pol0.is<udrp_a>());
    BOOST_CHECK(r_pol0.extract<udrp_a>()->state = -42);
}

BOOST_AUTO_TEST_CASE(is_valid)
{
    r_policy p0;
    BOOST_CHECK(p0.is_valid());
    r_policy p1(std::move(p0));
    BOOST_CHECK(!p0.is_valid());
    p0 = r_policy{udrp_a{}};
    BOOST_CHECK(p0.is_valid());
    p1 = std::move(p0);
    BOOST_CHECK(!p0.is_valid());
    p0 = r_policy{udrp_a{}};
    BOOST_CHECK(p0.is_valid());
}

BOOST_AUTO_TEST_CASE(generic_assignment)
{
    r_policy p0;
    BOOST_CHECK(p0.is<fair_replace>());
    BOOST_CHECK(&(p0 = udrp_a{}) == &p0);
    BOOST_CHECK(p0.is_valid());
    BOOST_CHECK(p0.is<udrp_a>());
    p0 = udrp1{};
    BOOST_CHECK(p0.is<udrp1>());
    BOOST_CHECK((!std::is_assignable<r_policy, void>::value));
    BOOST_CHECK((!std::is_assignable<r_policy, int &>::value));
    BOOST_CHECK((!std::is_assignable<r_policy, const int &>::value));
    BOOST_CHECK((!std::is_assignable<r_policy, int &&>::value));
}

BOOST_AUTO_TEST_CASE(type_index)
{
    r_policy p0;
    BOOST_CHECK(p0.get_type_index() == std::type_index(typeid(fair_replace)));
    p0 = r_policy{udrp1{}};
    BOOST_CHECK(p0.get_type_index() == std::type_index(typeid(udrp1)));
}

BOOST_AUTO_TEST_CASE(get_ptr)
{
    r_policy p0;
    BOOST_CHECK(p0.get_ptr() == p0.extract<fair_replace>());
    BOOST_CHECK(static_cast<const r_policy &>(p0).get_ptr()
                == static_cast<const r_policy &>(p0).extract<fair_replace>());
    p0 = r_policy{udrp1{}};
    BOOST_CHECK(p0.get_ptr() == p0.extract<udrp1>());
    BOOST_CHECK(static_cast<const r_policy &>(p0).get_ptr() == static_cast<const r_policy &>(p0).extract<udrp1>());
}
