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

#define BOOST_TEST_MODULE bfe_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <functional>
#include <iostream>
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

#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/type_name.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

using udbfe_func_t = vector_double (*)(const problem &, const vector_double &);

inline vector_double udbfe0(const problem &p, const vector_double &dvs)
{
    return vector_double(p.get_nf() * (dvs.size() / p.get_nx()), .5);
}

BOOST_AUTO_TEST_CASE(type_traits_tests)
{
    BOOST_CHECK(is_udbfe<default_bfe>::value);
    BOOST_CHECK(!is_udbfe<const default_bfe>::value);
    BOOST_CHECK(!is_udbfe<default_bfe &>::value);
    BOOST_CHECK(!is_udbfe<const default_bfe &>::value);

    BOOST_CHECK(is_udbfe<decltype(&udbfe0)>::value);
    BOOST_CHECK(is_udbfe<udbfe_func_t>::value);

    struct non_udbfe_00 {
    };
    BOOST_CHECK(!is_udbfe<non_udbfe_00>::value);
    BOOST_CHECK(!has_bfe_call_operator<non_udbfe_00>::value);

    struct non_udbfe_01 {
        vector_double operator()();
    };
    BOOST_CHECK(!is_udbfe<non_udbfe_01>::value);
    BOOST_CHECK(!has_bfe_call_operator<non_udbfe_01>::value);

    struct non_udbfe_02 {
        // NOTE: non-const operator.
        vector_double operator()(const problem &, const vector_double &);
    };
    BOOST_CHECK(!is_udbfe<non_udbfe_02>::value);
    BOOST_CHECK(!has_bfe_call_operator<non_udbfe_02>::value);

    struct non_udbfe_03 {
        // NOTE: not def ctible.
        non_udbfe_03() = delete;
        vector_double operator()(const problem &, const vector_double &) const;
    };
    BOOST_CHECK(!is_udbfe<non_udbfe_03>::value);
    BOOST_CHECK(has_bfe_call_operator<non_udbfe_03>::value);

    BOOST_CHECK(is_udbfe<decltype(&udbfe0)>::value);
    struct udbfe_00 {
        vector_double operator()(const problem &, const vector_double &) const;
    };
    BOOST_CHECK(is_udbfe<udbfe_00>::value);

    // Test std::function as well.
    BOOST_CHECK(is_udbfe<std::function<vector_double(const problem &, const vector_double &)>>::value);
}

struct udbfe1 {
    vector_double operator()(const problem &, const vector_double &) const
    {
        return vector_double{};
    }
    std::string foo = "hello world";
};

struct udbfe2 {
    udbfe2() = default;
    udbfe2(const udbfe2 &other) : foo{new std::string{*other.foo}} {}
    udbfe2(udbfe2 &&) = default;
    vector_double operator()(const problem &, const vector_double &) const
    {
        return vector_double{};
    }
    std::string get_name() const
    {
        return "frobniz";
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::constant;
    }
    std::unique_ptr<std::string> foo = std::unique_ptr<std::string>{new std::string{"hello world"}};
};

BOOST_AUTO_TEST_CASE(basic_tests)
{
    bfe bfe0;
    problem p;

    // Public methods.
    BOOST_CHECK(bfe0.extract<default_bfe>() != nullptr);
    BOOST_CHECK(bfe0.extract<udbfe_func_t>() == nullptr);
    BOOST_CHECK(static_cast<const bfe &>(bfe0).extract<default_bfe>() != nullptr);
    BOOST_CHECK(static_cast<const bfe &>(bfe0).extract<udbfe_func_t>() == nullptr);
    BOOST_CHECK(bfe0.is<default_bfe>());
    BOOST_CHECK(!bfe0.is<udbfe_func_t>());
    BOOST_CHECK(bfe0.get_name() == "Default batch fitness evaluator");
    BOOST_CHECK(bfe{udbfe1{}}.get_name() == detail::type_name<udbfe1>());
    BOOST_CHECK(bfe0.get_extra_info().empty());
    BOOST_CHECK(bfe0.get_thread_safety() == thread_safety::basic);

    // Constructors, assignments.
    bfe bfe1{udbfe0};
    BOOST_CHECK(bfe1.is<udbfe_func_t>());
    BOOST_CHECK(*bfe1.extract<udbfe_func_t>() == udbfe0);
    // Generic constructor with copy.
    udbfe1 b1;
    bfe bfe2{b1};
    BOOST_CHECK(b1.foo == "hello world");
    BOOST_CHECK(bfe2.extract<udbfe1>()->foo == "hello world");
    // Generic constructor with move.
    udbfe2 b2;
    bfe bfe3{std::move(b2)};
    BOOST_CHECK(b2.foo.get() == nullptr);
    BOOST_CHECK(bfe3.extract<udbfe2>()->foo.get() != nullptr);
    BOOST_CHECK(*bfe3.extract<udbfe2>()->foo == "hello world");
    // Copy constructor.
    udbfe2 b3;
    bfe bfe4{b3}, bfe5{bfe4};
    BOOST_CHECK(*bfe5.extract<udbfe2>()->foo == "hello world");
    BOOST_CHECK(bfe5.extract<udbfe2>()->foo.get() != bfe4.extract<udbfe2>()->foo.get());
    BOOST_CHECK(bfe5(p, vector_double{}) == vector_double{});
    BOOST_CHECK(bfe5.get_name() == "frobniz");
    BOOST_CHECK(bfe5.get_thread_safety() == thread_safety::constant);
    // Move constructor.
    bfe bfe6{std::move(bfe5)};
    BOOST_CHECK(*bfe6.extract<udbfe2>()->foo == "hello world");
    BOOST_CHECK(bfe6(p, vector_double{}) == vector_double{});
    BOOST_CHECK(bfe6.get_name() == "frobniz");
    BOOST_CHECK(bfe6.get_thread_safety() == thread_safety::constant);
    // Revive bfe5 via copy assignment.
    bfe5 = bfe6;
    BOOST_CHECK(*bfe5.extract<udbfe2>()->foo == "hello world");
    BOOST_CHECK(bfe5(p, vector_double{}) == vector_double{});
    BOOST_CHECK(bfe5.get_name() == "frobniz");
    BOOST_CHECK(bfe5.get_thread_safety() == thread_safety::constant);
    // Revive bfe5 via move assignment.
    bfe bfe7{std::move(bfe5)};
    bfe5 = std::move(bfe6);
    BOOST_CHECK(*bfe5.extract<udbfe2>()->foo == "hello world");
    BOOST_CHECK(bfe5(p, vector_double{}) == vector_double{});
    BOOST_CHECK(bfe5.get_name() == "frobniz");
    BOOST_CHECK(bfe5.get_thread_safety() == thread_safety::constant);
    // Self move-assignment.
    bfe5 = std::move(*&bfe5);
    BOOST_CHECK(*bfe5.extract<udbfe2>()->foo == "hello world");
    BOOST_CHECK(bfe5(p, vector_double{}) == vector_double{});
    BOOST_CHECK(bfe5.get_name() == "frobniz");
    BOOST_CHECK(bfe5.get_thread_safety() == thread_safety::constant);

    // Minimal iostream test.
    {
        std::ostringstream oss;
        oss << bfe0;
        BOOST_CHECK(!oss.str().empty());
    }

    // Minimal serialization test.
    {
        std::string before;
        std::stringstream ss;
        {
            before = boost::lexical_cast<std::string>(bfe0);
            boost::archive::binary_oarchive oarchive(ss);
            oarchive << bfe0;
        }
        bfe0 = bfe{udbfe0};
        BOOST_CHECK(bfe0.is<udbfe_func_t>());
        BOOST_CHECK(before != boost::lexical_cast<std::string>(bfe0));
        {
            boost::archive::binary_iarchive iarchive(ss);
            iarchive >> bfe0;
        }
        BOOST_CHECK(before == boost::lexical_cast<std::string>(bfe0));
        BOOST_CHECK(bfe0.is<default_bfe>());
    }
}

BOOST_AUTO_TEST_CASE(optional_tests)
{
    // get_name().
    struct udbfe_00 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
        std::string get_name() const
        {
            return "frobniz";
        }
    };
    BOOST_CHECK_EQUAL(bfe{udbfe_00{}}.get_name(), "frobniz");
    struct udbfe_01 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
        // Missing const.
        std::string get_name()
        {
            return "frobniz";
        }
    };
    BOOST_CHECK(bfe{udbfe_01{}}.get_name() != "frobniz");

    // get_extra_info().
    struct udbfe_02 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
        std::string get_extra_info() const
        {
            return "frobniz";
        }
    };
    BOOST_CHECK_EQUAL(bfe{udbfe_02{}}.get_extra_info(), "frobniz");
    struct udbfe_03 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
        // Missing const.
        std::string get_extra_info()
        {
            return "frobniz";
        }
    };
    BOOST_CHECK(bfe{udbfe_03{}}.get_extra_info().empty());

    // get_thread_safety().
    struct udbfe_04 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
        thread_safety get_thread_safety() const
        {
            return thread_safety::constant;
        }
    };
    BOOST_CHECK_EQUAL(bfe{udbfe_04{}}.get_thread_safety(), thread_safety::constant);
    struct udbfe_05 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
        // Missing const.
        thread_safety get_thread_safety()
        {
            return thread_safety::constant;
        }
    };
    BOOST_CHECK_EQUAL(bfe{udbfe_05{}}.get_thread_safety(), thread_safety::basic);
}

BOOST_AUTO_TEST_CASE(stream_operator)
{
    struct udbfe_00 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
    };
    {
        std::ostringstream oss;
        oss << bfe{udbfe_00{}};
        BOOST_CHECK(!oss.str().empty());
    }
    struct udbfe_01 {
        vector_double operator()(const problem &, const vector_double &) const
        {
            return vector_double{};
        }
        std::string get_extra_info() const
        {
            return "bartoppo";
        }
    };
    {
        std::ostringstream oss;
        oss << bfe{udbfe_01{}};
        const auto st = oss.str();
        BOOST_CHECK(boost::contains(st, "bartoppo"));
        BOOST_CHECK(boost::contains(st, "Extra info:"));
    }
    std::cout << bfe{} << '\n';
}

BOOST_AUTO_TEST_CASE(call_operator)
{
    struct udbfe_00 {
        vector_double operator()(const problem &p, const vector_double &dvs) const
        {
            return vector_double(p.get_nf() * (dvs.size() / p.get_nx()), 1.);
        }
    };
    bfe bfe0{udbfe_00{}};
    BOOST_CHECK(bfe0(problem{}, vector_double{.5}) == vector_double{1.});
    BOOST_CHECK(bfe0(problem{null_problem{3}}, vector_double{.5}) == (vector_double{1., 1., 1.}));
    // Try with a function.
    bfe bfe0a{udbfe0};
    BOOST_CHECK(bfe0a(problem{null_problem{3}}, vector_double{.5}) == (vector_double{.5, .5, .5}));
    // Try passing in a wrong dvs.
    BOOST_CHECK_EXCEPTION(
        bfe0(problem{rosenbrock{}}, vector_double{.5}), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(
                ia.what(),
                "Invalid argument for a batch fitness evaluation: the length of the vector "
                "representing the decision vectors, 1, is not an exact multiple of the dimension of the problem, 2");
        });
    // Try a udfbe which returns a bogus vector of fitnesses.
    struct udbfe_01 {
        vector_double operator()(const problem &p, const vector_double &dvs) const
        {
            return vector_double(p.get_nf() * (dvs.size() / p.get_nx()) + 1u, 1.);
        }
    };
    bfe bfe1{udbfe_01{}};
    BOOST_CHECK_EXCEPTION(
        bfe1(problem{null_problem{3}}, vector_double{.5}), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(),
                                   "An invalid result was produced by a batch fitness evaluation: the length of "
                                   "the vector representing the fitness vectors, 4, is not an exact multiple of "
                                   "the fitness dimension of the problem, 3");
        });
    // Try a udfbe which returns a bogus number of fitnesses.
    struct udbfe_02 {
        vector_double operator()(const problem &p, const vector_double &dvs) const
        {

            return vector_double(p.get_nf() * ((dvs.size() + 1u) / p.get_nx()), 1.);
        }
    };
    bfe bfe2{udbfe_02{}};
    BOOST_CHECK_EXCEPTION(
        bfe2(problem{null_problem{3}}, vector_double{.5}), std::invalid_argument, [](const std::invalid_argument &ia) {
            return boost::contains(ia.what(),
                                   "An invalid result was produced by a batch fitness evaluation: the number of "
                                   "produced fitness vectors, 2, differs from the number of input decision vectors, 1");
        });
}

struct udbfe_a {
    vector_double operator()(const problem &p, const vector_double &dvs) const
    {
        return vector_double(p.get_nf() * (dvs.size() / p.get_nx()), 1.);
    }
    std::string get_name() const
    {
        return "abba";
    }
    std::string get_extra_info() const
    {
        return "dabba";
    }
    thread_safety get_thread_safety() const
    {
        return thread_safety::constant;
    }
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &state;
    }
    int state = 42;
};

PAGMO_S11N_BFE_EXPORT(udbfe_a)

// Serialization tests.
BOOST_AUTO_TEST_CASE(s11n)
{
    bfe bfe0{udbfe_a{}};
    BOOST_CHECK(bfe0.extract<udbfe_a>()->state == 42);
    bfe0.extract<udbfe_a>()->state = -42;
    // Store the string representation.
    std::stringstream ss;
    auto before = boost::lexical_cast<std::string>(bfe0);
    // Now serialize, deserialize and compare the result.
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << bfe0;
    }
    // Change the content of p before deserializing.
    bfe0 = bfe{};
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> bfe0;
    }
    auto after = boost::lexical_cast<std::string>(bfe0);
    BOOST_CHECK_EQUAL(before, after);
    BOOST_CHECK(bfe0.is<udbfe_a>());
    BOOST_CHECK(bfe0.extract<udbfe_a>()->state = -42);
}

BOOST_AUTO_TEST_CASE(lambda_std_function)
{
    auto fun = [](const problem &p, const vector_double &dvs) {
        return vector_double(p.get_nf() * (dvs.size() / p.get_nx()), 1.);
    };
    BOOST_CHECK(!is_udbfe<decltype(fun)>::value);
#if !defined(_MSC_VER)
    BOOST_CHECK(is_udbfe<decltype(+fun)>::value);
#endif
    auto stdfun = std::function<vector_double(const problem &, const vector_double &)>(fun);
    BOOST_CHECK(is_udbfe<decltype(stdfun)>::value);

#if !defined(_MSC_VER)
    {
        bfe bfe0{+fun};
        BOOST_CHECK(bfe0(problem{}, vector_double{.5}) == vector_double{1.});
        BOOST_CHECK(bfe0(problem{null_problem{3}}, vector_double{.5}) == (vector_double{1., 1., 1.}));
    }
#endif

    {
        bfe bfe0{stdfun};
        BOOST_CHECK(bfe0(problem{}, vector_double{.5}) == vector_double{1.});
        BOOST_CHECK(bfe0(problem{null_problem{3}}, vector_double{.5}) == (vector_double{1., 1., 1.}));
    }
}

BOOST_AUTO_TEST_CASE(is_valid)
{
    bfe p0;
    BOOST_CHECK(p0.is_valid());
    bfe p1(std::move(p0));
    BOOST_CHECK(!p0.is_valid());
    p0 = bfe{udbfe_a{}};
    BOOST_CHECK(p0.is_valid());
    p1 = std::move(p0);
    BOOST_CHECK(!p0.is_valid());
    p0 = bfe{udbfe_a{}};
    BOOST_CHECK(p0.is_valid());
}

BOOST_AUTO_TEST_CASE(generic_assignment)
{
    bfe p0;
    BOOST_CHECK(p0.is<default_bfe>());
    BOOST_CHECK(&(p0 = udbfe_a{}) == &p0);
    BOOST_CHECK(p0.is_valid());
    BOOST_CHECK(p0.is<udbfe_a>());
    p0 = udbfe0;
    BOOST_CHECK(p0.is<udbfe_func_t>());
    p0 = &udbfe0;
    BOOST_CHECK(p0.is<udbfe_func_t>());
    BOOST_CHECK((!std::is_assignable<bfe, void>::value));
    BOOST_CHECK((!std::is_assignable<bfe, int &>::value));
    BOOST_CHECK((!std::is_assignable<bfe, const int &>::value));
    BOOST_CHECK((!std::is_assignable<bfe, int &&>::value));
}

BOOST_AUTO_TEST_CASE(type_index)
{
    bfe p0;
    BOOST_CHECK(p0.get_type_index() == std::type_index(typeid(default_bfe)));
    p0 = bfe{udbfe1{}};
    BOOST_CHECK(p0.get_type_index() == std::type_index(typeid(udbfe1)));
}

BOOST_AUTO_TEST_CASE(get_ptr)
{
    bfe p0;
    BOOST_CHECK(p0.get_ptr() == p0.extract<default_bfe>());
    BOOST_CHECK(static_cast<const bfe &>(p0).get_ptr() == static_cast<const bfe &>(p0).extract<default_bfe>());
    p0 = bfe{udbfe1{}};
    BOOST_CHECK(p0.get_ptr() == p0.extract<udbfe1>());
    BOOST_CHECK(static_cast<const bfe &>(p0).get_ptr() == static_cast<const bfe &>(p0).extract<udbfe1>());
}
