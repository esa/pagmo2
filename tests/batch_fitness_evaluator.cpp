/* Copyright 2017-2018 PaGMO development team

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

#define BOOST_TEST_MODULE bfe_test
#include <boost/test/included/unit_test.hpp>

#include <sstream>
#include <string>

#include <boost/lexical_cast.hpp>

#include <pagmo/batch_evaluators/batch_fitness_evaluator.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/serialization.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

using udbfe_func_t = vector_double (*)(const problem &, const vector_double &);

inline vector_double udbfe0(const problem &, const vector_double &)
{
    return vector_double{};
}

BOOST_AUTO_TEST_CASE(basic_tests)
{
    batch_fitness_evaluator bfe0;

    // Public methods.
    BOOST_CHECK(bfe0.extract<default_bfe>() != nullptr);
    BOOST_CHECK(bfe0.extract<udbfe_func_t>() == nullptr);
    BOOST_CHECK(static_cast<const batch_fitness_evaluator &>(bfe0).extract<default_bfe>() != nullptr);
    BOOST_CHECK(static_cast<const batch_fitness_evaluator &>(bfe0).extract<udbfe_func_t>() == nullptr);
    BOOST_CHECK(bfe0.is<default_bfe>());
    BOOST_CHECK(!bfe0.is<udbfe_func_t>());
    BOOST_CHECK(bfe0.get_name() == "Default batch fitness evaluator");
    BOOST_CHECK(bfe0.get_extra_info().empty());
    BOOST_CHECK(bfe0.get_thread_safety() == thread_safety::basic);

    // Constructors, assignments.
    batch_fitness_evaluator bfe1{udbfe0};
    BOOST_CHECK(bfe1.is<udbfe_func_t>());
    BOOST_CHECK(*bfe1.extract<udbfe_func_t>() == udbfe0);

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
            cereal::JSONOutputArchive oarchive(ss);
            oarchive(bfe0);
        }
        bfe0 = batch_fitness_evaluator{udbfe0};
        BOOST_CHECK(bfe0.is<udbfe_func_t>());
        BOOST_CHECK(before != boost::lexical_cast<std::string>(bfe0));
        {
            cereal::JSONInputArchive iarchive(ss);
            iarchive(bfe0);
        }
        BOOST_CHECK(before == boost::lexical_cast<std::string>(bfe0));
        BOOST_CHECK(bfe0.is<default_bfe>());
    }
}
