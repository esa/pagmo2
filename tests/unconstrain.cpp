/* Copyright 2017 PaGMO development team

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

#define BOOST_TEST_MODULE unconstrain_test
#include <boost/test/included/unit_test.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <limits>
#include <stdexcept>
#include <string>

#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/problems/zdt.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;
BOOST_AUTO_TEST_CASE(unconstrain_construction_test)
{
    null_problem constrained_udp{2, 3, 4};
    // We test the default constructor
    BOOST_CHECK_NO_THROW(unconstrain{});
    BOOST_CHECK_NO_THROW(problem{unconstrain{}});
    // We test the constructor
    BOOST_CHECK_NO_THROW((problem{unconstrain{constrained_udp, "death penalty"}}));
    BOOST_CHECK_NO_THROW((problem{unconstrain{constrained_udp, "kuri"}}));
    BOOST_CHECK_NO_THROW((problem{unconstrain{constrained_udp, "weighted", vector_double(7, 1.)}}));
    BOOST_CHECK_NO_THROW((problem{unconstrain{constrained_udp, "ignore_c"}}));
    BOOST_CHECK_NO_THROW((problem{unconstrain{constrained_udp, "ignore_o"}}));

    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "death penalty"}}.get_nc()), 0u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "kuri"}}.get_nc()), 0u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "weighted", vector_double(7, 1.)}}.get_nc()), 0u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "ignore_c"}}.get_nc()), 0u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "ignore_o"}}.get_nc()), 0u);

    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "death penalty"}}.get_nobj()), 2u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "kuri"}}.get_nobj()), 2u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "weighted", vector_double(7, 1.)}}.get_nobj()), 2u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "ignore_c"}}.get_nobj()), 2u);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "ignore_o"}}.get_nobj()), 1u);

    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "death penalty"}}.has_gradient()), false);
    BOOST_CHECK_EQUAL((problem{unconstrain{constrained_udp, "death penalty"}}.has_hessians()), false);
    // We test the various throws
    BOOST_CHECK_THROW((unconstrain{null_problem{2, 0, 0}, "kuri"}), std::invalid_argument);
    BOOST_CHECK_THROW((unconstrain{null_problem{2, 3, 4}, "weighted", vector_double(6, 1.)}), std::invalid_argument);
    BOOST_CHECK_THROW((unconstrain{null_problem{2, 3, 4}, "mispelled"}), std::invalid_argument);
    BOOST_CHECK_THROW((unconstrain{null_problem{2, 3, 4}, "kuri", vector_double(3, 1.)}), std::invalid_argument);
}

struct my_udp {
    vector_double fitness(const vector_double &x) const
    {
        return x;
    }
    vector_double::size_type get_nobj() const
    {
        return 2u;
    }
    vector_double::size_type get_nec() const
    {
        return 2u;
    }
    vector_double::size_type get_nic() const
    {
        return 2u;
    }
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{-1, -1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 1}};
    }
    std::string get_name() const
    {
        return "a simple problem with constraint for testing";
    }
};

BOOST_AUTO_TEST_CASE(unconstrain_fitness_test)
{
    {
        unconstrain p0{my_udp{}, "death penalty"};
        // we test the method death penalty
        BOOST_CHECK(p0.fitness(vector_double(6, 0.)) == vector_double(2., 0.));
        BOOST_CHECK(p0.fitness(vector_double(6, 1.)) == vector_double(2., std::numeric_limits<double>::max()));
    }
    {
        unconstrain p0{my_udp{}, "kuri"};
        // we test the method kuri
        BOOST_CHECK(p0.fitness(vector_double(6, 0.)) == vector_double(2., 0.));
        BOOST_CHECK(p0.fitness(vector_double{0., 0., 1., 1., -1., 1.})
                    == vector_double(2, std::numeric_limits<double>::max() * (1. - 1. / 4.)));
        BOOST_CHECK(p0.fitness(vector_double{0., 0., 1., 1., -1., -1.})
                    == vector_double(2, std::numeric_limits<double>::max() * (1. - 2. / 4.)));
        BOOST_CHECK(p0.fitness(vector_double{0., 0., 0., 1., 0., 0.})
                    == vector_double(2, std::numeric_limits<double>::max() * (1. - 3. / 4.)));
    }
    {
        unconstrain p0{my_udp{}, "weighted", vector_double(4, 1.)};
        BOOST_CHECK(p0.fitness(vector_double(6, 0.)) == vector_double(2., 0.));
        BOOST_CHECK(p0.fitness(vector_double{0., 0., 1., 1., -1., 1.}) == vector_double(2, 3.));
        BOOST_CHECK(p0.fitness(vector_double{0., 0., 1., 1., -1., -1.}) == vector_double(2, 2.));
        BOOST_CHECK(p0.fitness(vector_double{0., 0., 0., 1., 0., 0.}) == vector_double(2, 1.));
    }
    {
        unconstrain p0{my_udp{}, "ignore_c"};
        BOOST_CHECK(p0.fitness(vector_double(6, 0.)) == vector_double(2., 0.));
        BOOST_CHECK((p0.fitness(vector_double{1., 2., 1., 1., -1., 1.}) == vector_double{1., 2.}));
        BOOST_CHECK((p0.fitness(vector_double{3., 4., 1., 1., -1., -1.}) == vector_double{3., 4.}));
        BOOST_CHECK((p0.fitness(vector_double{5., 6., 0., 1., 0., 0.}) == vector_double{5., 6.}));
    }
    {
        unconstrain p0{my_udp{}, "ignore_o"};
        BOOST_CHECK(p0.fitness(vector_double(6, 0.)) == vector_double(1., 0.));
        BOOST_CHECK((p0.fitness(vector_double{1., 2., 1., 0., -1., -1.}) == vector_double{1.}));
        BOOST_CHECK_CLOSE(p0.fitness(vector_double{1., 2., 1., 1., -1., -1.})[0], std::sqrt(2.), 1e-8);
        BOOST_CHECK_CLOSE(p0.fitness(vector_double{1., 2., 1., 1., 1., -1.})[0], std::sqrt(2.) + std::sqrt(1.), 1e-8);
        BOOST_CHECK_CLOSE(p0.fitness(vector_double{1., 2., 1., 1., 1., 2.})[0], std::sqrt(2.) + std::sqrt(5.), 1e-8);
    }
}