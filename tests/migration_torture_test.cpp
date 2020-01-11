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

#define BOOST_TEST_MODULE migration_torture_test
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <initializer_list>
#include <tuple>

#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/topologies/ring.hpp>

using namespace pagmo;

// A test to stress the migration machinery:
// do only 1 generation per evolve, 100 individuals
// per island, many evolves.
BOOST_AUTO_TEST_CASE(migration_torture_00)
{
    for (auto mt : {migration_type::p2p, migration_type::broadcast}) {
        for (auto mh : {migrant_handling::preserve, migrant_handling::evict}) {
            archipelago archi{ring{.8}, 20, de{1}, rosenbrock{100}, 100u};

            archi.set_migration_type(mt);
            archi.set_migrant_handling(mh);

            archi.evolve(500);

            // Add a few islands while evolving.
            for (auto i = 0; i < 20; ++i) {
                archi.push_back(de{1}, rosenbrock{100}, 100u);
            }

            for (auto i = 0; i < 20; ++i) {
                // Get out a few archi members while evolving.
                (void)archi.get_topology();
                (void)archi.get_migration_log();
                const auto mig_db = archi.get_migrants_db();
                BOOST_CHECK(mig_db.size() == 40u);
                for (const auto &mig_g : mig_db) {
                    BOOST_CHECK(std::get<0>(mig_g).size() == std::get<1>(mig_g).size());
                    BOOST_CHECK(std::get<0>(mig_g).size() == std::get<2>(mig_g).size());
                    for (decltype(std::get<0>(mig_g).size()) j = 0; j < std::get<0>(mig_g).size(); ++j) {
                        BOOST_CHECK(std::get<1>(mig_g)[j].size() == 100u);
                        BOOST_CHECK(std::get<2>(mig_g)[j].size() == 1u);
                    }
                }
            }

            BOOST_CHECK_NO_THROW(archi.wait_check());

            for (const auto &t : archi.get_migration_log()) {
                // Check timestamp.
                BOOST_CHECK(std::get<0>(t) >= 0.);
                // Check that source and destination islands are different.
                BOOST_CHECK(std::get<4>(t) != std::get<5>(t));
            }
        }
    }
}
