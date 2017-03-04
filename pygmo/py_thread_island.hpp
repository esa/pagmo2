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

#ifndef PYGMO_PY_THREAD_ISLAND_HPP
#define PYGMO_PY_THREAD_ISLAND_HPP

#include "python_includes.hpp"

#include <pagmo/island.hpp>

#include "common_utils.hpp"

namespace pygmo
{

// NOTE: we need to re-implement the methods involving wait() in thread_island. The reason is the following.
// Boost.Python locks the GIL when crossing the boundary from Python into C++. So, if we call wait() from Python,
// BP will lock the GIL and then we will be waiting for evolutions in the island to finish. During this time, no
// Python code will be executed because the GIL is locked. This means that if we have a Python thread doing background
// work (e.g., managing the task queue in pythonic islands), it will have to wait before doing any progress. By
// unlocking the GIL before calling thread_island::wait(), we give the chance to other Python threads to continue
// doing some work.
// NOTE: here we are re-implementing all methods that call wait(), either directly or indirectly. An alternative
// solution would be to make thread_island()::wait() virtual. This is probably the way to go if we have more methods
// using wait(), but for the moment we have just 2.
struct py_thread_island : pagmo::thread_island {
    using pagmo::thread_island::thread_island;
    void wait() const
    {
        // NOTE: here we have 2 RAII classes interacting with the GIL. The GIL releaser is the *second* one,
        // and it is the one that is responsible for unlocking the Python interpreter while wait() is running.
        // The *first* one, the GIL thread ensurer, does something else: it makes sure that we can call the Python
        // interpreter from the current C++ thread. In a normal situation, in which islands are just instantiated
        // from the main thread, the gte object is superfluous. However, if we are interacting with islands from a
        // separate C++ thread, then we need to make sure that every time we call into the Python interpreter (e.g., by
        // using the GIL releaser below) we inform Python we are about to call from a separate thread. This is what
        // the GTE object does. This use case is, for instance, what happens with the PADE algorithm when, algo, prob,
        // etc. are all C++ objects (when at least one object is pythonic, we will not end up using the thread island).
        gil_thread_ensurer gte;
        gil_releaser gr;
        static_cast<const pagmo::thread_island *>(this)->wait();
    }
    ~py_thread_island()
    {
        wait();
    }
};
}

#endif
