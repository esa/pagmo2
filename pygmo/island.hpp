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

#ifndef PYGMO_ISLAND_HPP
#define PYGMO_ISLAND_HPP

#include "python_includes.hpp"

#include <boost/python/extract.hpp>
#include <boost/python/object.hpp>
#include <memory>
#include <string>
#include <type_traits>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/threading.hpp>

#include "common_base.hpp"
#include "common_utils.hpp"

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

// Disable the static UDI checks for bp::object.
template <>
struct disable_udi_checks<bp::object> : std::true_type {
};

template <>
struct isl_inner<bp::object> final : isl_inner_base, pygmo::common_base {
    template <typename T>
    static void check_thread_safety(const T &x)
    {
        if (static_cast<int>(x.get_thread_safety()) < static_cast<int>(thread_safety::copyonly)) {
            pygmo_throw(PyExc_ValueError,
                        ("pythonic islands require objects which provide at least the copyonly thread "
                         "safety level, but the object '"
                         + x.get_name() + "' does not provide any thread safety guarantee")
                            .c_str());
        }
    }
    // Just need the def ctor, delete everything else.
    isl_inner() = default;
    isl_inner(const isl_inner &) = delete;
    isl_inner(isl_inner &&) = delete;
    isl_inner &operator=(const isl_inner &) = delete;
    isl_inner &operator=(isl_inner &&) = delete;
    explicit isl_inner(const bp::object &o)
    {
        check_not_type(o, "island");
        check_mandatory_method(o, "run_evolve", "island");
        m_value = pygmo::deepcopy(o);
    }
    virtual std::unique_ptr<isl_inner_base> clone() const override final
    {
        // This will make a deep copy using the ctor above.
        return make_unique<isl_inner>(m_value);
    }
    // Mandatory methods.
    virtual void run_evolve(algorithm &algo, ulock_t &algo_lock, population &pop, ulock_t &pop_lock) override final
    {
        // NOTE: run_evolve() is called from a separate thread in pagmo::island, need to construct a GTE before doing
        // anything with the interpreter (including the throws in the checks below).
        pygmo::gil_thread_ensurer gte;

        // NOTE: the idea of these checks is the following: we will have to copy algo and pop in order to invoke
        // the pythonic UDI's evolve method, which has a signature which is different from C++. If algo/prob are
        // bp::object, via the GTE above we have made sure we can safely copy them so we don't need to check anything.
        // Otherwise, we run the check, which is against the copyonly thread safety level as that is all we need.
        if (!algo.is<bp::object>()) {
            check_thread_safety(algo);
        }
        if (!pop.get_problem().is<bp::object>()) {
            check_thread_safety(pop.get_problem());
        }

        // Everything fine, copy algo/pop and unlock.
        bp::object algo_copy(algo);
        algo_lock.unlock();
        bp::object pop_copy(pop);
        pop_lock.unlock();

        // Invoke the run_evolve() method of the UDI and get out the evolved pop.
        // NOTE: here bp::extract will extract a copy of pop, and then a move ctor will take place,
        // which will just move the internal problem pointer. No additional thread safety guarantees are needed.
        population new_pop = bp::extract<population>(m_value.attr("run_evolve")(algo_copy, pop_copy));
        // Re-lock and assign.
        pop_lock.lock();
        pop = std::move(new_pop);
    }
    // Optional methods.
    virtual std::string get_name() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
    }
    virtual std::string get_extra_info() const override final
    {
        return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
    }
    bp::object m_value;
};
}
}

#endif
