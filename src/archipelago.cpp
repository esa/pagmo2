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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/archipelago.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/island.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

// NOTE: same utility method as in pagmo::island, see there.
void archipelago::wait_check_ignore()
{
    try {
        wait_check();
    } catch (...) {
    }
}

/// Default constructor.
/**
 * The default constructor will initialise an empty archipelago.
 */
archipelago::archipelago() {}

/// Copy constructor.
/**
 * The islands of \p other will be copied into \p this via archipelago::push_back().
 *
 * @param other the archipelago that will be copied.
 *
 * @throws unspecified any exception thrown by archipelago::push_back().
 */
archipelago::archipelago(const archipelago &other)
{
    for (const auto &iptr : other.m_islands) {
        // This will end up copying the island members,
        // and assign the archi pointer as well.
        push_back(*iptr);
    }
}

/// Move constructor.
/**
 * The move constructor will wait for any ongoing evolution in \p other to finish
 * and it will then transfer the state of \p other into \p this. After the move,
 * \p other is left in an unspecified but valid state.
 *
 * @param other the archipelago that will be moved.
 */
archipelago::archipelago(archipelago &&other) noexcept
{
    // NOTE: in move operations we have to wait, because the ongoing
    // island evolutions are interacting with their hosting archi 'other'.
    // We cannot just move in the vector of islands.
    other.wait_check_ignore();
    // Move in the islands.
    m_islands = std::move(other.m_islands);
    // Re-direct the archi pointers to point to this.
    for (const auto &iptr : m_islands) {
        iptr->m_ptr->archi_ptr = this;
    }
}

/// Copy assignment.
/**
 * Copy assignment is implemented as copy construction followed by a move assignment.
 *
 * @param other the assignment argument.
 *
 * @return a reference to \p this.
 *
 * @throws unspecified any exception thrown by the copy constructor.
 */
archipelago &archipelago::operator=(const archipelago &other)
{
    if (this != &other) {
        *this = archipelago(other);
    }
    return *this;
}

/// Move assignment.
/**
 * Move assignment will transfer the state of \p other into \p this, after any ongoing
 * evolution in \p this and \p other has finished.
 *
 * @param other the assignment argument.
 *
 * @return a reference to \p this.
 */
archipelago &archipelago::operator=(archipelago &&other) noexcept
{
    if (this != &other) {
        // NOTE: as in the move ctor, we need to wait on other and this as well.
        // This mirrors the island's behaviour.
        wait_check_ignore();
        other.wait_check_ignore();
        // Move in the islands.
        m_islands = std::move(other.m_islands);
        // Re-direct the archi pointers to point to this.
        for (const auto &iptr : m_islands) {
            iptr->m_ptr->archi_ptr = this;
        }
    }
    return *this;
}

/// Destructor.
/**
 * The destructor will call archipelago::wait_check() internally, ignoring any exception that might be thrown,
 * and run checks in debug mode.
 */
archipelago::~archipelago()
{
    // NOTE: this is not strictly necessary, but it will not hurt. And, if we add further
    // sanity checks, we know the archi is stopped.
    wait_check_ignore();
    assert(std::all_of(m_islands.begin(), m_islands.end(),
                       [this](const std::unique_ptr<island> &iptr) { return iptr->m_ptr->archi_ptr == this; }));
}

/// Mutable island access.
/**
 * This subscript operator can be used to access the <tt>i</tt>-th island of the archipelago (that is,
 * the <tt>i</tt>-th island that was inserted via push_back()). References returned by this method are valid even
 * after a push_back() invocation. Assignment and destruction of the archipelago will invalidate island references
 * obtained via this method.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. note::
 *
 *    The mutable version of the subscript operator exists solely to allow calling non-const methods
 *    on the islands. Assigning an island via a reference obtained through this operator will result
 *    in undefined behaviour.
 *
 * \endverbatim
 *
 * @param i the index of the island to be accessed.
 *
 * @return a reference to the <tt>i</tt>-th island of the archipelago.
 *
 * @throws std::out_of_range if \p i is not less than the size of the archipelago.
 */
island &archipelago::operator[](size_type i)
{
    if (i >= size()) {
        pagmo_throw(std::out_of_range, "cannot access the island at index " + std::to_string(i)
                                           + ": the archipelago has a size of only " + std::to_string(size()));
    }
    return *m_islands[i];
}

/// Const island access.
/**
 * This subscript operator can be used to access the <tt>i</tt>-th island of the archipelago (that is,
 * the <tt>i</tt>-th island that was inserted via push_back()). References returned by this method are valid even
 * after a push_back() invocation. Assignment and destruction of the archipelago will invalidate island references
 * obtained via this method.
 *
 * @param i the index of the island to be accessed.
 *
 * @return a const reference to the <tt>i</tt>-th island of the archipelago.
 *
 * @throws std::out_of_range if \p i is not less than the size of the archipelago.
 */
const island &archipelago::operator[](size_type i) const
{
    if (i >= size()) {
        pagmo_throw(std::out_of_range, "cannot access the island at index " + std::to_string(i)
                                           + ": the archipelago has a size of only " + std::to_string(size()));
    }
    return *m_islands[i];
}

void archipelago::evolve(unsigned n)
{
    for (auto &iptr : m_islands) {
        iptr->evolve(n);
    }
}

/// Block until all evolutions have finished.
/**
 * This method will call island::wait() on all the islands of the archipelago. Exceptions thrown by island
 * evolutions can be re-raised via wait_check(): they will **not** be re-thrown by this method. Also, contrary to
 * wait_check(), this method will **not** reset the status of the archipelago: after a call to wait(), status() will
 * always return either evolve_status::idle or evolve_status::idle_error.
 */
void archipelago::wait() noexcept
{
    for (const auto &iptr : m_islands) {
        iptr->wait();
    }
}

/// Block until all evolutions have finished and raise the first exception that was encountered.
/**
 * This method will call island::wait_check() on all the islands of the archipelago (following
 * the order in which the islands were inserted into the archipelago).
 * The first exception raised by island::wait_check() will be re-raised by this method,
 * and all the exceptions thrown by the other calls to island::wait_check() will be ignored.
 *
 * Note that wait_check() resets the status of the archipelago: after a call to wait_check(), status() will always
 * return evolve_status::idle.
 *
 * @throws unspecified any exception thrown by any evolution task queued in the archipelago's
 * islands.
 */
void archipelago::wait_check()
{
    for (auto it = m_islands.begin(); it != m_islands.end(); ++it) {
        try {
            (*it)->wait_check();
        } catch (...) {
            for (it = it + 1; it != m_islands.end(); ++it) {
                (*it)->wait_check_ignore();
            }
            throw;
        }
    }
}

/// Status of the archipelago.
/**
 * This method will return a pagmo::evolve_status flag indicating the current status of
 * asynchronous operations in the archipelago. The flag will be:
 *
 * * evolve_status::idle if, for all the islands in the archipelago, island::status() returns
 *   evolve_status::idle;
 * * evolve_status::busy if, for at least one island in the archipelago, island::status() returns
 *   evolve_status::busy, and for no island island::status() returns an error status;
 * * evolve_status::idle_error if no island in the archipelago is busy and for at least one island
 *   island::status() returns evolve_status::idle_error;
 * * evolve_status::busy_error if, for at least one island in the archipelago, island::status() returns
 *   an error status and at least one island is busy.
 *
 * Note that after a call to wait_check(), status() will always return evolve_status::idle,
 * and after a call to wait(), status() will always return either evolve_status::idle or
 * evolve_status::idle_error.
 *
 * @return a flag indicating the current status of asynchronous operations in the archipelago.
 */
evolve_status archipelago::status() const
{
    decltype(m_islands.size()) n_idle = 0, n_busy = 0, n_idle_error = 0, n_busy_error = 0;
    for (const auto &iptr : m_islands) {
        switch (iptr->status()) {
            case evolve_status::idle:
                ++n_idle;
                break;
            case evolve_status::busy:
                ++n_busy;
                break;
            case evolve_status::idle_error:
                ++n_idle_error;
                break;
            case evolve_status::busy_error:
                ++n_busy_error;
                break;
        }
    }

    // If we have at last one busy error, the global state
    // is also busy error.
    if (n_busy_error) {
        return evolve_status::busy_error;
    }

    // The other error case.
    if (n_idle_error) {
        if (n_busy) {
            // At least one island is idle with error. If any other
            // island is busy, we return busy error.
            return evolve_status::busy_error;
        }
        // No island is busy at all, at least one island is idle with error.
        return evolve_status::idle_error;
    }

    // No error in any island. If they are all idle, the global state is idle,
    // otherwise busy.
    return n_idle == m_islands.size() ? evolve_status::idle : evolve_status::busy;
}

/// Get the fitness vectors of the islands' champions.
/**
 * @return a collection of the fitness vectors of the islands' champions.
 *
 * @throws unspecified any exception thrown by population::champion_f() or
 * by memory errors in standard containers.
 */
std::vector<vector_double> archipelago::get_champions_f() const
{
    std::vector<vector_double> retval;
    for (const auto &isl_ptr : m_islands) {
        retval.emplace_back(isl_ptr->get_population().champion_f());
    }
    return retval;
}

/// Get the decision vectors of the islands' champions.
/**
 * @return a collection of the decision vectors of the islands' champions.
 *
 * @throws unspecified any exception thrown by population::champion_x() or
 * by memory errors in standard containers.
 */
std::vector<vector_double> archipelago::get_champions_x() const
{
    std::vector<vector_double> retval;
    for (const auto &isl_ptr : m_islands) {
        retval.emplace_back(isl_ptr->get_population().champion_x());
    }
    return retval;
}

/// Stream operator.
/**
 * This operator will stream to \p os a human-readable representation of the input
 * archipelago \p archi.
 *
 * @param os the target stream.
 * @param archi the archipelago that will be streamed.
 *
 * @return a reference to \p os.
 *
 * @throws unspecified any exception thrown by:
 * - the streaming of primitive types,
 * - island::get_algorithm(), island::get_population().
 */
std::ostream &operator<<(std::ostream &os, const archipelago &archi)
{
    stream(os, "Number of islands: ", archi.size(), "\n");
    stream(os, "Status: ", archi.status(), "\n\n");
    stream(os, "Islands summaries:\n\n");
    detail::table t({"#", "Type", "Algo", "Prob", "Size", "Status"}, "\t");
    for (decltype(archi.size()) i = 0; i < archi.size(); ++i) {
        const auto pop = archi[i].get_population();
        t.add_row(i, archi[i].get_name(), archi[i].get_algorithm().get_name(), pop.get_problem().get_name(), pop.size(),
                  archi[i].status());
    }
    stream(os, t);
    return os;
}

} // namespace pagmo
