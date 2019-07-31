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
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/archipelago.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/io.hpp>
#include <pagmo/island.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

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
 * \verbatim embed:rst:leading-asterisk
 * The default constructor will initialise an empty archipelago with a
 * default-constructed (i.e., :cpp:class:`~pagmo::unconnected`) topology,
 * a point-to-point :cpp:enum:`~pagmo::migration_type` and a
 * preserve :cpp:enum:`~pagmo::migrant_handling` policy.
 * \endverbatim
 */
archipelago::archipelago()
    : m_migr_type(migration_type::p2p),           // Default: point-to-point migration type.
      m_migr_handling(migrant_handling::preserve) // Default: preserve migrants.
{
}

/// Copy constructor.
/**
 * The copy constructor will perform a deep copy of \p other.
 *
 * @param other the archipelago that will be copied.
 *
 * @throws unspecified any exception thrown by the public interface
 * of pagmo::archipelago.
 */
archipelago::archipelago(const archipelago &other)
{
    for (const auto &iptr : other.m_islands) {
        // This will end up copying the island members,
        // and assign the archi pointer, and associating ids to island pointers.
        push_back(*iptr);
    }

    // Set the migrants.
    m_migrants = other.get_migrants_db();

    // Set the migration log.
    m_migr_log = other.get_migration_log();

    // Set the topology.
    m_topology = other.get_topology();

    // Migration type and migrant handling policy.
    m_migr_type.store(other.m_migr_type.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_migr_handling.store(other.m_migr_handling.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

/// Move constructor.
/**
 * The move constructor will wait for any ongoing evolution in \p other to finish
 * and it will then transfer the state of \p other into \p this. After the move,
 * \p other is left in a state which is assignable and destructible.
 *
 * @param other the archipelago that will be moved.
 */
archipelago::archipelago(archipelago &&other) noexcept
{
    // NOTE: in move operations we have to wait, because the ongoing
    // island evolutions are interacting with their hosting archi 'other'.
    // We cannot just move in the vector of islands.
    // NOTE: we want to ensure that other is in a known state
    // after the move, so that we can run assertion checks in
    // the destructor in debug mode.
    other.wait_check_ignore();

    // Move in the islands, make sure that other is cleared.
    m_islands = std::move(other.m_islands);
    other.m_islands.clear();
    // Re-direct the archi pointers to point to this.
    for (const auto &iptr : m_islands) {
        iptr->m_ptr->archi_ptr = this;
    }

    // Move over the indices, clear other.
    // NOTE: the indices are still valid as above we just moved in a vector
    // of unique_ptrs, without changing their content.
    m_idx_map = std::move(other.m_idx_map);
    other.m_idx_map.clear();

    // Move over the migrants, clear other.
    m_migrants = std::move(other.m_migrants);
    other.m_migrants.clear();

    // Move over the migration log, clear other.
    m_migr_log = std::move(other.m_migr_log);
    other.m_migr_log.clear();

    // Move over the topology. No need to clear here as we know
    // in which state the topology will be in after the move.
    m_topology = std::move(other.m_topology);

    // Migration type and migrant handling policy.
    m_migr_type.store(other.m_migr_type.load(std::memory_order_relaxed), std::memory_order_relaxed);
    m_migr_handling.store(other.m_migr_handling.load(std::memory_order_relaxed), std::memory_order_relaxed);
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
 * evolution in \p this and \p other has finished. After the move,
 * \p other is left in a state which is assignable and destructible.
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
        // NOTE: we want to ensure that other is in a known state
        // after the move, so that we can run assertion checks in
        // the destructor in debug mode.
        wait_check_ignore();
        other.wait_check_ignore();

        // Move in the islands, clear other.
        m_islands = std::move(other.m_islands);
        other.m_islands.clear();
        // Re-direct the archi pointers to point to this.
        for (const auto &iptr : m_islands) {
            iptr->m_ptr->archi_ptr = this;
        }

        // Move the indices map, clear other.
        m_idx_map = std::move(other.m_idx_map);
        other.m_idx_map.clear();

        // Move over the migrants, clear other.
        m_migrants = std::move(other.m_migrants);
        other.m_migrants.clear();

        // Move over the migration log, clear other.
        m_migr_log = std::move(other.m_migr_log);
        other.m_migr_log.clear();

        // Move over the topology.
        m_topology = std::move(other.m_topology);

        // Migration type and migrant handling policy.
        m_migr_type.store(other.m_migr_type.load(std::memory_order_relaxed), std::memory_order_relaxed);
        m_migr_handling.store(other.m_migr_handling.load(std::memory_order_relaxed), std::memory_order_relaxed);
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
    // NOTE: make sure we stop the archi before running checks below without locking.
    // NOTE: this is also important to ensure everything is stopped before we start
    // destroying things, so that the destruction order will not matter.
    wait_check_ignore();

    // NOTE: we made sure in the move ctor/assignment that the island vector, the migrants and the indices
    // map are all cleared out after a move. Thus we can safely assert the following.
    assert(std::all_of(m_islands.begin(), m_islands.end(),
                       [this](const std::unique_ptr<island> &iptr) { return iptr->m_ptr->archi_ptr == this; }));
    assert(m_idx_map.size() == m_islands.size());
    assert(m_migrants.size() == m_islands.size());
#if !defined(NDEBUG)
    for (size_type i = 0; i < m_islands.size(); ++i) {
        // Ensure that the vectors in the migrant db have
        // consistent sizes.
        assert(std::get<0>(m_migrants[i]).size() == std::get<1>(m_migrants[i]).size());
        assert(std::get<1>(m_migrants[i]).size() == std::get<2>(m_migrants[i]).size());

        // Ensure the map of indices is correct.
        assert(m_idx_map.find(m_islands[i].get()) != m_idx_map.end());
        assert(m_idx_map.find(m_islands[i].get())->second == i);
    }
#endif
}

/// Mutable island access.
/**
 * This subscript operator can be used to access the <tt>i</tt>-th island of the archipelago (that is,
 * the <tt>i</tt>-th island that was inserted via push_back()). References returned by this method are valid even
 * after a push_back() invocation. Assignment and destruction of the archipelago will invalidate island references
 * obtained via this method.
 *
 * \verbatim embed:rst:leading-asterisk
 * .. warning::
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

/// Size.
/**
 * @return the number of islands in the archipelago.
 */
archipelago::size_type archipelago::size() const
{
    return m_islands.size();
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

    // The other error case: at least one island is idle with error.
    if (n_idle_error) {
        if (n_busy) {
            // At least one island is idle with error and at least one island is busy:
            // return busy error.
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

void archipelago::push_back_impl(std::unique_ptr<island> &&new_island)
{
    // Assign the pointer to this.
    // NOTE: perhaps this can be delayed until the last line?
    // The reason would be that, in case of exceptions,
    // new_island will be destroyed while pointing
    // to an archipelago, although the island is not
    // actually in the archipelago. In theory this
    // could lead to assertion failures on destruction, if
    // we implement archipelago-based checks in the dtor
    // of island. This is not the case at the moment.
    new_island->m_ptr->archi_ptr = this;

    // Try to make space for the new island in the islands vector.
    // LCOV_EXCL_START
    if (m_islands.size() == std::numeric_limits<decltype(m_islands.size())>::max()) {
        pagmo_throw(std::overflow_error, "cannot add a new island to an archipelago due to an overflow condition");
    }
    // LCOV_EXCL_STOP
    m_islands.reserve(m_islands.size() + 1u);

    // Try to make space for the new migrants entry.
    // LCOV_EXCL_START
    if (m_migrants.size() == std::numeric_limits<decltype(m_migrants.size())>::max()) {
        pagmo_throw(std::overflow_error, "cannot add a new island to an archipelago due to an overflow condition");
    }
    // LCOV_EXCL_STOP
    {
        std::lock_guard<std::mutex> lock(m_migrants_mutex);
        m_migrants.reserve(m_migrants.size() + 1u);
    }

    // Map the new island idx.
    {
        // NOTE: if anything fails here, we won't have modified the state of the archi
        // (apart from reserving memory).
        std::lock_guard<std::mutex> lock(m_idx_map_mutex);
        assert(m_idx_map.find(new_island.get()) == m_idx_map.end());
        m_idx_map.emplace(new_island.get(), m_islands.size());
    }

    // Add an empty entry to the migrants db.
    try {
        std::lock_guard<std::mutex> lock(m_migrants_mutex);
        m_migrants.emplace_back();
    } catch (...) {
        // LCOV_EXCL_START
        // NOTE: we get here only if the lock throws, because we made space for the
        // new migrants above already. Better to abort in such case, as we have no
        // reasonable path for recovering from this.
        std::cerr << "An unrecoverable error arose while adding an island to the archipelago, aborting now."
                  << std::endl;
        std::abort();
        // LCOV_EXCL_STOP
    }

    // Actually add the island. This cannot fail as we already reserved space.
    m_islands.push_back(std::move(new_island));

    // Finally, push back the topology. This is required to be thread safe, no need for locks.
    // If this fails, we will have a possibly *bad* topology in the archi, but this can
    // always happen via a bogus set_topology() and there's nothing we can do about it.
    m_topology.push_back();
}

// Get the index of an island.
// This function will return the index of the island \p isl in the archipelago. If \p isl does
// not belong to the archipelago, an error will be reaised.
archipelago::size_type archipelago::get_island_idx(const island &isl) const
{
    std::lock_guard<std::mutex> lock(m_idx_map_mutex);
    const auto ret = m_idx_map.find(&isl);
    if (ret == m_idx_map.end()) {
        pagmo_throw(std::invalid_argument,
                    "the index of an island in an archipelago was requested, but the island is not in the archipelago");
    }
    return ret->second;
}

/// Get the database of migrants.
/**
 * \verbatim embed:rst:leading-asterisk
 * During the evolution of an archipelago, islands will periodically
 * store the individuals selected for migration in a *migrant database*.
 * This is a vector of :cpp:type:`~pagmo::individuals_group_t` whose
 * size is equal to the number of islands in the archipelago, and which
 * contains the current candidate outgoing migrants for each island.
 * \endverbatim
 *
 * @return a copy of the database of migrants.
 *
 * @throws unspecified any exception thrown by threading primitives or by memory allocation errors.
 */
archipelago::migrants_db_t archipelago::get_migrants_db() const
{
    std::lock_guard<std::mutex> lock(m_migrants_mutex);
    return m_migrants;
}

/// Get the migration log.
/**
 * \verbatim embed:rst:leading-asterisk
 * Each time an individual migrates from an island (the source) to another
 * (the destination), an entry will be added to the migration log.
 * The entry is a tuple containing:
 *
 * - a timestamp of the migration,
 * - the ID of the individual that migrated,
 * - the decision and fitness vectors of the individual that migrated,
 * - the indices of the source and destination islands.
 *
 * The migration log is a collection of migration entries.
 *
 * \endverbatim
 *
 * @return a copy of the migration log.
 *
 * @throws unspecified any exception thrown by threading primitives or by memory allocation errors.
 */
archipelago::migration_log_t archipelago::get_migration_log() const
{
    std::lock_guard<std::mutex> lock(m_migr_log_mutex);
    return m_migr_log;
}

// Append entries to the migration log.
void archipelago::append_migration_log(const migration_log_t &mlog)
{
    // Don't do anything if mlog is empty.
    if (mlog.empty()) {
        return;
    }

    // Lock & append.
    std::lock_guard<std::mutex> lock(m_migr_log_mutex);
    m_migr_log.insert(m_migr_log.end(), mlog.begin(), mlog.end());
}

// Extract the migrants in the db entry for island i.
// After extraction, the db entry will be empty.
individuals_group_t archipelago::extract_migrants(size_type i)
{
    std::lock_guard<std::mutex> lock(m_migrants_mutex);

    if (i >= m_migrants.size()) {
        pagmo_throw(std::out_of_range, "cannot access the migrants of the island at index " + std::to_string(i)
                                           + ": the migrants database has a size of only "
                                           + std::to_string(m_migrants.size()));
    }

    // Move-construct the return value.
    individuals_group_t retval(std::move(m_migrants[i]));

    // Ensure the tuple we moved-from is completely
    // cleared out.
    std::get<0>(m_migrants[i]).clear();
    std::get<1>(m_migrants[i]).clear();
    std::get<2>(m_migrants[i]).clear();

    return retval;
}

// Get the migrants in the db entry for island i.
// This function will *not* clear out the db entry.
individuals_group_t archipelago::get_migrants(size_type i) const
{
    std::lock_guard<std::mutex> lock(m_migrants_mutex);

    if (i >= m_migrants.size()) {
        pagmo_throw(std::out_of_range, "cannot access the migrants of the island at index " + std::to_string(i)
                                           + ": the migrants database has a size of only "
                                           + std::to_string(m_migrants.size()));
    }

    // Return a copy of the migrants for island i.
    return m_migrants[i];
}

// Move-insert in the db entry for island i a set of migrants.
void archipelago::set_migrants(size_type i, individuals_group_t &&inds)
{
    std::lock_guard<std::mutex> lock(m_migrants_mutex);

    if (i >= m_migrants.size()) {
        pagmo_throw(std::out_of_range, "cannot access the migrants of the island at index " + std::to_string(i)
                                           + ": the migrants database has a size of only "
                                           + std::to_string(m_migrants.size()));
    }

    // Move in the new individuals.
    std::get<0>(m_migrants[i]) = std::move(std::get<0>(inds));
    std::get<1>(m_migrants[i]) = std::move(std::get<1>(inds));
    std::get<2>(m_migrants[i]) = std::move(std::get<2>(inds));
}

/// Get a copy of the topology.
/**
 * @return a copy of the topology.
 *
 * @throws unspecified any exception thrown by copying the topology.
 */
topology archipelago::get_topology() const
{
    // NOTE: topology is supposed to be thread-safe,
    // no need to protect the access.
    return m_topology;
}

/// Set a new topology.
/**
 * This function will first wait for any ongoing evolution in the archipelago to conclude,
 * and it will then set a new topology for the archipelago.
 *
 * Note that it is the user's responsibility to ensure that the new topology
 * is consistent with the archipelago's properties.
 *
 * @param topo the new topology.
 *
 * @throws unspecified any exception thrown by copying the topology.
 */
void archipelago::set_topology(topology topo)
{
    // NOTE: make sure we finish any ongoing evolution before setting the topology.
    // The assignment will trigger the destructor of the UDT, so we need to make
    // sure there's no interaction with the UDT happening.
    wait_check_ignore();
    m_topology = std::move(topo);
}

namespace detail
{

namespace
{

// Helpers to implement the archipelago::get_island_connections() function below.
template <typename T>
std::pair<std::vector<archipelago::size_type>, vector_double> get_island_connections_impl(const T &topo, std::size_t i,
                                                                                          std::true_type)
{
    // NOTE: get_connections() is required to be thread-safe.
    return topo.get_connections(i);
}

template <typename T>
std::pair<std::vector<archipelago::size_type>, vector_double> get_island_connections_impl(const T &topo, std::size_t i,
                                                                                          std::false_type)
{
    // NOTE: get_connections() is required to be thread-safe.
    auto tmp = topo.get_connections(i);

    std::pair<std::vector<archipelago::size_type>, vector_double> retval;
    retval.first.reserve(boost::numeric_cast<decltype(retval.first.size())>(tmp.first.size()));

    std::transform(tmp.first.begin(), tmp.first.end(), std::back_inserter(retval.first),
                   [](const std::size_t &n) { return boost::numeric_cast<archipelago::size_type>(n); });
    retval.second = std::move(tmp.second);

    return retval;
}

} // namespace

} // namespace detail

// Get the list of connection to the island at index i.
// The returned value is made of two vectors of equal size:
// - the indicies of the connecting islands,
// - the weights of the connections.
// This function will take care of safely converting the topology
// indices to island indices, if necessary.
std::pair<std::vector<archipelago::size_type>, vector_double> archipelago::get_island_connections(size_type i) const
{
    // NOTE: the get_connections() method of the topology
    // returns indices represented by std::size_t, but this method
    // returns indices represented by size_type. Hence, we formally
    // need to go through a conversion. We do a bit of TMP to avoid
    // the conversion in the likely case that std::size_t and size_type
    // are the same type.
    return detail::get_island_connections_impl(m_topology, boost::numeric_cast<std::size_t>(i),
                                               std::is_same<std::size_t, size_type>{});
}

/// Get the migration type.
/**
 * @return the migration type for this archipelago.
 */
migration_type archipelago::get_migration_type() const
{
    return m_migr_type.load(std::memory_order_relaxed);
}

/// Set a new migration type.
/**
 * @param mt a new migration type for this archipelago.
 */
void archipelago::set_migration_type(migration_type mt)
{
    m_migr_type.store(mt, std::memory_order_relaxed);
}

/// Get the migrant handling policy.
/**
 * @return the migrant handling policy for this archipelago.
 */
migrant_handling archipelago::get_migrant_handling() const
{
    return m_migr_handling.load(std::memory_order_relaxed);
}

/// Set a new migrant handling policy.
/**
 * @param mh a new migrant handling policy for this archipelago.
 */
void archipelago::set_migrant_handling(migrant_handling mh)
{
    m_migr_handling.store(mh, std::memory_order_relaxed);
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
    stream(os, "Topology: ", archi.get_topology().get_name(), "\n");
    stream(os, "Migration type: ", archi.get_migration_type(), "\n");
    stream(os, "Migrant handling policy: ", archi.get_migrant_handling(), "\n");
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

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Provide the stream operator overloads for migration_type and migrant_handling.
std::ostream &operator<<(std::ostream &os, migration_type mt)
{
    os << (mt == migration_type::p2p ? "point-to-point" : "broadcast");
    return os;
}

std::ostream &operator<<(std::ostream &os, migrant_handling mh)
{
    os << (mh == migrant_handling::preserve ? "preserve" : "evict");
    return os;
}

#endif

} // namespace pagmo
