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

#ifndef PAGMO_ARCHIPELAGO_HPP
#define PAGMO_ARCHIPELAGO_HPP

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/iterator/indirect_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/archipelago_fwd.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/detail/visibility.hpp>
#include <pagmo/island.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/type_traits.hpp>
#include <pagmo/types.hpp>

namespace pagmo
{

/// Migration type.
/**
 * \verbatim embed:rst:leading-asterisk
 * This enumeration represents the available migration policies
 * in an :cpp:class:`~pagmo::archipelago`:
 *
 * - with the point-to-point migration policy, during migration an island will
 *   consider individuals from only one of the connecting islands;
 * - with the broadcast migration policy, during migration an island will consider
 *   individuals from *all* the connecting islands.
 *
 * \endverbatim
 */
enum class migration_type {
    p2p,      ///< Point-to-point migration.
    broadcast ///< Broadcast migration.
};

/// Migrant handling policy.
/**
 * \verbatim embed:rst:leading-asterisk
 * This enumeration represents the available migrant handling
 * policies in an :cpp:class:`~pagmo::archipelago`.
 *
 * During migration,
 * individuals are selected from the islands and copied into a migration
 * database, from which they can be fetched by other islands.
 * This policy establishes what happens to the migrants in the database
 * after they have been fetched by a destination island:
 *
 * - with the preserve policy, a copy of the candidate migrants
 *   remains in the database;
 * - with the evict policy, the candidate migrants are
 *   removed from the database.
 *
 * \endverbatim
 */
enum class migrant_handling {
    preserve, ///< Perserve migrants in the database.
    evict     ///< Evict migrants from the database.
};

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Provide the stream operator overloads for migration_type and migrant_handling.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, migration_type);
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, migrant_handling);

#endif

// TODO
// - ctors with topology arguments.

/// Archipelago.
/**
 * \image html archi_no_text.png
 *
 * \verbatim embed:rst:leading-asterisk
 * An archipelago is a collection of :cpp:class:`~pagmo::island` objects connected by a
 * :cpp:class:`~pagmo::topology`. The islands in the archipelago can exchange individuals
 * (i.e., candidate solutions) via a process called *migration*. The individuals migrate
 * across the routes described by the topology, and the islands' replacement
 * and selection policies (see :cpp:class:`~pagmo::r_policy` and :cpp:class:`~pagmo::s_policy`)
 * establish how individuals are replaced in and selected from the islands' populations.
 *
 * The interface of :cpp:class:`~pagmo::archipelago` mirrors partially the interface
 * of :cpp:class:`~pagmo::island`: the evolution is initiated by a call to :cpp:func:`~pagmo::archipelago::evolve()`,
 * and at any time the user can query the
 * state of the archipelago and access its island members. The user can explicitly wait for pending evolutions
 * to conclude by calling the :cpp:func:`~pagmo::archipelago::wait()` and :cpp:func:`~pagmo::archipelago::wait_check()`
 * methods. The status of
 * ongoing evolutions in the archipelago can be queried via :cpp:func:`~pagmo::archipelago::status()`.
 *
 * .. warning::
 *
 *    The only operations allowed on a moved-from :cpp:class:`pagmo::archipelago` are destruction
 *    and assignment. Any other operation will result in undefined behaviour.
 *
 * \endverbatim
 */
class PAGMO_DLL_PUBLIC archipelago
{
    // Make friends with island.
    friend class PAGMO_DLL_PUBLIC island;

    using container_t = std::vector<std::unique_ptr<island>>;
    using size_type_implementation = container_t::size_type;
    using iterator_implementation = boost::indirect_iterator<container_t::iterator>;
    using const_iterator_implementation = boost::indirect_iterator<container_t::const_iterator>;

    // NOTE: same utility method as in pagmo::island, see there.
    PAGMO_DLL_LOCAL void wait_check_ignore();

public:
    /// The size type of the archipelago.
    /**
     * This is an unsigned integer type used to represent the number of islands in the
     * archipelago.
     */
    using size_type = size_type_implementation;

    /// Database of migrants.
    /**
     * \verbatim embed:rst:leading-asterisk
     * During the evolution of an archipelago, islands will periodically
     * store the individuals selected for migration in a *migrant database*.
     * This is a vector of :cpp:type:`~pagmo::individuals_group_t` whose
     * size is equal to the number of islands in the archipelago, and which
     * contains the current candidate outgoing migrants for each island.
     * \endverbatim
     */
    using migrants_db_t = std::vector<individuals_group_t>;

    /// Entry for the migration log.
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
     * \endverbatim
     */
    using migration_entry_t
        = std::tuple<double, unsigned long long, vector_double, vector_double, size_type, size_type>;

    /// Migration log.
    /**
     * \verbatim embed:rst:leading-asterisk
     * The migration log is a collection of :cpp:type:`~pagmo::archipelago::migration_entry_t` entries.
     * \endverbatim
     */
    using migration_log_t = std::vector<migration_entry_t>;

private:
    // A map to connect island pointers to an idx
    // in the archipelago. This will be used by islands
    // during migration in order to establish the island
    // indices within the archipelago.
    using idx_map_t = std::unordered_map<const island *, size_type>;

public:
    /// Mutable iterator.
    /**
     * Dereferencing a mutable iterator will yield a reference to an island within the archipelago.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    Mutable iterators are provided solely in order to allow calling non-const methods
     *    on the islands. Assigning an island via a mutable iterator will result in undefined behaviour.
     *
     * \endverbatim
     */
    using iterator = iterator_implementation;
    /// Const iterator.
    /**
     * Dereferencing a const iterator will yield a const reference to an island within the archipelago.
     */
    using const_iterator = const_iterator_implementation;
    // Default constructor.
    archipelago();
    // Copy constructor.
    archipelago(const archipelago &);
    // Move constructor.
    archipelago(archipelago &&) noexcept;

private:
#if defined(_MSC_VER)
    template <typename...>
    using n_ctor_enabler = int;
#else
    template <typename... Args>
    using n_ctor_enabler = enable_if_t<std::is_constructible<island, Args...>::value, int>;
#endif
    // The "default" constructor from n islands. Just forward
    // the input arguments to n calls to push_back().
    template <typename... Args>
    void n_ctor(size_type n, const Args &... args)
    {
        for (size_type i = 0; i < n; ++i) {
            // NOTE: we don't perfectly forward, in order to avoid moving twice
            // from the same objects. This also ensures that, when
            // using a ctor without seed, the seed is set to random
            // for each island.
            push_back(args...);
        }
    }
    // These implement the constructor which contains a seed argument.
    // We need to constrain with enable_if otherwise, due to the default
    // arguments in the island constructors, the wrong constructor would be called.
    template <typename Algo, typename Prob, typename S1, typename S2,
              enable_if_t<detail::conjunction<std::is_constructible<algorithm, const Algo &>,
                                              std::is_constructible<problem, const Prob &>, std::is_integral<S1>,
                                              std::is_integral<S2>>::value,
                          int> = 0>
    void n_ctor(size_type n, const Algo &a, const Prob &p, S1 size, S2 seed)
    {
        std::mt19937 eng(static_cast<std::mt19937::result_type>(static_cast<unsigned>(seed)));
        std::uniform_int_distribution<unsigned> udist;
        for (size_type i = 0; i < n; ++i) {
            push_back(a, p, boost::numeric_cast<population::size_type>(size), udist(eng));
        }
    }
    // Same as previous, with bfe argument.
    // NOTE: performance wise, it would be better for these constructors from bfe
    // to batch initialise *all* archi individuals
    // (whereas now we batch init n times, one for each island). Keep this in mind
    // for future developments.
    template <
        typename Algo, typename Prob, typename Bfe, typename S1, typename S2,
        enable_if_t<detail::conjunction<
                        std::is_constructible<algorithm, const Algo &>, std::is_constructible<problem, const Prob &>,
                        std::is_constructible<bfe, const Bfe &>, std::is_integral<S1>, std::is_integral<S2>>::value,
                    int> = 0>
    void n_ctor(size_type n, const Algo &a, const Prob &p, const Bfe &b, S1 size, S2 seed)
    {
        std::mt19937 eng(static_cast<std::mt19937::result_type>(static_cast<unsigned>(seed)));
        std::uniform_int_distribution<unsigned> udist;
        for (size_type i = 0; i < n; ++i) {
            push_back(a, p, b, boost::numeric_cast<population::size_type>(size), udist(eng));
        }
    }
    // Constructor with UDI argument.
    template <typename Isl, typename Algo, typename Prob, typename S1, typename S2,
              enable_if_t<detail::conjunction<is_udi<Isl>, std::is_constructible<algorithm, const Algo &>,
                                              std::is_constructible<problem, const Prob &>, std::is_integral<S1>,
                                              std::is_integral<S2>>::value,
                          int> = 0>
    void n_ctor(size_type n, const Isl &isl, const Algo &a, const Prob &p, S1 size, S2 seed)
    {
        std::mt19937 eng(static_cast<std::mt19937::result_type>(static_cast<unsigned>(seed)));
        std::uniform_int_distribution<unsigned> udist;
        for (size_type i = 0; i < n; ++i) {
            push_back(isl, a, p, boost::numeric_cast<population::size_type>(size), udist(eng));
        }
    }
    // Same as previous, with bfe argument.
    template <typename Isl, typename Algo, typename Prob, typename Bfe, typename S1, typename S2,
              enable_if_t<detail::conjunction<is_udi<Isl>, std::is_constructible<algorithm, const Algo &>,
                                              std::is_constructible<problem, const Prob &>,
                                              std::is_constructible<bfe, const Bfe &>, std::is_integral<S1>,
                                              std::is_integral<S2>>::value,
                          int> = 0>
    void n_ctor(size_type n, const Isl &isl, const Algo &a, const Prob &p, const Bfe &b, S1 size, S2 seed)
    {
        std::mt19937 eng(static_cast<std::mt19937::result_type>(static_cast<unsigned>(seed)));
        std::uniform_int_distribution<unsigned> udist;
        for (size_type i = 0; i < n; ++i) {
            push_back(isl, a, p, b, boost::numeric_cast<population::size_type>(size), udist(eng));
        }
    }

public:
    /// Constructor from \p n islands.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This constructor is enabled only if the parameter pack ``Args``
     *    can be used to construct a :cpp:class:`pagmo::island`.
     *
     * \endverbatim
     *
     * This constructor will first initialise an empty archipelago with a default-constructed
     * topology, and then forward \p n times the input arguments \p args to the
     * push_back() method. If, however, the parameter pack contains an argument which
     * would be interpreted as a seed by the invoked island constructor, then this seed
     * will be used to initialise a random number generator that in turn will be used to generate
     * the seeds of populations of the islands that will be created within the archipelago. In other words,
     * passing a seed argument to this constructor will not generate \p n islands with the same
     * seed, but \p n islands whose population seeds have been randomly generated starting from
     * the supplied seed argument.
     *
     * @param n the desired number of islands.
     * @param args the arguments that will be used for the construction of each island.
     *
     * @throws unspecified any exception thrown by the invoked pagmo::island constructor
     * or by archipelago::push_back().
     */
    template <typename... Args, n_ctor_enabler<const Args &...> = 0>
    explicit archipelago(size_type n, const Args &... args)
    {
        n_ctor(n, args...);
    }
    // Copy assignment.
    archipelago &operator=(const archipelago &);
    archipelago &operator=(archipelago &&) noexcept;
    // Destructor.
    ~archipelago();
    // Mutable island access.
    island &operator[](size_type);
    // Const island access.
    const island &operator[](size_type) const;
    /// Size.
    /**
     * @return the number of islands in the archipelago.
     */
    size_type size() const
    {
        return m_islands.size();
    }

private:
#if defined(_MSC_VER)
    template <typename...>
    using push_back_enabler = int;
#else
    template <typename... Args>
    using push_back_enabler = enable_if_t<std::is_constructible<island, Args...>::value, int>;
#endif

    // Implementation of push_back().
    void push_back_impl(std::unique_ptr<island> &&);

public:
    /// Add island.
    /**
     * \verbatim embed:rst:leading-asterisk
     * .. note::
     *
     *    This method is enabled only if the parameter pack ``Args``
     *    can be used to construct a :cpp:class:`pagmo::island`.
     *
     * \endverbatim
     *
     * This method will construct an island from the supplied arguments and add it to the archipelago.
     * Islands are added at the end of the archipelago (that is, the new island will have an index
     * equal to the value of size() before the call to this method). pagmo::topology::push_back()
     * will also be called on the pagmo::topology associated to this archipelago.
     *
     * @param args the arguments that will be used for the construction of the island.
     *
     * @throws std::overflow_error if the size of the archipelago is greater than an
     * implementation-defined maximum.
     * @throws unspecified any exception thrown by:
     * - memory allocation errors,
     * - threading primitives,
     * - pagmo::topology::push_back(),
     * - the invoked constructor of pagmo::island.
     */
    template <typename... Args, push_back_enabler<Args &&...> = 0>
    void push_back(Args &&... args)
    {
        push_back_impl(detail::make_unique<island>(std::forward<Args>(args)...));
    }
    /// Evolve archipelago.
    /**
     * This method will call island::evolve() on all the islands of the archipelago.
     * The input parameter \p n will be passed to the invocations of island::evolve() for each island.
     * archipelago::status() can be used to query the status of the asynchronous operations in the
     * archipelago.
     *
     * @param n the parameter that will be passed to island::evolve().
     *
     * @throws unspecified any exception thrown by island::evolve().
     */
    void evolve(unsigned n = 1);
    // Block until all evolutions have finished.
    void wait() noexcept;
    // Block until all evolutions have finished and raise the first exception that was encountered.
    void wait_check();
    // Status of the archipelago.
    evolve_status status() const;

    /// Mutable begin iterator.
    /**
     * This method will return a mutable iterator pointing to the beginning of the internal island container. That is,
     * the returned iterator will either point to the first island of the archipelago (if size() is nonzero)
     * or it will be the same iterator returned by archipelago::end() (is size() is zero).
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    Mutable iterators are provided solely in order to allow calling non-const methods
     *    on the islands. Assigning an island via a mutable iterator will result in undefined behaviour.
     *
     * \endverbatim
     *
     * @return a mutable iterator to the beginning of the island container.
     */
    iterator begin()
    {
        return iterator(m_islands.begin());
    }
    /// Mutable end iterator.
    /**
     * This method will return a mutable iterator pointing to the end of the internal island container.
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * \verbatim embed:rst:leading-asterisk
     * .. warning::
     *
     *    Mutable iterators are provided solely in order to allow calling non-const methods
     *    on the islands. Assigning an island via a mutable iterator will result in undefined behaviour.
     *
     * \endverbatim
     *
     * @return a mutable iterator to the end of the island container.
     */
    iterator end()
    {
        return iterator(m_islands.end());
    }
    /// Const begin iterator.
    /**
     * This method will return a const iterator pointing to the beginning of the internal island container. That is,
     * the returned iterator will either point to the first island of the archipelago (if size() is nonzero)
     * or it will be the same iterator returned by archipelago::end() const (is size() is zero).
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * @return a const iterator to the beginning of the island container.
     */
    const_iterator begin() const
    {
        return const_iterator(m_islands.begin());
    }
    /// Const end iterator.
    /**
     * This method will return a const iterator pointing to the end of the internal island container.
     *
     * Adding an island to the archipelago will invalidate all existing iterators.
     *
     * @return a const iterator to the end of the island container.
     */
    const_iterator end() const
    {
        return const_iterator(m_islands.end());
    }

    // Get the fitness vectors of the islands' champions.
    std::vector<vector_double> get_champions_f() const;
    // Get the decision vectors of the islands' champions.
    std::vector<vector_double> get_champions_x() const;

    migration_log_t get_migration_log() const;
    migrants_db_t get_migrants_db() const;

    topology get_topology() const;
    void set_topology(topology);

    migration_type get_migration_type() const;
    void set_migration_type(migration_type);
    migrant_handling get_migrant_handling() const;
    void set_migrant_handling(migrant_handling);

    /// Save to archive.
    /**
     * This method will save to \p ar the islands of the archipelago.
     *
     * @param ar the output archive.
     *
     * @throws unspecified any exception thrown by the serialization of pagmo::island, pagmo::topology
     * or primitive types.
     */
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        detail::to_archive(ar, m_islands, get_migrants_db(), get_migration_log(), get_topology(),
                           m_migr_type.load(std::memory_order_relaxed),
                           m_migr_handling.load(std::memory_order_relaxed));
    }
    /// Load from archive.
    /**
     * This method will load into \p this the content of \p ar, after any ongoing evolution
     * in \p this has finished.
     *
     * @param ar the input archive.
     *
     * @throws unspecified any exception thrown by the deserialization of pagmo::island, pagmo::topology
     * or primitive types, or by memory errors in standard containers.
     */
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        // NOTE: the idea here is that we will be loading the member of archi one by one in
        // separate variables, move assign the loaded data into a tmp archi and finally move-assign
        // the tmp archi into this. This allows the method to be exception safe, and to have
        // archi objects always in a consistent state at every stage of the deserialization.

        // The tmp archi. This is def-cted and idle, we will be able to move-in data without
        // worrying about synchronization.
        archipelago tmp;

        // The islands.
        container_t tmp_islands;
        ar >> tmp_islands;

        // Map the islands to indices.
        idx_map_t tmp_idx_map;
        for (size_type i = 0; i < tmp_islands.size(); ++i) {
            tmp_idx_map.emplace(tmp_islands[i].get(), i);
        }

        // The migrants.
        migrants_db_t tmp_migrants;
        ar >> tmp_migrants;

        // The migration log.
        migration_log_t tmp_migr_log;
        ar >> tmp_migr_log;

        // The topology.
        topology tmp_topo;
        ar >> tmp_topo;

        // Migration type and migrant handling policy.
        migration_type tmp_migr_type;
        migrant_handling tmp_migr_handling;

        ar >> tmp_migr_type;
        ar >> tmp_migr_handling;

        // From now on, everything is noexcept. Thus, there is
        // no danger that tmp is destructed while in an inconsistent
        // state.
        tmp.m_islands = std::move(tmp_islands);
        tmp.m_idx_map = std::move(tmp_idx_map);
        tmp.m_migrants = std::move(tmp_migrants);
        tmp.m_migr_log = std::move(tmp_migr_log);
        tmp.m_topology = std::move(tmp_topo);
        tmp.m_migr_type.store(tmp_migr_type, std::memory_order_relaxed);
        tmp.m_migr_handling.store(tmp_migr_handling, std::memory_order_relaxed);

        // NOTE: this final assignment will take care of setting the islands' archi pointers
        // appropriately via archi's move assignment operator.
        *this = std::move(tmp);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    // Private utilities for use only by island.
    // Extract/get/set migrants for the island at the given index.
    PAGMO_DLL_LOCAL individuals_group_t extract_migrants(size_type);
    PAGMO_DLL_LOCAL individuals_group_t get_migrants(size_type) const;
    PAGMO_DLL_LOCAL void set_migrants(size_type, individuals_group_t &&);
    // Helper to add entries to the migration log.
    PAGMO_DLL_LOCAL void append_migration_log(const migration_log_t &);
    // Get the index of an island.
    PAGMO_DLL_LOCAL size_type get_island_idx(const island &) const;
    // Get the connections to the island at the given index.
    PAGMO_DLL_LOCAL std::pair<std::vector<size_type>, vector_double> get_island_connections(size_type) const;

private:
    container_t m_islands;
    // The map from island pointers to indices in the archi.
    // It needs to be protected by a mutex.
    mutable std::mutex m_idx_map_mutex;
    idx_map_t m_idx_map;
    // The migrants.
    mutable std::mutex m_migrants_mutex;
    migrants_db_t m_migrants;
    // The migration log.
    mutable std::mutex m_migr_log_mutex;
    migration_log_t m_migr_log;
    // The topology.
    // NOTE: the topology does not need
    // an associated mutex as it is supposed
    // to be thread-safe already.
    topology m_topology;
    // Migration type and migrant handling policy.
    std::atomic<migration_type> m_migr_type;
    std::atomic<migrant_handling> m_migr_handling;
};

// Stream operator.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const archipelago &);

} // namespace pagmo

// Disable tracking for the serialisation of archipelago.
BOOST_CLASS_TRACKING(pagmo::archipelago, boost::serialization::track_never)

#endif
