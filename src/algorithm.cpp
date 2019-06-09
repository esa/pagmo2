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

#include <iostream>
#include <string>
#include <utility>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/null_algorithm.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>

// MINGW-specific warnings.
#if defined(__GNUC__) && defined(__MINGW32__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=pure"
#endif

namespace pagmo
{

/// Default constructor.
/**
 * The default constructor will initialize a pagmo::algorithm containing a pagmo::null_algorithm.
 *
 * @throws unspecified any exception thrown by the constructor from UDA.
 */
algorithm::algorithm() : algorithm(null_algorithm{}) {}

void algorithm::generic_ctor_impl()
{
    // We detect if set_seed is implemented in the algorithm, in which case the algorithm is stochastic
    m_has_set_seed = ptr()->has_set_seed();
    // We detect if set_verbosity is implemented in the algorithm
    m_has_set_verbosity = ptr()->has_set_verbosity();
    // We store at construction the value returned from the user implemented get_name
    m_name = ptr()->get_name();
    // Store the thread safety value.
    m_thread_safety = ptr()->get_thread_safety();
}

/// Copy constructor
/**
 * The copy constructor will deep copy the input algorithm \p other.
 *
 * @param other the algorithm to be copied.
 *
 * @throws unspecified any exception thrown by:
 * - memory allocation errors in standard containers,
 * - the copying of the internal UDA.
 */
algorithm::algorithm(const algorithm &other)
    : m_ptr(other.m_ptr->clone()), m_has_set_seed(other.m_has_set_seed), m_has_set_verbosity(other.m_has_set_verbosity),
      m_name(other.m_name), m_thread_safety(other.m_thread_safety)
{
}

/// Move constructor
/**
 * @param other the algorithm from which \p this will be move-constructed.
 */
algorithm::algorithm(algorithm &&other) noexcept
    : m_ptr(std::move(other.m_ptr)), m_has_set_seed(std::move(other.m_has_set_seed)),
      m_has_set_verbosity(other.m_has_set_verbosity), m_name(std::move(other.m_name)),
      m_thread_safety(std::move(other.m_thread_safety))
{
}

/// Move assignment operator
/**
 * @param other the assignment target.
 *
 * @return a reference to \p this.
 */
algorithm &algorithm::operator=(algorithm &&other) noexcept
{
    if (this != &other) {
        m_ptr = std::move(other.m_ptr);
        m_has_set_seed = std::move(other.m_has_set_seed);
        m_has_set_verbosity = other.m_has_set_verbosity;
        m_name = std::move(other.m_name);
        m_thread_safety = std::move(other.m_thread_safety);
    }
    return *this;
}

/// Copy assignment operator
/**
 * Copy assignment is implemented as a copy constructor followed by a move assignment.
 *
 * @param other the assignment target.
 *
 * @return a reference to \p this.
 *
 * @throws unspecified any exception thrown by the copy constructor.
 */
algorithm &algorithm::operator=(const algorithm &other)
{
    // Copy ctor + move assignment.
    return *this = algorithm(other);
}

/// Evolve method.
/**
 * This method will invoke the <tt>%evolve()</tt> method of the UDA. This is where the core of the optimization
 * (*evolution*) is made.
 *
 * @param pop starting population
 *
 * @return evolved population
 *
 * @throws unspecified any exception thrown by the <tt>%evolve()</tt> method of the UDA.
 */
population algorithm::evolve(const population &pop) const
{
    return ptr()->evolve(pop);
}

/// Set the seed for the stochastic evolution.
/**
 * Sets the seed to be used in the <tt>%evolve()</tt> method of the UDA for all stochastic variables. If the UDA
 * satisfies pagmo::has_set_seed, then its <tt>%set_seed()</tt> method will be invoked. Otherwise, an error will be
 * raised.
 *
 * @param seed seed.
 *
 * @throws not_implemented_error if the UDA does not satisfy pagmo::has_set_seed.
 * @throws unspecified any exception thrown by the <tt>%set_seed()</tt> method of the UDA.
 */
void algorithm::set_seed(unsigned seed)
{
    ptr()->set_seed(seed);
}

/// Set the verbosity of logs and screen output.
/**
 * This method will set the level of verbosity for the algorithm. If the UDA satisfies pagmo::has_set_verbosity,
 * then its <tt>%set_verbosity()</tt> method will be invoked. Otherwise, an error will be raised.
 *
 * The exact meaning of the input parameter \p level is dependent on the UDA.
 *
 * @param level the desired verbosity level.
 *
 * @throws not_implemented_error if the UDA does not satisfy pagmo::has_set_verbosity.
 * @throws unspecified any exception thrown by the <tt>%set_verbosity()</tt> method of the UDA.
 */
void algorithm::set_verbosity(unsigned level)
{
    ptr()->set_verbosity(level);
}

/// Algorithm's extra info.
/**
 * If the UDA satisfies pagmo::has_extra_info, then this method will return the output of its
 * <tt>%get_extra_info()</tt> method. Otherwise, an empty string will be returned.
 *
 * @return extra info about the UDA.
 *
 * @throws unspecified any exception thrown by the <tt>%get_extra_info()</tt> method of the UDA.
 */
std::string algorithm::get_extra_info() const
{
    return ptr()->get_extra_info();
}

/// Check if the algorithm is in a valid state.
/**
 * @return ``false`` if ``this`` was moved from, ``true`` otherwise.
 */
bool algorithm::is_valid() const
{
    return static_cast<bool>(m_ptr);
}

/// Streaming operator for pagmo::algorithm
/**
 * This function will stream to \p os a human-readable representation of the input
 * algorithm \p a.
 *
 * @param os input <tt>std::ostream</tt>.
 * @param a pagmo::algorithm object to be streamed.
 *
 * @return a reference to \p os.
 *
 * @throws unspecified any exception thrown by querying various algorithm properties and streaming them into \p os.
 */
std::ostream &operator<<(std::ostream &os, const algorithm &a)
{
    os << "Algorithm name: " << a.get_name();
    if (!a.has_set_seed()) {
        stream(os, " [deterministic]");
    } else {
        stream(os, " [stochastic]");
    }
    stream(os, "\n\tThread safety: ", a.get_thread_safety(), '\n');
    const auto extra_str = a.get_extra_info();
    if (!extra_str.empty()) {
        stream(os, "\nExtra info:\n", extra_str);
    }
    return os;
}

} // namespace pagmo
