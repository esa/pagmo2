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

#ifndef PAGMO_IO_HPP
#define PAGMO_IO_HPP

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>

namespace pagmo
{

#if !defined(PAGMO_DOXYGEN_INVOKED)

// LCOV_EXCL_START
// Forward declaration
template <typename... Args>
inline void stream(std::ostream &, const Args &...);
// LCOV_EXCL_STOP

#endif

namespace detail
{

template <typename T>
inline void stream_impl(std::ostream &os, const T &x)
{
    os << x;
}

inline void stream_impl(std::ostream &os, const bool &b)
{
    if (b) {
        os << "true";
    } else {
        os << "false";
    }
}

// Maximum number of elements printed for a container
// (vector, map, etc.).
constexpr unsigned max_stream_output_length()
{
    return 5u;
}

// Helper to stream a [begin, end) range.
template <typename It>
inline void stream_range(std::ostream &os, It begin, It end)
{
    // Special-case an empty range.
    if (begin == end) {
        os << "[]";
        return;
    }

    os << '[';

    for (auto counter = 0u;; ++counter) {
        if (counter == max_stream_output_length()) {
            // NOTE: if we are here, it means we have more
            // elements in the range to print, but we already
            // printed the maximum number of elements.
            // Add the ellipsis and exit.
            os << "... ";
            break;
        }

        // Stream the current element of the range.
        stream(os, *begin);

        // NOTE: because we handled the empty range earlier,
        // ++begin is always well-defined at the first iteration
        // of this loop. Following iterations will happen only
        // if begin != end.
        if (++begin == end) {
            // We printed the last element. Omit the comma,
            // and exit.
            break;
        }

        // We have more elements to print, or perhaps the
        // ellipsis. Print comma and add space.
        os << ", ";
    }

    os << ']';
}

// Implementation for vector.
template <typename T>
inline void stream_impl(std::ostream &os, const std::vector<T> &v)
{
    stream_range(os, v.begin(), v.end());
}

template <typename T, typename U>
inline void stream_impl(std::ostream &os, const std::pair<T, U> &p)
{
    stream(os, '(', p.first, ',', p.second, ')');
}

template <typename T, typename U>
inline void stream_impl(std::ostream &os, const std::map<T, U> &m)
{
    unsigned counter = 0;
    stream(os, '{');
    for (auto it = m.begin(); it != m.end(); ++counter) {
        if (counter == max_stream_output_length()) {
            stream(os, "...");
            break;
        }
        stream(os, it->first, " : ", it->second);
        ++it;
        if (it != m.end()) {
            stream(os, ",  ");
        }
    }
    stream(os, '}');
}

template <typename T, typename... Args>
inline void stream_impl(std::ostream &os, const T &x, const Args &... args)
{
    stream_impl(os, x);
    stream_impl(os, args...);
}

// A small helper function that transforms x to string, using internally pagmo::stream.
template <typename T>
inline std::string to_string(const T &x)
{
    std::ostringstream oss;
    stream(oss, x);
    return oss.str();
}

// Gizmo to create simple ascii tables.
struct PAGMO_DLL_PUBLIC table {
    using s_size_t = std::string::size_type;
    // Construct from table headers, and optional indentation to be used when printing
    // the table.
    table(std::vector<std::string> headers, std::string indent = "");
    // Add a row to the table. The input arguments are converted to string using to_string.
    // assembled in a row, and the row is then added to the table. The maximum column widths
    // are updated if elements in args require more width than currently allocated.
    template <typename... Args>
    void add_row(const Args &... args)
    {
        if (sizeof...(args) != m_headers.size()) {
            pagmo_throw(std::invalid_argument, "the table was constructed with " + to_string(m_headers.size())
                                                   + " columns, but a row with " + to_string(sizeof...(args))
                                                   + " columns is being added: the two values must be equal");
        }
        // Convert to a vector of strings, and add the row.
        m_rows.emplace_back(std::vector<std::string>{to_string(args)...});
        // Update the column widths as needed.
        std::transform(m_rows.back().begin(), m_rows.back().end(), m_sizes.begin(), m_sizes.begin(),
                       [](const std::string &str, const s_size_t &size) { return (std::max)(str.size(), size); });
    }

    std::string m_indent;
    std::vector<std::string> m_headers;
    std::vector<s_size_t> m_sizes;
    std::vector<std::vector<std::string>> m_rows;
};

// Print the table to stream.
PAGMO_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const table &);

} // end of namespace detail

/// The pagmo streaming function.
/**
 * This function will direct to the output stream \p os the input arguments \p args.
 *
 * @param os the target stream.
 * @param args the objects that will be directed to to \p os.
 */
template <typename... Args>
inline void stream(std::ostream &os, const Args &... args)
{
    detail::stream_impl(os, args...);
}

/// The pagmo print function.
/**
 * This function is equivalent to calling pagmo::stream with \p std::cout as first argument.
 *
 * @param args the objects that will be printed to screen.
 */
template <typename... Args>
inline void print(const Args &... args)
{
    stream(std::cout, args...);
}

} // end of namespace pagmo

#endif
