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
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/exceptions.hpp>

#define PAGMO_MAX_OUTPUT_LENGTH 5u

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

template <typename T>
inline void stream_impl(std::ostream &os, const std::vector<T> &v)
{
    auto len = v.size();
    if (len <= PAGMO_MAX_OUTPUT_LENGTH) {
        os << '[';
        for (decltype(v.size()) i = 0u; i < v.size(); ++i) {
            stream(os, v[i]);
            if (i != v.size() - 1u) {
                os << ", ";
            }
        }
        os << ']';
    } else {
        os << '[';
        for (decltype(v.size()) i = 0u; i < PAGMO_MAX_OUTPUT_LENGTH; ++i) {
            stream(os, v[i], ", ");
        }
        os << "... ]";
    }
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
        if (counter == PAGMO_MAX_OUTPUT_LENGTH) {
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
class table
{
    using s_size_t = std::string::size_type;

public:
    // Construct from table headers, and optional indentation to be used when printing
    // the table.
    table(std::vector<std::string> headers, std::string indent = "")
        : m_indent(std::move(indent)), m_headers(std::move(headers))
    {
        std::transform(m_headers.begin(), m_headers.end(), std::back_inserter(m_sizes),
                       [](const std::string &s) { return s.size(); });
    }
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
    // Print the table to stream.
    friend std::ostream &operator<<(std::ostream &os, const table &t)
    {
        // Small helper functor to print a single row.
        auto print_row = [&t, &os](const std::vector<std::string> &row) {
            std::transform(row.begin(), row.end(), t.m_sizes.begin(), std::ostream_iterator<std::string>(os),
                           [](const std::string &str, const s_size_t &size) {
                               return str + std::string(size - str.size() + 2u, ' ');
                           });
        };
        os << t.m_indent;
        print_row(t.m_headers);
        os << '\n' << t.m_indent;
        std::transform(t.m_sizes.begin(), t.m_sizes.end(), std::ostream_iterator<std::string>(os),
                       [](const s_size_t &size) { return std::string(size + 2u, '-'); });
        os << '\n';
        for (const auto &v : t.m_rows) {
            os << t.m_indent;
            print_row(v);
            os << '\n';
        }
        return os;
    }

private:
    std::string m_indent;
    std::vector<std::string> m_headers;
    std::vector<s_size_t> m_sizes;
    std::vector<std::vector<std::string>> m_rows;
};

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

#undef PAGMO_MAX_OUTPUT_LENGTH

#endif
