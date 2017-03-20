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

#ifndef PAGMO_IO_HPP
#define PAGMO_IO_HPP

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "threading.hpp"

#define PAGMO_MAX_OUTPUT_LENGTH 5u

namespace pagmo
{

#if !defined(PAGMO_DOXYGEN_INVOKED)

// Forward declaration
template <typename... Args>
inline void stream(std::ostream &, const Args &...);

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

inline void stream_impl(std::ostream &os, thread_safety ts)
{
    switch (ts) {
        case thread_safety::none:
            os << "none";
            break;
        case thread_safety::copyonly:
            os << "copyonly";
            break;
        case thread_safety::basic:
            os << "basic";
            break;
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

template <typename T, typename... Args>
inline void stream_impl(std::ostream &os, const T &x, const Args &... args)
{
    stream_impl(os, x);
    stream_impl(os, args...);
}

class table
{
    using s_size_t = std::string::size_type;

public:
    table(std::vector<std::string> headers, std::string indent = "")
        : m_indent(std::move(indent)), m_headers(std::move(headers))
    {
        std::transform(m_headers.begin(), m_headers.end(), std::back_inserter(m_sizes),
                       [](const std::string &s) { return s.size(); });
    }
    void add_row(std::vector<std::string> row)
    {
        if (row.size() != m_headers.size()) {
            throw std::invalid_argument("");
        }
        m_rows.emplace_back(std::move(row));
        std::transform(
            m_rows.back().begin(), m_rows.back().end(), m_sizes.begin(), m_sizes.begin(),
            [](const std::string &str, const s_size_t &size) { return str.size() > size ? str.size() : size; });
    }
    friend std::ostream &operator<<(std::ostream &os, const table &t)
    {
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
