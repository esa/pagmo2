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
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <pagmo/io.hpp>

namespace pagmo
{

namespace detail
{

// Construct from table headers, and optional indentation to be used when printing
// the table.
table::table(std::vector<std::string> headers, std::string indent)
    : m_indent(std::move(indent)), m_headers(std::move(headers))
{
    std::transform(m_headers.begin(), m_headers.end(), std::back_inserter(m_sizes),
                   [](const std::string &s) { return s.size(); });
}

// Print the table to stream.
std::ostream &operator<<(std::ostream &os, const table &t)
{
    // Small helper functor to print a single row.
    auto print_row = [&t, &os](const std::vector<std::string> &row) {
        std::transform(row.begin(), row.end(), t.m_sizes.begin(), std::ostream_iterator<std::string>(os),
                       [](const std::string &str, const table::s_size_t &size) {
                           return str + std::string(size - str.size() + 2u, ' ');
                       });
    };
    os << t.m_indent;
    print_row(t.m_headers);
    os << '\n' << t.m_indent;
    std::transform(t.m_sizes.begin(), t.m_sizes.end(), std::ostream_iterator<std::string>(os),
                   [](const table::s_size_t &size) { return std::string(size + 2u, '-'); });
    os << '\n';
    for (const auto &v : t.m_rows) {
        os << t.m_indent;
        print_row(v);
        os << '\n';
    }
    return os;
}

} // namespace detail

} // namespace pagmo
