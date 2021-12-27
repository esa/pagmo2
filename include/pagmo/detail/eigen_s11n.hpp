/* Copyright 2017-2021 PaGMO development team

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

#ifndef PAGMO_DETAIL_EIGEN_S11N_HPP
#define PAGMO_DETAIL_EIGEN_S11N_HPP

#include <pagmo/detail/eigen.hpp>
#include <pagmo/s11n.hpp>

// Boost.serialization support for Eigen::Matrix.
namespace boost
{

namespace serialization
{

// Implement the serialization of the Eigen::Matrix class
template <class Archive, class S, int R, int C, int O, int MR, int MC>
inline void save(Archive &ar, Eigen::Matrix<S, R, C, O, MR, MC> const &cb, unsigned)
{
    // Let's first save the dimension
    auto nrows = cb.rows();
    auto ncols = cb.cols();
    ar << nrows;
    ar << ncols;
    // And then the numbers
    for (decltype(nrows) i = 0; i < nrows; ++i) {
        for (decltype(nrows) j = 0; j < ncols; ++j) {
            ar << cb(i, j);
        }
    }
}
template <class Archive, class S, int R, int C, int O, int MR, int MC>
inline void load(Archive &ar, Eigen::Matrix<S, R, C, O, MR, MC> &cb, unsigned)
{
    decltype(cb.rows()) nrows;
    decltype(cb.cols()) ncols;
    // Let's first restore the dimension
    ar >> nrows;
    ar >> ncols;
    cb.resize(nrows, ncols);
    // And then the numbers
    for (decltype(nrows) i = 0; i < nrows; ++i) {
        for (decltype(nrows) j = 0; j < ncols; ++j) {
            ar >> cb(i, j);
        }
    }
}

template <class Archive, class S, int R, int C, int O, int MR, int MC>
inline void serialize(Archive &ar, Eigen::Matrix<S, R, C, O, MR, MC> &cb, unsigned version)
{
    split_free(ar, cb, version);
}

} // namespace serialization

} // namespace boost

#endif
