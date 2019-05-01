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

#ifndef PYGMO_ALGORITHM_HPP
#define PYGMO_ALGORITHM_HPP

#include <pygmo/python_includes.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/python/object.hpp>
#include <boost/python/object/pickle_support.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>

#include <pygmo/common_base.hpp>
#include <pygmo/object_serialization.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

// Disable the static UDA checks for bp::object.
template <>
struct disable_uda_checks<bp::object> : std::true_type {
};

template <>
struct algo_inner<bp::object> final : algo_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    algo_inner() = default;
    algo_inner(const algo_inner &) = delete;
    algo_inner(algo_inner &&) = delete;
    algo_inner &operator=(const algo_inner &) = delete;
    algo_inner &operator=(algo_inner &&) = delete;
    explicit algo_inner(const bp::object &);
    virtual std::unique_ptr<algo_inner_base> clone() const override final;
    // Mandatory methods.
    virtual population evolve(const population &) const override final;
    // Optional methods.
    virtual void set_seed(unsigned) override final;
    virtual bool has_set_seed() const override final;
    virtual pagmo::thread_safety get_thread_safety() const override final;
    virtual std::string get_name() const override final;
    virtual std::string get_extra_info() const override final;
    virtual void set_verbosity(unsigned) override final;
    virtual bool has_set_verbosity() const override final;
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        ar << boost::serialization::base_object<algo_inner_base>(*this);
        ar << pygmo::object_to_vchar(m_value);
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        ar >> boost::serialization::base_object<algo_inner_base>(*this);
        std::vector<char> v;
        ar >> v;
        m_value = pygmo::vchar_to_object(v);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    bp::object m_value;
};

} // namespace detail

} // namespace pagmo

// Register the algo_inner specialisation for bp::object.
PAGMO_S11N_ALGORITHM_EXPORT_KEY(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the algorithm class.
struct algorithm_pickle_suite : bp::pickle_suite {
    static bp::tuple getstate(const pagmo::algorithm &);
    static void setstate(pagmo::algorithm &, const bp::tuple &);
};

} // namespace pygmo

#endif
