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

#ifndef PYGMO_PROBLEM_HPP
#define PYGMO_PROBLEM_HPP

#include <pygmo/python_includes.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/python/object.hpp>
#include <boost/python/object/pickle_support.hpp>
#include <boost/python/tuple.hpp>

#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include <pygmo/common_base.hpp>
#include <pygmo/object_serialization.hpp>

namespace pagmo
{

namespace detail
{

namespace bp = boost::python;

// Disable the static UDP checks for bp::object.
template <>
struct disable_udp_checks<bp::object> : std::true_type {
};

// NOTE: here we are specialising the prob_inner implementation template for bp::object.
// We need to do this because the default implementation works on C++ types by detecting
// their methods via type-traits at compile-time, but here we need to check the presence
// of methods at runtime. That is, we need to replace the type-traits with runtime
// inspection of Python objects.
//
// We cannot be as precise as in C++ detecting the methods' signatures (it might be
// possible with the inspect module in principle, but it looks messy and it might break if the methods
// are implemented as C/C++ extensions). The main policy adopted here is: if the bp::object
// has a callable attribute with the required name, then the "runtime type-trait" is considered
// satisfied, otherwise not.
template <>
struct prob_inner<bp::object> final : prob_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    explicit prob_inner(const bp::object &);
    virtual std::unique_ptr<prob_inner_base> clone() const override final;
    // Mandatory methods.
    virtual vector_double fitness(const vector_double &) const override final;
    virtual std::pair<vector_double, vector_double> get_bounds() const override final;
    // Optional methods.
    virtual vector_double batch_fitness(const vector_double &) const override final;
    virtual bool has_batch_fitness() const override final;
    virtual vector_double::size_type get_nobj() const override final;
    virtual vector_double::size_type get_nec() const override final;
    virtual vector_double::size_type get_nic() const override final;
    virtual vector_double::size_type get_nix() const override final;
    virtual std::string get_name() const override final;
    virtual std::string get_extra_info() const override final;
    virtual bool has_gradient() const override final;
    virtual vector_double gradient(const vector_double &) const override final;
    virtual bool has_gradient_sparsity() const override final;
    virtual sparsity_pattern gradient_sparsity() const override final;
    virtual bool has_hessians() const override final;
    virtual std::vector<vector_double> hessians(const vector_double &) const override final;
    virtual bool has_hessians_sparsity() const override final;
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final;
    virtual void set_seed(unsigned) override final;
    virtual bool has_set_seed() const override final;
    // Hard code no thread safety for python problems.
    virtual pagmo::thread_safety get_thread_safety() const override final;
    template <typename Archive>
    void save(Archive &ar, unsigned) const
    {
        ar << boost::serialization::base_object<prob_inner_base>(*this);
        ar << pygmo::object_to_vchar(m_value);
    }
    template <typename Archive>
    void load(Archive &ar, unsigned)
    {
        ar >> boost::serialization::base_object<prob_inner_base>(*this);
        std::vector<char> v;
        ar >> v;
        m_value = pygmo::vchar_to_object(v);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    bp::object m_value;
};
} // namespace detail

} // namespace pagmo

// Register the prob_inner specialisation for bp::object.
PAGMO_S11N_PROBLEM_EXPORT_KEY(boost::python::object)

namespace pygmo
{

namespace bp = boost::python;

// Serialization support for the problem class.
struct problem_pickle_suite : bp::pickle_suite {
    static bp::tuple getstate(const pagmo::problem &);
    static void setstate(pagmo::problem &, const bp::tuple &);
};

} // namespace pygmo

#endif
