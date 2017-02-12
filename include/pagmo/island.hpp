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

#ifndef PAGMO_ISLAND_HPP
#define PAGMO_ISLAND_HPP

#include <memory>
#include <utility>

#include "algorithm.hpp"
#include "population.hpp"
#include "serialization.hpp"
#include "type_traits.hpp"

namespace pagmo
{

template <typename T>
class has_enqueue_evolution
{
    template <typename U>
    using enqueue_evolution_t = decltype(std::declval<U &>().enqueue_evolution(std::declval<const algorithm &>()));
    static const bool implementation_defined = is_detected<enqueue_evolution_t, T>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_enqueue_evolution<T>::value;

template <typename T>
class has_wait
{
    template <typename U>
    using wait_t = decltype(std::declval<const U &>().wait());
    static const bool implementation_defined = is_detected<wait_t, T>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_wait<T>::value;

template <typename T>
class has_get_population
{
    template <typename U>
    using get_population_t = decltype(std::declval<const U &>().get_population());
    static const bool implementation_defined = std::is_same<population, detected_t<get_population_t, T>>::value;

public:
    static const bool value = implementation_defined;
};

template <typename T>
const bool has_get_population<T>::value;

namespace detail
{

struct isl_inner_base {
    virtual ~isl_inner_base()
    {
    }
    virtual isl_inner_base *clone() const = 0;
    virtual void enqueue_evolution(const algorithm &) = 0;
    virtual void wait() const = 0;
    virtual population get_population() const = 0;
    template <typename Archive>
    void serialize(Archive &)
    {
    }
};

template <typename T>
struct isl_inner final : isl_inner_base {
    // Static checks.
    static_assert(std::is_default_constructible<T>::value && std::is_copy_constructible<T>::value
                      && std::is_move_constructible<T>::value && std::is_destructible<T>::value,
                  "An island must be default-constructible, copy-constructible, move-constructible and destructible.");
    static_assert(has_enqueue_evolution<T>::value,
                  "An island must provide an enqueue_evolution() method: the method was either not "
                  "provided or not implemented correctly.");
    static_assert(has_wait<T>::value, "An island must provide a wait() method: the method was either not "
                                      "provided or not implemented correctly.");
    static_assert(has_get_population<T>::value,
                  "An island must provide a get_population() method: the method was either not "
                  "provided or not implemented correctly.");
    // We just need the def ctor, delete everything else.
    isl_inner() = default;
    isl_inner(const isl_inner &) = delete;
    isl_inner(isl_inner &&) = delete;
    isl_inner &operator=(const isl_inner &) = delete;
    isl_inner &operator=(isl_inner &&) = delete;
    // Constructors from T (copy and move variants).
    explicit isl_inner(const T &x) : m_value(x)
    {
    }
    explicit isl_inner(T &&x) : m_value(std::move(x))
    {
    }
    // The clone method, used in the copy constructor of island.
    virtual isl_inner_base *clone() const override final
    {
        return ::new isl_inner(m_value);
    }
    // The enqueue_evolution() method.
    virtual void enqueue_evolution(const algorithm &algo) override final
    {
        m_value.enqueue_evolution(algo);
    }
    // The wait() method.
    virtual void wait() const override final
    {
        m_value.wait();
    }
    // The get_population() method.
    virtual population get_population() const override final
    {
        return m_value.get_population();
    }
    // Serialization.
    template <typename Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<isl_inner_base>(this), m_value);
    }
    T m_value;
};
}

class island
{
public:
    template <typename Island, typename Algo>
    explicit island(Island &&isl, Algo &&a)
        : m_ptr(::new detail::isl_inner<uncvref_t<Island>>(std::forward<Island>(isl))), m_algo(std::forward<Algo>(a))
    {
    }
    void evolve()
    {
        m_ptr->enqueue_evolution(m_algo);
    }
    void wait() const
    {
        m_ptr->wait();
    }
    population get_population() const
    {
        return m_ptr->get_population();
    }

private:
    // Pointer to the inner base island
    std::unique_ptr<detail::isl_inner_base> m_ptr;
    algorithm m_algo;
};
}

#endif
