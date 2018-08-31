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

#include <pagmo/serialization.hpp>

#define BOOST_TEST_MODULE cereal_thread_safety
#include <boost/test/included/unit_test.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

struct base1 {
    virtual ~base1() = default;
    virtual int do_something() const = 0;
    template <class Archive>
    void serialize(Archive &)
    {
    }
};

struct derived1 final : base1 {
    virtual int do_something() const override
    {
        return 42;
    }
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<base1>(this));
    }
};

struct derived2 final : base1 {
    virtual int do_something() const override
    {
        return 24;
    }
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<base1>(this));
    }
};

CEREAL_REGISTER_TYPE(derived1)
CEREAL_REGISTER_TYPE(derived2)

static std::mutex mutex;

static std::atomic<unsigned> a_counter(0u);

static const unsigned size = 16u;

template <typename Oa, typename Ia>
static inline void thread_func()
{
    ++a_counter;
    while (a_counter.load() != size) {
    }
    std::stringstream ss;
    {
        Oa oarchive(ss);

        // Create instances of the derived classes, but only keep base class pointers
        std::unique_ptr<base1> ptr1(new derived1());
        std::unique_ptr<base1> ptr2(new derived2());
        oarchive(ptr1);
        oarchive(ptr2);
    }

    {
        Ia iarchive(ss);

        // De-serialize the data as base class pointers, and watch as they are
        // re-instantiated as derived classes
        std::unique_ptr<base1> ptr1, ptr2;
        iarchive(ptr1);
        iarchive(ptr2);

        std::lock_guard<std::mutex> lock(mutex);
        BOOST_CHECK_EQUAL(ptr1->do_something(), 42);
        BOOST_CHECK_EQUAL(ptr2->do_something(), 24);
    }
}

template <typename Oa, typename Ia>
static inline void test_archive()
{
    a_counter.store(0u);

    std::vector<std::thread> threads;
    threads.reserve(size);

    for (auto i = 0u; i < size; ++i) {
        threads.emplace_back(thread_func<Oa, Ia>);
    }

    for (auto &t : threads) {
        t.join();
    }
}

BOOST_AUTO_TEST_CASE(cereal_thread_safety_test_00)
{
    test_archive<cereal::BinaryOutputArchive, cereal::BinaryInputArchive>();
    test_archive<cereal::PortableBinaryOutputArchive, cereal::PortableBinaryInputArchive>();
    test_archive<cereal::JSONOutputArchive, cereal::JSONInputArchive>();
}