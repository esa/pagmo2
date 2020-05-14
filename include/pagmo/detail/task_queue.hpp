/* Copyright 2017-2020 PaGMO development team

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

#ifndef PAGMO_TASK_QUEUE_HPP
#define PAGMO_TASK_QUEUE_HPP

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>

#include <pagmo/detail/visibility.hpp>
#include <pagmo/exceptions.hpp>

namespace pagmo
{

namespace detail
{

struct task_queue;
struct task_queue_thread;

static std::mutex task_queue_thread_park_mutex{};
static std::queue<std::unique_ptr<task_queue_thread>> task_queue_thread_park{};

struct PAGMO_DLL_PUBLIC task_queue_thread {
    task_queue_thread();
    ~task_queue_thread();
    static std::unique_ptr<task_queue_thread> get()
    {
        if (task_queue_thread_park.empty()) {
            return std::make_unique<task_queue_thread>();
        } else {
            std::unique_lock lock{task_queue_thread_park_mutex};
            // Repeat the emptiness check as the answer could have changed while
            // waiting for the lock to be granted.
            if (task_queue_thread_park.empty()) {
                return std::make_unique<task_queue_thread>();
            } else {
                auto thread = std::move(task_queue_thread_park.front());
                task_queue_thread_park.pop();
                return thread;
            }
        }
    }
    // NOTE: we call this only from dtor, it is here in order to be able to test it.
    // So the exception handling in dtor will suffice, keep it in mind if things change.
    void stop();
    void park();
    void unpark(task_queue *queue);
    // Data members.
    bool m_park, m_stop;
    std::condition_variable m_cond, m_park_cond;
    std::mutex m_mutex, m_park_mutex;
    task_queue *m_queue;
    std::thread m_thread;
};

struct PAGMO_DLL_PUBLIC task_queue {
    task_queue();
    ~task_queue();
    // Main enqueue function.
    template <typename F>
    std::future<void> enqueue(F &&f)
    {
        using p_task_type = std::packaged_task<void()>;
        // NOTE: here we have a 2-stage construction of the task:
        // - std::packaged_task gives us the std::future machinery,
        // - std::function (in m_tasks) gives the uniform type interface via type erasure.
        auto task = std::make_shared<p_task_type>(std::forward<F>(f));
        std::future<void> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_thread->m_mutex);
            if (m_thread->m_queue == nullptr) {
                pagmo_throw(std::runtime_error, "cannot enqueue task if the thread is parked");
            } else if (m_thread->m_queue != this) {
                pagmo_throw(std::runtime_error, "cannot enqueue task if the thread is no longer watching this queue");
            } else if (m_thread->m_park) {
                pagmo_throw(std::runtime_error, "cannot enqueue task if the task queue thread is parking");
            } else {
                m_tasks.push([task]() { (*task)(); });
                // NOTE: notify_one is noexcept.
                m_thread->m_cond.notify_one();
            }
        }
        return res;
    }
    // NOTE: we call this only from dtor, it is here in order to be able to test it.
    // So the exception handling in dtor will suffice, keep it in mind if things change.
    void stop();
    // Data members.
    std::queue<std::function<void()>> m_tasks;
    std::unique_ptr<task_queue_thread> m_thread;
};

} // namespace detail
} // namespace pagmo

#endif
