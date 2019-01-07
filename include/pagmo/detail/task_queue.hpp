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

#ifndef PAGMO_TASK_QUEUE_HPP
#define PAGMO_TASK_QUEUE_HPP

#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>

#include <pagmo/exceptions.hpp>

namespace pagmo
{

namespace detail
{

struct task_queue {
    task_queue() : m_stop(false)
    {
        m_thread = std::thread([this]() {
            try {
                while (true) {
                    std::unique_lock<std::mutex> lock(this->m_mutex);
                    while (!this->m_stop && this->m_tasks.empty()) {
                        // Need to wait for something to happen only if the task
                        // list is empty and we are not stopping.
                        // NOTE: wait will be noexcept in C++14.
                        this->m_cond.wait(lock);
                    }
                    if (this->m_stop && this->m_tasks.empty()) {
                        // If the stop flag was set, and we do not have more tasks,
                        // just exit.
                        break;
                    }
                    // NOTE: move constructor of std::function could throw, unfortunately.
                    std::function<void()> task(std::move(this->m_tasks.front()));
                    this->m_tasks.pop();
                    lock.unlock();
                    task();
                }
                // LCOV_EXCL_START
            } catch (...) {
                // The errors we could get here are:
                // - threading primitives,
                // - move-construction of std::function,
                // - queue popping (I guess unlikely, as the destructor of std::function
                //   is noexcept).
                // In any case, not much that can be done to recover from this, better to abort.
                // NOTE: logging candidate.
                std::abort();
                // LCOV_EXCL_STOP
            }
        });
    }
    ~task_queue()
    {
        // NOTE: logging candidate (catch any exception,
        // log it and abort as there is not much we can do).
        try {
            stop();
            // LCOV_EXCL_START
        } catch (...) {
            std::abort();
            // LCOV_EXCL_STOP
        }
    }
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
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_stop) {
                // Enqueueing is not allowed if the queue is stopped.
                pagmo_throw(std::runtime_error, "cannot enqueue task while the task queue is stopping");
            }
            m_tasks.push([task]() { (*task)(); });
        }
        // NOTE: notify_one is noexcept.
        m_cond.notify_one();
        return res;
    }
    // NOTE: we call this only from dtor, it is here in order to be able to test it.
    // So the exception handling in dtor will suffice, keep it in mind if things change.
    void stop()
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_stop) {
                // Already stopped.
                return;
            }
            m_stop = true;
        }
        // Notify the thread that queue has been stopped, wait for it
        // to consume the remaining tasks and exit.
        m_cond.notify_one();
        m_thread.join();
    }
    // Data members.
    bool m_stop;
    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::queue<std::function<void()>> m_tasks;
    std::thread m_thread;
};
} // namespace detail
} // namespace pagmo

#endif
