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

#ifndef PAGMO_DETAIL_TASK_QUEUE_HPP
#define PAGMO_DETAIL_TASK_QUEUE_HPP

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#include <pagmo/detail/visibility.hpp>

namespace pagmo::detail
{

struct PAGMO_DLL_PUBLIC task_queue {
    task_queue();
    ~task_queue();

    // Make extra sure we never try to move/copy.
    task_queue(const task_queue &) = delete;
    task_queue(task_queue &&) = delete;
    task_queue &operator=(const task_queue &) = delete;
    task_queue &operator=(task_queue &&) = delete;

    using task_type = std::packaged_task<void()>;
    // Main enqueue function.
    std::future<void> enqueue_impl(task_type &&);
    template <typename F>
    std::future<void> enqueue(F &&f)
    {
        return enqueue_impl(task_type(std::forward<F>(f)));
    }

    void wait_all();

    // Data members.
    bool m_stop = false;
    // NOTE: if this flag is set to false,
    // it means that no task is currently
    // being executed. Note that the opposite is not
    // true, i.e., the flag could be true
    // without any running task.
    std::atomic<bool> m_task_running = false;
    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::queue<task_type> m_tasks;
    // NOTE: it's important that this comes last,
    // so that when the thread starts all other
    // members have been inited.
    std::thread m_thread;
};

} // namespace pagmo::detail

#endif
