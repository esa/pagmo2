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

#include <cassert>
#include <cstdlib>
#include <future>
#include <tbb/concurrent_queue.h>
#include <utility>

#include <pagmo/detail/task_queue.hpp>

namespace pagmo::detail
{

task_queue::task_queue()
    : m_stop(false), m_thread([this]() {
          try {
              while (true) {
                  std::unique_lock lock(this->m_mutex);

                  // NOTE: stop waiting if either we are stopping the queue
                  // or there are tasks to be consumed.
                  // NOTE: wait() is noexcept.
                  this->m_cond.wait(lock, [this]() { return this->m_stop || !this->m_tasks.empty(); });

                  if (this->m_stop && this->m_tasks.empty()) {
                      // If the stop flag was set, and we do not have more tasks,
                      // just exit.
                      break;
                  }

                  auto task(std::move(this->m_tasks.front()));
                  this->m_tasks.pop();
                  lock.unlock();
                  task();
              }
              // LCOV_EXCL_START
          } catch (...) {
              // The errors we could get here are:
              // - threading primitives,
              // - queue popping (I guess unlikely, as the destructor of
              //   std::packaged_task is noexcept).
              // In any case, not much that can be done to recover from this, better to abort.
              // NOTE: logging candidate.
              std::abort();
              // LCOV_EXCL_STOP
          }
      })
{
}

std::future<void> task_queue::enqueue_impl(task_type &&task)
{
    auto res = task.get_future();
    assert(!m_stop);
    {
        std::unique_lock lock(m_mutex);
        m_tasks.push(std::move(task));
    }
    // NOTE: notify_one is noexcept.
    m_cond.notify_one();
    return res;
}

task_queue::~task_queue()
{
    // NOTE: logging candidate (catch any exception,
    // log it and abort as there is not much we can do).
    try {
        {
            std::unique_lock lock(m_mutex);
            assert(!m_stop);
            m_stop = true;
        }
        // Notify the thread that queue has been stopped, wait for it
        // to consume the remaining tasks and exit.
        m_cond.notify_one();
        m_thread.join();
        // LCOV_EXCL_START
    } catch (...) {
        std::abort();
        // LCOV_EXCL_STOP
    }
}

// Glorified version of the initialise on first use idiom
// Guaranteed to return nullptr if and only if reset is true
auto &park_q(const bool reset = false)
{
    // Need to store a pointer to tbb container as tbb::concurrent_queue is not assignable
    using part_q_t = tbb::concurrent_queue<std::unique_ptr<task_queue>>;
    static auto q = std::unique_ptr<part_q_t>();

    const bool q_is_null = !q;
    if (reset) {
        if (q_is_null) return q;
        return q = std::unique_ptr<part_q_t>();
    }
    if (q_is_null) q = std::make_unique<part_q_t>();
    return q;
}

// Used by the fork_island
void task_queue::reset_park_q()
{
    park_q(true);
}

void task_queue::park(std::unique_ptr<task_queue> &&tq)
{
    park_q()->push(std::move(tq));
}

std::unique_ptr<task_queue> task_queue::unpark_or_construct()
{
    std::unique_ptr<task_queue> tq;
    if (park_q()->try_pop(tq)) return tq;
    return std::make_unique<task_queue>();
}

} // namespace pagmo::detail
