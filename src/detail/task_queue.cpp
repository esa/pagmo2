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

#include <boost/lockfree/queue.hpp>
#include <cassert>
#include <cstdlib>
#include <future>
#include <utility>

#include <boost/function.hpp>
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
// Guaranteed to return null ptr if and only if reset is true
auto &park_q(const bool reset = false)
{
    // boost::lockfree::queue requires trivial destructor so we can't use unique_ptr's. However
    // think of these pointers as "owning", they will be converted to unique_ptrs before being
    // returned by unpark_or_construct and we make sure we delete them before resetting the queue
    using park_q_t = boost::lockfree::queue<task_queue *>;
    static auto q = std::unique_ptr<park_q_t>();

    const bool q_is_null = !q;
    if (reset) {
        if (q_is_null) return q;
        // Manually destruct the parked task_queues
        // Can't use unique_ptr as boost requires trivial destructor
        const auto deleter = boost::function<void(task_queue *)>([](task_queue *tq) { delete tq; });
        q->consume_all(deleter);
        // Set and return null ptr
        return q = std::unique_ptr<park_q_t>();
    }
    if (q_is_null)
        // Create new park queue with initial capacity 16
        q = std::make_unique<park_q_t>(16);
    return q;
}

void task_queue::reset_park_q()
{
    park_q(true);
}

void task_queue::park(std::unique_ptr<task_queue> &&tq)
{
    park_q()->push(tq.release());
}

std::unique_ptr<task_queue> task_queue::unpark_or_construct()
{
    task_queue *tq;
    // Try popping a task_queue out of the park, if that succeeds
    // construct and return an owning pointer to the popped instance
    if (park_q()->pop(tq)) return std::unique_ptr<task_queue>(tq);
    // Otherwise return an owning pointer to a new task_queue
    return std::make_unique<task_queue>();
}

} // namespace pagmo::detail
