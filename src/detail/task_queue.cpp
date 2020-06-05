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
#include <utility>

#include <boost/function.hpp>
#include <boost/lockfree/queue.hpp>
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

                  if (this->m_tasks.empty()) {
                      // If we do not have more tasks check stop and park flags
                      if (this->m_stop) break;
                      if (this->m_park) this->m_parked.notify_one();
                  } else {
                      auto task(std::move(this->m_tasks.front()));
                      this->m_tasks.pop();
                      lock.unlock();
                      task();
                  }
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

namespace
{
// RAII helper to ensure destruction of parked task_queues
struct queue_holder {
    boost::lockfree::queue<task_queue *> m_queue{32};
    bool m_destruct = true;
    ~queue_holder() noexcept
    {
        static const auto deleter = boost::function<void(task_queue *)>([](task_queue *tq) { delete tq; });
        if (m_destruct) m_queue.consume_all(deleter);
    }
};

queue_holder park_q;
} // namespace

void task_queue::park(std::unique_ptr<task_queue> &&tq)
{
    std::unique_lock lock(tq->m_mutex);
    // wait for any remaining work to complete
    tq->m_park = true;
    tq->m_parked.wait(lock, [&tq = std::as_const(tq)]() { return tq->m_tasks.empty(); });
    park_q.m_queue.push(tq.release());
}

std::unique_ptr<task_queue> task_queue::unpark_or_construct()
{
    task_queue *tq = nullptr;
    park_q.m_queue.pop(tq);
    if (tq == nullptr) {
        return std::make_unique<task_queue>();
    } else {
        tq->m_park = false;
        return std::unique_ptr<task_queue>(tq);
    }
}

void task_queue::set_destruct_parked_task_queues(bool new_value)
{
    park_q.m_destruct = new_value;
}

} // namespace pagmo::detail
