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

#include <cassert>
#include <cstdlib>
#include <future>
#include <mutex>
#include <utility>

#include <pagmo/detail/task_queue.hpp>

namespace pagmo::detail
{

task_queue::task_queue()
    : m_thread([this]() {
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

                  // Pop the first task from the queue.
                  auto task(std::move(this->m_tasks.front()));
                  this->m_tasks.pop();

                  // Set m_task_maybe_running to true (even tohugh the task
                  // has not started yet) so that wait_all() will not
                  // exit before the task is finished.
                  assert(!m_task_maybe_running.load());
                  m_task_maybe_running.store(true);

                  // Release the lock, so that we can enqueue
                  // more tasks while the current one is running.
                  lock.unlock();

                  // Run the current task.
                  task();

                  // Task is finished, set m_task_maybe_running to false.
                  m_task_maybe_running.store(false);

                  // Notify that we popped a task from the queue
                  // and we consumed it. The only consumer of
                  // this notification is wait_all().
                  m_cond.notify_one();
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

// Helper to wait for all the enqueued tasks
// to be processed. This method is called by
// the island_data destructor in order to make sure that
// all tasks have finished before the queue is put
// into the cache.
void task_queue::wait_all()
{
    std::unique_lock lock(m_mutex);

    m_cond.wait(lock, [this]() {
        // NOTE: here we check for m_task_maybe_running (in addition to m_tasks
        // being empty) in order to deal with possible spurious wakeups of m_cond
        // while the last task is running. In such a case, we would be exiting
        // this function (because m_tasks is empty) while the last
        // task is still running, which would invalidate the guarantee
        // provided by this function that all running tasks
        // have finished by the time it returns.
        return m_task_maybe_running.load() == false && m_tasks.empty();
    });
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

} // namespace pagmo::detail
