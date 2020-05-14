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

#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include <pagmo/detail/task_queue.hpp>

namespace pagmo
{

namespace detail
{

task_queue_thread::task_queue_thread()
    : m_park(true), m_stop(false), m_queue(nullptr), m_thread(std::thread([this]() {
          try {
              while (true) {
                  // Wait for the thread to be awakened from parking by having
                  // its new queue associated by a call to unpark(task_queue*)
                  {
                      std::unique_lock<std::mutex> lock(this->m_mutex);
                      while (!this->m_stop && this->m_queue == nullptr) {
                          this->m_cond.wait(lock);
                      }
                  }
                  // Check if we've been asked to stop
                  if (this->m_queue == nullptr && this->m_stop) return;
                  // Otherwise we've been unparked, so set the flag accordingly
                  // and inform the requesting thread
                  {
                      std::unique_lock<std::mutex> lock(this->m_park_mutex);
                      this->m_park = false;
                  }
                  this->m_park_cond.notify_one();
                  // Process work from the queue
                  while (true) {
                      std::unique_lock<std::mutex> lock(this->m_mutex);
                      while (!this->m_park && !this->m_stop && this->m_queue->m_tasks.empty()) {
                          this->m_cond.wait(lock);
                      }
                      if (this->m_queue->m_tasks.empty()) {
                          // If we have no more tasks react to park or stop flags
                          if (this->m_park) break;
                          if (this->m_stop) return;
                      } else {
                          // NOTE: move constructor of std::function could throw, unfortunately.
                          std::function<void()> task(std::move(this->m_queue->m_tasks.front()));
                          this->m_queue->m_tasks.pop();
                          lock.unlock();

                          task();
                      }
                  }
                  // We've been parked. Disassociate from the queue which
                  // triggered this, and signal we've completed all work and are
                  // now fully parked
                  {
                      std::unique_lock<std::mutex> lock(this->m_park_mutex);
                      this->m_queue = nullptr;
                  }
                  this->m_park_cond.notify_one();
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
      }))
{
}

task_queue_thread::~task_queue_thread()
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

// NOTE: we call this only from dtor, it is here in order to be able to test it.
// So the exception handling in dtor will suffice, keep it in mind if things change.
void task_queue_thread::stop()
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_stop) return;
        m_stop = true;
    }
    // Notify the thread that queue has been stopped, wait for it
    // to consume the remaining tasks and exit.
    m_cond.notify_one();
    m_thread.join();
}

void task_queue_thread::park()
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_park || m_queue == nullptr) {
            pagmo_throw(std::runtime_error, "Cannot park a thread that's already parked");
        }
        m_park = true;
    }
    // Notify the thread that queue has been parked
    m_cond.notify_one();
    // Wait for the thread to complete all work in this queue. We know this has
    // happened when the thread resets its pointer to this task_queue
    std::unique_lock<std::mutex> lock(m_park_mutex);
    while (m_queue != nullptr) {
        m_park_cond.wait(lock);
    }
}

void task_queue_thread::unpark(task_queue *queue)
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!(m_park && m_queue == nullptr)) {
            pagmo_throw(std::runtime_error, "Cannot unpark a thread that's not parked");
        }
        m_queue = queue;
    }
    // Notify the thread that queue has been unparked
    m_cond.notify_one();
    // Wait for the thread to come out of park
    std::unique_lock<std::mutex> lock(m_park_mutex);
    while (m_park) {
        m_park_cond.wait(lock);
    }
}

task_queue::task_queue() : m_thread(task_queue_thread::get())
{
    m_thread->unpark(this);
}

task_queue::~task_queue()
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

// NOTE: we call this only from dtor, it is here in order to be able to test it.
// So the exception handling in dtor will suffice, keep it in mind if things change.
void task_queue::stop()
{
    m_thread->park();
    task_queue_thread_park.push(std::move(m_thread));
}

} // namespace detail

} // namespace pagmo
