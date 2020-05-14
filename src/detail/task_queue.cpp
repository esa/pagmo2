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

#include <tbb/concurrent_queue.h>

#include <pagmo/detail/task_queue.hpp>

namespace pagmo
{

namespace detail
{

task_queue_thread::task_queue_thread()
    : m_park_req(false), m_stop_req(false), m_queue(nullptr), m_thread(std::thread([this]() {
          try {
              while (true) {
                  {
                      // Wait for the thread to be awakened from parking
                      std::unique_lock<std::mutex> lock(m_mutex);
                      m_unpark.wait(lock, [&]() { return state() != PARKED; });
                      // Possible states here: STOPPED, WAITING_FOR_WORK, STOPPING (unlikely), PARKING (unlikely)
                      if (state() == STOPPED) return;
                  }
                  // Possible states here: WAITING_FOR_WORK, STOPPING (unlikely), PARKING (unlikely)
                  while (true) {
                      std::unique_lock<std::mutex> lock(m_mutex);
                      m_main.wait(lock, [&]() { return state() != WAITING_FOR_WORK || !m_queue->m_tasks.empty(); });
                      // Possible states here: WAITING_FOR_WORK, STOPPING, PARKING
                      if (not m_queue->m_tasks.empty()) {
                          // NOTE: move constructor of std::function could throw, unfortunately.
                          std::function<void()> task(std::move(m_queue->m_tasks.front()));
                          m_queue->m_tasks.pop();
                          lock.unlock();

                          task();
                      } else if (state() != WAITING_FOR_WORK) {
                          break;
                      }
                  }
                  // Possible states here: STOPPING, PARKING
                  if (state() == STOPPING) {
                      // Disassociate from the queue and exit
                      m_queue = nullptr;
                      // Possible states here: STOPPED
                      return;
                  } else {
                      std::unique_lock<std::mutex> lock(m_mutex);
                      // Disassociate from the queue and reset the flag
                      m_queue = nullptr;
                      m_park_req = false;
                  }
                  // Signal we've completed all work and are now fully parked
                  m_parked.notify_all();
                  // Possible states here: PARKED
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

static tbb::concurrent_queue<std::unique_ptr<task_queue_thread>> task_queue_thread_park{};

std::unique_ptr<task_queue_thread> task_queue_thread::get(task_queue *new_owner)
{
    std::unique_ptr<task_queue_thread> thread;
    if (not task_queue_thread_park.try_pop(thread)) {
        thread = std::make_unique<task_queue_thread>();
    }
    thread->unpark(new_owner);
    return thread;
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
void task_queue_thread::stop(bool block)
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop_req = true;
    }
    // Notify the thread of the stop request (wherever it might be waiting)
    m_main.notify_all();
    m_unpark.notify_all();

    // Wait for it to consume the remaining tasks and exit.
    if (block && m_thread.joinable()) {
        m_thread.join();
    }
}

void task_queue_thread::park(bool block)
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        // If needed, wait for the thread to be unparked
        if (state() != WAITING_FOR_WORK) {
            pagmo_throw(std::runtime_error, "cannot park thread that isn't waiting for work");
        }
        // Request that the thread park itself
        m_park_req = true;
    }
    // Notify the thread that queue a park has been requested
    m_main.notify_all();

    if (block) {
        std::unique_lock<std::mutex> lock(m_mutex);
        // Wait for the thread to complete all work in this queue. We know this has
        // happened when the thread resets its pointer to this task_queue
        m_parked.wait(lock, [&]() { return state() == PARKED; });
    }
}

void task_queue_thread::unpark(task_queue *queue)
{
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        // If needed, wait for the thread to be parked
        if (state() != PARKED) {
            m_main.notify_all();
            m_parked.wait(lock, [&]() { return state() == PARKED; });
        }
        m_queue = queue;
    }
    // Notify the thread it has a new owner
    m_unpark.notify_all();
}
task_queue_thread::tq_thread_state task_queue_thread::state() const
{
    if (m_stop_req) {
        if (m_queue == nullptr) {
            return STOPPED;
        } else {
            return STOPPING;
        }
    } else {
        if (m_park_req) {
            if (m_queue == nullptr) {
                pagmo_throw(std::runtime_error, "thread without queue found with park_req set, which shouldn't happen");
            } else {
                return PARKING;
            }
        } else {
            if (m_queue == nullptr) {
                return PARKED;
            } else {
                return WAITING_FOR_WORK;
            }
        }
    }
}

task_queue::task_queue() : m_thread(task_queue_thread::get(this)) {}

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
