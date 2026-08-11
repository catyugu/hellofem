// hellofem::common — Timer for measuring and logging elapsed durations
// SPDX-License-Identifier: MIT

#pragma once

#include "TimeLogger.h"

#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace hellofem::common {

    /// Timer for measuring and logging elapsed time durations.
    ///
    /// \code{.cpp}
    /// Timer timer("Assembling over cells");
    /// \endcode
    ///
    /// The timer starts at construction; elapsed time is registered when the
    /// timer goes out of scope or Timer::flush() is called. start()/stop()/
    /// resume() give explicit control. A summary of all registered timings is
    /// printed by timing_table().
    template <typename T = std::chrono::high_resolution_clock>
    class Timer {
    public:
        /// Create and start a timer. If `task` is set, the elapsed time is
        /// registered in the logger on destruction.
        Timer(std::optional<std::string> task = std::nullopt)
            : _task(std::move(task))
        {
        }

        /// Stop a running timer and register its elapsed time, if a task name
        /// was set.
        ~Timer()
        {
            if (_start_time.has_value() and _task.has_value()) {
                _acc += T::now() - *_start_time;
                TimeLogger::instance().register_timing(*_task, _acc);
            }
        }

        /// Reset elapsed time and (re-)start.
        void start()
        {
            _acc = T::duration::zero();
            _start_time = T::now();
        }

        /// Elapsed time since the timer was started.
        template <typename Period = std::ratio<1>>
        std::chrono::duration<double, Period> elapsed() const
        {
            if (_start_time.has_value())
                return T::now() - *_start_time + _acc;
            else
                return _acc;
        }

        /// Stop the timer and return elapsed time.
        template <typename Period = std::ratio<1>>
        std::chrono::duration<double, Period> stop()
        {
            if (_start_time.has_value()) {
                _acc += T::now() - *_start_time;
                _start_time = std::nullopt;
            }
            return _acc;
        }

        /// Resume a stopped timer (no-op if running).
        void resume()
        {
            if (!_start_time.has_value())
                _start_time = T::now();
        }

        /// Flush the elapsed time to the logger. The timer must be stopped;
        /// it can be flushed only once.
        void flush()
        {
            if (_start_time.has_value())
                throw std::runtime_error(
                    "Timer must be stopped before flushing.");

            if (_task.has_value()) {
                TimeLogger::instance().register_timing(*_task, _acc);
                _task = std::nullopt;
            }
        }

    private:
        // Task name to register in the logger
        std::optional<std::string> _task;

        // Accumulated elapsed time
        typename T::duration _acc = T::duration::zero();

        // Start time (std::nullopt when stopped)
        std::optional<typename T::time_point> _start_time = T::now();
    };

} // namespace hellofem::common
