// hellofem::common — timing registry singleton
// SPDX-License-Identifier: MIT

#pragma once

#include "Table.h"

#include <chrono>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace hellofem::common {

    /// Time logger aggregating data collected by Timer, if registered.
    /// Monostate: timings accumulate into a single map.
    class TimeLogger {
    public:
        /// Singleton access.
        static TimeLogger& instance();

        /// Register a timing for later summary.
        void register_timing(std::string_view task,
                             std::chrono::duration<double, std::ratio<1>> wall);

        /// Summary of timings and tasks as a Table.
        Table timing_table() const;

        /// Timing for a given task: (count, total wall time).
        std::pair<int, std::chrono::duration<double, std::ratio<1>>>
        timing(std::string_view task) const;

        /// All logged elapsed times: task -> (count, total wall time).
        std::map<std::string,
                 std::pair<int, std::chrono::duration<double, std::ratio<1>>>,
                 std::less<>>
        timings() const;

    private:
        /// Constructor
        TimeLogger() = default;

        TimeLogger(const TimeLogger&) = delete;
        TimeLogger& operator=(const TimeLogger&) = delete;
        ~TimeLogger() = default;

        // task -> (num_timings, total_wall_time)
        std::map<std::string,
                 std::pair<int, std::chrono::duration<double, std::ratio<1>>>,
                 std::less<>>
            _timings;
    };

} // namespace hellofem::common
