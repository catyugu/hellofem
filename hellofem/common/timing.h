// hellofem::common — global timing convenience functions
// SPDX-License-Identifier: MIT

#pragma once

#include "Table.h"

#include <chrono>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace hellofem {

    /// Summary of all registered timings as a Table.
    Table timing_table();

    /// (count, total wall time) for a given task.
    std::pair<int, std::chrono::duration<double, std::ratio<1>>>
    timing(std::string_view task);

    /// All logged elapsed times: task -> (count, total wall time).
    std::map<std::string,
             std::pair<int, std::chrono::duration<double, std::ratio<1>>>,
             std::less<>>
    timings();

} // namespace hellofem
