// hellofem::common — timing registry implementation
// SPDX-License-Identifier: MIT

#include "TimeLogger.h"

#include "log.h"

#include <format>
#include <stdexcept>

using namespace hellofem;
using namespace hellofem::common;

TimeLogger& TimeLogger::instance()
{
    static TimeLogger _instance{};
    return _instance;
}

void TimeLogger::register_timing(
    std::string_view task, std::chrono::duration<double, std::ratio<1>> time)
{
    spdlog::debug("Elapsed time: {} ({})", time.count(), task);

    if (auto it = _timings.find(task); it != _timings.end()) {
        std::get<0>(it->second) += 1;
        std::get<1>(it->second) += time;
    }
    else
        _timings.insert({std::string(task), {1, time}});
}

Table TimeLogger::timing_table() const
{
    Table table("Summary of timings (s)");
    for (const auto& [task, data] : _timings) {
        const auto [num_timings, time] = data;
        table.set(task, "reps", num_timings);
        table.set(task, "avg", time.count() / static_cast<double>(num_timings));
        table.set(task, "tot", time.count());
    }
    return table;
}

std::pair<int, std::chrono::duration<double, std::ratio<1>>>
TimeLogger::timing(std::string_view task) const
{
    auto it = _timings.find(task);
    if (it == _timings.end()) {
        throw std::runtime_error(
            std::format("No timings registered for task \"{}\".", task));
    }
    return it->second;
}

std::map<std::string,
         std::pair<int, std::chrono::duration<double, std::ratio<1>>>,
         std::less<>>
TimeLogger::timings() const
{
    return _timings;
}
