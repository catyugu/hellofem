// hellofem::common — global timing convenience functions
// SPDX-License-Identifier: MIT

#include "timing.h"

#include "TimeLogger.h"

namespace hellofem {

    Table timing_table()
    {
        return common::TimeLogger::instance().timing_table();
    }

    std::pair<int, std::chrono::duration<double, std::ratio<1>>>
    timing(std::string_view task)
    {
        return common::TimeLogger::instance().timing(task);
    }

    std::map<std::string,
             std::pair<int, std::chrono::duration<double, std::ratio<1>>>,
             std::less<>>
    timings()
    {
        return common::TimeLogger::instance().timings();
    }

} // namespace hellofem
