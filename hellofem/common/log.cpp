// hellofem::common — logging backend setup (spdlog)
// SPDX-License-Identifier: MIT

#include "log.h"

#include <spdlog/cfg/argv.h>

namespace hellofem::common {

    void init_logging(int argc, char* argv[])
    {
        // Start quiet; SPDLOG_LEVEL on the command line can raise the level.
        spdlog::set_level(spdlog::level::warn);
        spdlog::cfg::load_argv_levels(argc, argv);
    }

} // namespace hellofem::common
