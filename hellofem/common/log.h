// hellofem::common — logging backend setup (spdlog)
// SPDX-License-Identifier: MIT

#pragma once

#include <spdlog/spdlog.h>

namespace hellofem::common {

    /// Optional initialisation of the spdlog backend. Log verbosity can be
    /// controlled from the command line with `SPDLOG_LEVEL=<level>`, where
    /// `<level>` is one of trace, debug, info, warn, error, critical.
    void init_logging(int argc, char* argv[]);

} // namespace hellofem::common
