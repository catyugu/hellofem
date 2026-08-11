// hellofem::common — build-time configuration queries
// SPDX-License-Identifier: MIT

#pragma once

#include "config.h"

#include <string_view>

namespace hellofem::common {

    /// Version string of the hellofem library.
    constexpr std::string_view version() { return HELLOFEM_VERSION; }

} // namespace hellofem::common
