// hellofem::app — COMSOL-style unit parsing
// SPDX-License-Identifier: MIT
#pragma once

#include <string_view>

namespace hellofem::app {

    /// Parse a unit string ("mm", "W/(m*K)", "kg*m/s^2", ...) to its SI
    /// multiplicative factor. Returns 1.0 for an empty unit.
    double parse_unit(std::string_view unit);

    /// Parse a value-with-optional-unit literal: "0.006" -> 0.006,
    /// "20[mV]" -> 0.02, "5[W/m^2/K]" -> 5.0 (already SI).
    double parse_si(std::string_view input);

} // namespace hellofem::app
