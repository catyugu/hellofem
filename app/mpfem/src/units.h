// hellofem::app — COMSOL-style unit parsing and conversion
// SPDX-License-Identifier: MIT
#pragma once

#include <string_view>

namespace hellofem::app {

    /// `value` stated in `unit`, in SI: the conversion of one quantity.
    ///
    /// Every unit is multiplicative except the absolute temperature scales,
    /// which are offsets (`30[degC]` is 303.15 K). This is the one place the
    /// two kinds are told apart, and the one place a quantity crosses between
    /// the unit a model states it in and the SI the solver works in.
    double to_si(double value, std::string_view unit);

    /// An SI `value` stated in `unit` — the inverse of `to_si`.
    double from_si(double value, std::string_view unit);

    /// Parse a value-with-optional-unit literal: "0.006" -> 0.006,
    /// "20[mV]" -> 0.02, "5[W/m^2/K]" -> 5.0 (already SI), "30[degC]" ->
    /// 303.15.
    double parse_si(std::string_view input);

} // namespace hellofem::app
