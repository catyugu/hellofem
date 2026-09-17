// hellofem::app — the model defaults of the COMSOL physics the app solves
// SPDX-License-Identifier: MIT
#pragma once

namespace hellofem::app {

    /// COMSOL's reference temperature, 20 °C: the initial temperature of a
    /// heat transfer physics and the reference temperature a thermal
    /// expansion coupling measures the strain from, both of which a model
    /// feature may override.
    inline constexpr double reference_temperature = 293.15;

} // namespace hellofem::app
