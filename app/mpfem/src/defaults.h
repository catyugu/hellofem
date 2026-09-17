// hellofem::app — the model defaults of the COMSOL physics the app solves
// SPDX-License-Identifier: MIT
#pragma once

namespace hellofem::app {

    /// COMSOL's reference temperature, 20 °C: the initial temperature of a
    /// heat transfer physics and the reference temperature a thermal
    /// expansion coupling measures the strain from, both of which a model
    /// feature may override.
    inline constexpr double reference_temperature = 293.15;

    /// COMSOL's element order for a physics interface the model does not
    /// give one: quadratic. It is independent of the geometry's order, so a
    /// linear mesh carries a quadratic field unless the model says otherwise.
    inline constexpr int default_element_order = 2;

} // namespace hellofem::app
