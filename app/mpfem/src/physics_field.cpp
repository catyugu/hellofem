// hellofem::app — a physics field of a case: behavior, variables, registration
// SPDX-License-Identifier: MIT

#include "physics_field.h"

#include <stdexcept>
#include <utility>

namespace hellofem::app {
    namespace {

        /// The registered kinds, in registration order. A function-local
        /// static, so registration has no static initialization order to
        /// respect.
        std::vector<FieldKind>& kind_table()
        {
            static std::vector<FieldKind> kinds;
            return kinds;
        }

    } // namespace

    Variable scalar_variable(std::string name, std::string unit,
        std::shared_ptr<const fem::Function<double>> field)
    {
        return Variable {std::move(name), std::move(unit),
            [field = std::move(field)](std::span<const double> points,
                std::span<double> values) {
                auto [scalars, shape] = field->eval(points, {values.size(), 3});
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = scalars[i];
            }};
    }

    void register_field(FieldKind kind)
    {
        if (field_kind(kind.physics_type))
            throw std::logic_error("the physics '"
                + std::string(kind.physics_type) + "' is registered twice");
        kind_table().push_back(std::move(kind));
    }

    const FieldKind* field_kind(std::string_view physics_type)
    {
        for (const FieldKind& kind : kind_table())
            if (kind.physics_type == physics_type)
                return &kind;
        return nullptr;
    }

    // Each physics registers its own kind next to the physics it binds; the
    // linker keeps a registration-only object out of a static library, so
    // the list of the shipped fields has to be named here.
    void register_electric_field();
    void register_heat_field();
    void register_solid_field();

    void register_builtin_fields()
    {
        // A driver and a test may both set the app up, and the kinds are
        // process-wide: the registration happens once.
        static const bool registered = [] {
            register_electric_field();
            register_heat_field();
            register_solid_field();
            return true;
        }();
        (void)registered;
    }

} // namespace hellofem::app
