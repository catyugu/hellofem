// hellofem::app — a physics field of a case: behavior, variables, registration
// SPDX-License-Identifier: MIT

#include "physics_field.h"

#include <algorithm>
#include <limits>
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
                std::span<const std::int32_t> cells,
                std::span<double> values) {
                field->eval(points, {values.size(), 3}, cells, values,
                    {values.size(), 1});
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

    double study_time_tolerance(const std::vector<Physics>& physics)
    {
        if (physics.empty())
            throw std::runtime_error(
                "study_time_tolerance: the model has no physics");
        double tolerance = std::numeric_limits<double>::max();
        for (const Physics& p : physics) {
            const FieldKind* kind = field_kind(p.type);
            if (not kind)
                throw std::runtime_error(
                    "study_time_tolerance: the physics '" + p.type
                    + "' has no registered field");
            tolerance = std::min(tolerance, kind->time_tolerance);
        }
        return tolerance;
    }

} // namespace hellofem::app
