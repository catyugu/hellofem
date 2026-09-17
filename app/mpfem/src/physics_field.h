// hellofem::app — a physics field of a case: behavior, variables, registration
// SPDX-License-Identifier: MIT
#pragma once

#include "case_context.h"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace hellofem::app {

    /// A quantity a physics exports: the name the model's Data export refers
    /// to, its unit as COMSOL writes it, and its value at given points.
    struct Variable {
        std::string name; // "T", "solid.disp", ...
        std::string unit; // "(K)", "(m)", ...
        /// Write the value at every point into `values`, one entry per
        /// point; `points` holds three coordinates per point and `cells`
        /// the cell each point is read on (negative for a point that has
        /// none), so that no cell has to be searched for.
        std::function<void(std::span<const double> points,
            std::span<const std::int32_t> cells, std::span<double> values)>
            eval;
    };

    /// The values of a scalar field as a `Variable` evaluator.
    Variable scalar_variable(std::string name, std::string unit,
        std::shared_ptr<const fem::Function<double>> field);

    /// One physics field of a case: the solver behind it, the variables it
    /// exports and the one-level step of the study. A field is built from one
    /// COMSOL physics interface of the model and registers its kind with
    /// `register_field`, so the case scheduler knows no physics by name.
    class PhysicsField {
    public:
        virtual ~PhysicsField() = default;

        /// Bring the field to its state at the first level of the study,
        /// which the scheduler prepares before it solves anything: the
        /// initial values of a field the study advances (see `solve_level`),
        /// and the pointwise constraints, which a state that is not the
        /// result of a solve does not satisfy. Called once per study.
        virtual void initialize(double t0) = 0;

        /// Whether the study advances this field in time, i.e. whether
        /// `solve_level` steps the field's time scheme rather than solving
        /// its steady system.
        ///
        /// The first level of a study is a solved one for every field this
        /// returns false for — a state that is algebraic has to be solved
        /// there like at any other level — and the state `initialize`
        /// prepared for the others.
        virtual bool advances_in_time() const { return false; }

        /// Solve the level at time `t`: the steady system, or one step of the
        /// time scheme for a field the study advances in time.
        virtual void solve_level(double t) = 0;

        /// Apply the multiphysics couplings this physics owns. Called once,
        /// after every field of the case is built, so the solutions the other
        /// fields published are available through the context.
        virtual void bind_couplings(CaseContext&) {}

        /// The quantities the model's export can name.
        virtual std::span<const Variable> variables() const = 0;
    };

    /// A physics the app can solve: the COMSOL physics interface type it
    /// answers for and the factory that binds it to a case.
    using FieldFactory = std::unique_ptr<PhysicsField> (*)(
        const Physics&, CaseContext&);

    struct FieldKind {
        std::string_view physics_type; // "HeatTransfer", "ConductiveMedia", ...
        FieldFactory create;
    };

    /// Register a field kind. The fields of a case are built from the kinds
    /// registered here, in the order of the model's physics; registering a
    /// physics type twice is an error.
    void register_field(FieldKind kind);

    /// The kind answering for `physics_type`, or nullptr.
    const FieldKind* field_kind(std::string_view physics_type);

    /// Registration of a field kind at program start-up: a physics holds one
    /// of these next to the physics it binds, so that the app knows the
    /// field without naming it anywhere else.
    ///
    /// @note An object that nothing else refers to is dropped from a static
    /// library, so the consumers link the app library whole (see
    /// `app/CMakeLists.txt`); without that the field would silently never
    /// load.
    class FieldRegistration {
    public:
        FieldRegistration(std::string_view physics_type, FieldFactory create)
        {
            register_field(FieldKind {physics_type, create});
        }
    };

} // namespace hellofem::app
