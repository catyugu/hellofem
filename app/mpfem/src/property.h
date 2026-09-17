// hellofem::app — model material properties and scalar model expressions
// SPDX-License-Identifier: MIT
#pragma once

#include "Expression.h"
#include "fem/Function.h"
#include "la/Vector.h"
#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hellofem::app {

    /// A scalar model value: a constant or an expression over the model
    /// parameters, the coordinates and time. Evaluation happens at a point
    /// `(x,y,z,t)`; the parameter values are captured at construction.
    class ScalarExpression {
    public:
        ScalarExpression() = default;
        explicit ScalarExpression(double value)
            : value_(value)
        {
        }

        /// Compile `text` with the model parameters bound by value.
        ScalarExpression(std::string_view text,
            const std::unordered_map<std::string, double>& params = {});

        /// Value at a point.
        double eval(double x, double y, double z, double t) const
        {
            return impl_ ? impl_->expr.eval(x, y, z, t) : value_;
        }

    private:
        // Shared so that the value is copyable together with the variable
        // storage the parsed expression reads (immutable after construction).
        struct Impl {
            std::unordered_map<std::string, double> vars;
            mutable Expression expr;
        };
        double value_ = 0.0;
        std::shared_ptr<const Impl> impl_;
    };

    /// A cell-wise (DG0) material property: a value per COMSOL domain,
    /// either a constant or a model expression. An expression may reference
    /// the model parameters, the coordinates, time and the solution fields
    /// bound with `bind_field`; a bound field shadows a parameter of the
    /// same name (as in COMSOL, where the dependent variable wins inside a
    /// physics-owned expression).
    class CellProperty {
    public:
        /// @param[in] params Model parameters, bound by value.
        CellProperty(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags,
            std::unordered_map<std::string, double> params);

        /// Set a model expression on domain `dom` (1-based COMSOL domain id).
        /// An expression that reads a bound field makes the property
        /// solution dependent.
        void set_expression(int dom, std::string_view text);

        /// Bind the solution field that `symbol` refers to in the
        /// expressions (scalar field). Call before `set_expression`.
        ///
        /// The field is read at the cell centroids of the mesh this property
        /// was built on, so it must live on that same mesh.
        void bind_field(std::string symbol,
            std::shared_ptr<const fem::Function<double>> field);

        /// Whether any expression reads a bound solution field, i.e.
        /// whether the property has to be re-evaluated as the solution
        /// changes.
        bool field_dependent() const { return field_dependent_; }

        /// Re-evaluate every expression-based value at time `t`. Field-bound
        /// expressions read the field values at the cell centroids.
        void update(double t);

        /// The per-cell values as a DG0 function.
        std::shared_ptr<fem::Function<double>> function() const { return f_; }

    private:
        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags_;
        std::shared_ptr<fem::Function<double>> f_;
        /// Expression per domain; a domain without one keeps the zero of `f_`.
        std::map<int, std::shared_ptr<Expression>> values_;
        bool field_dependent_ = false;

        // Variable storage shared by the parsed expressions: the model
        // parameters plus one slot per bound field. Slots are never
        // re-created once an expression holds their address.
        std::unordered_map<std::string, double> vars_;
        std::unordered_map<std::string, double*> var_ptrs_;

        struct BoundField {
            std::string symbol;
            std::shared_ptr<const fem::Function<double>> function;
            std::vector<double> values; // one per cell
            /// Whether an expression reads the field. A field no expression
            /// reads is never evaluated.
            bool used = false;
        };
        std::vector<BoundField> fields_;

        // Cell centroids (3 per cell, cell-major) and the cell each of them
        // belongs to, filled on first use. The centroids are read on their
        // own cell, so no search for a containing cell is needed.
        std::vector<double> centroids_;
        std::vector<std::int32_t> centroid_cells_;
        bool centroids_ready_ = false;

        // The expression that applies to every cell, resolved from the domain
        // tags on first use — the update loop then indexes an array instead of
        // looking the domain of a cell up. A cell whose domain defines no
        // expression keeps a null entry.
        std::vector<Expression*> cell_expressions_;

        std::int32_t num_cells() const;
        double& cell_entry(std::int32_t cell);
        void cell_centroids();
        void resolve_cell_expressions();
    };

} // namespace hellofem::app
