// hellofem::fem — Dirichlet boundary conditions
// SPDX-License-Identifier: MIT

#pragma once

#include "Constant.h"
#include "DofMap.h"
#include "FiniteElement.h"
#include "Function.h"
#include "FunctionSpace.h"
#include "mesh/Topology.h"

#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace hellofem::fem {

    /// Dirichlet boundary conditions: a value (constant or function) and
    /// the set of dofs it applies to.
    template <std::floating_point T>
    class DirichletBC {
    public:
        /// Value type.
        using value_type = T;

        /// Locate dofs sitting on the closure of a set of topological
        /// entities of dimension `dim`.
        /// @param[in] topology The mesh topology (entities must exist).
        /// @param[in] dofmap The dofmap.
        /// @param[in] dim Dimension of the entities to apply on.
        /// @param[in] entities Indices of the entities.
        /// @return The scalar (physical) dof indices.
        static std::vector<std::int32_t> locate_dofs_topological(
            const mesh::Topology& topology, const DofMap& dofmap, int dim,
            std::span<const std::int32_t> entities);

        /// Locate dofs whose dof coordinates satisfy a marker function.
        /// @param[in] V The function space.
        /// @param[in] marker_fn Returns true for dofs to apply on.
        /// @return The scalar (physical) dof indices.
        static std::vector<std::int32_t>
        locate_dofs_geometrical(const FunctionSpace<T>& V,
            std::function<bool(std::span<const T>)> marker_fn);

        /// Create a DirichletBC from a scalar value and a list of dofs.
        /// @param[in] g The scalar value.
        /// @param[in] dofs Scalar dof indices.
        /// @param[in] V The function space.
        DirichletBC(T g, std::span<const std::int32_t> dofs,
            std::shared_ptr<const FunctionSpace<T>> V)
            : _function_space(std::move(V)), _value(std::make_shared<const Constant<T>>(g)), _dofs(dofs.begin(), dofs.end())
        {
        }

        /// Create a DirichletBC from a constant value and a list of dofs.
        DirichletBC(std::shared_ptr<const Constant<T>> g,
            std::span<const std::int32_t> dofs,
            std::shared_ptr<const FunctionSpace<T>> V)
            : _function_space(std::move(V)), _value(std::move(g)), _dofs(dofs.begin(), dofs.end())
        {
        }

        /// The function space the BC applies to.
        std::shared_ptr<const FunctionSpace<T>> function_space() const
        {
            return _function_space;
        }

        /// The boundary value (a function or a constant).
        const std::variant<std::shared_ptr<const Function<T>>,
            std::shared_ptr<const Constant<T>>>&
        value() const
        {
            return _value;
        }

        /// The scalar dof indices the BC applies to.
        std::span<const std::int32_t> dof_indices() const { return _dofs; }

        /// Apply the boundary condition: `x[dof] = alpha * (g[dof] - x0[dof])`
        /// for each boundary dof.
        /// @param[in,out] x The vector to modify.
        /// @param[in] x0 Previous value (subtracted if present).
        /// @param[in] alpha Scaling factor.
        void set(std::span<T> x, std::optional<std::span<const T>> x0,
            T alpha) const
        {
            if (const auto* g = std::get_if<std::shared_ptr<const Constant<T>>>(&_value)) {
                const T gv = (*g)->value[0];
                for (std::int32_t d : _dofs)
                    x[static_cast<std::size_t>(d)]
                        = alpha * (gv - (x0 ? (*x0)[static_cast<std::size_t>(d)] : T {0}));
            }
            else {
                const auto& gf = *std::get<std::shared_ptr<const Function<T>>>(_value);
                for (std::int32_t d : _dofs)
                    x[static_cast<std::size_t>(d)]
                        = alpha * (gf.x()->array()[static_cast<std::size_t>(d)] - (x0 ? (*x0)[static_cast<std::size_t>(d)] : T {0}));
            }
        }

        /// Set `markers[dof] = true` for each boundary dof.
        /// @param[in,out] markers The marker array.
        void mark_dofs(std::span<std::int8_t> markers) const
        {
            for (std::int32_t d : _dofs)
                markers[static_cast<std::size_t>(d)] = true;
        }

    private:
        // Function space.
        std::shared_ptr<const FunctionSpace<T>> _function_space;

        // Boundary value.
        std::variant<std::shared_ptr<const Function<T>>,
            std::shared_ptr<const Constant<T>>>
            _value;

        // Scalar dof indices the BC applies to.
        std::vector<std::int32_t> _dofs;
    };

} // namespace hellofem::fem
