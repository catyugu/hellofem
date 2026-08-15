// hellofem::fem — evaluation of expressions over mesh entities
// SPDX-License-Identifier: MIT

#pragma once

#include "Constant.h"
#include "Expression.h"
#include "Function.h"
#include "mesh/Geometry.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace hellofem::fem {

    /// Evaluate an expression over a list of cells.
    ///
    /// @param[in] expr The expression.
    /// @param[in] mesh The mesh.
    /// @param[in] cells Cell indices to evaluate over.
    /// @return Values, row-major with shape `(num_cells, num_points,
    /// value_size)`.
    template <std::floating_point T>
    std::vector<T> tabulate_expression(const Expression<T>& expr,
        const mesh::Mesh<T>& mesh, std::span<const std::int32_t> cells)
    {
        if (cells.empty())
            return {};

        const auto x_dofmap = mesh.geometry().dofmaps().front();
        std::span<const T> x = mesh.geometry().x();
        const std::size_t ngeom = x_dofmap.extent(1);

        // Pack the coefficients: one block per cell, stride = total
        // coefficient space dimension.
        const std::size_t num_cells = cells.size();
        std::vector<std::int32_t> offsets {0};
        for (const auto& c : expr.coefficients())
            offsets.push_back(offsets.back()
                + c->function_space()->element()->space_dimension());
        const std::size_t cstride = offsets.back();

        std::vector<T> constants;
        for (const auto& c : expr.constants())
            constants.insert(constants.end(), c->value.begin(), c->value.end());

        std::vector<T> coeffs(num_cells * cstride, 0);
        for (std::size_t e = 0; e < num_cells; ++e) {
            for (std::size_t c = 0; c < expr.coefficients().size(); ++c) {
                const auto& u = *expr.coefficients()[c];
                const DofMap& dofmap = *u.function_space()->dofmap();
                std::span<const T> ua = u.x()->array();
                const int bs = dofmap.bs();
                const std::size_t offset = offsets[c];
                auto dofs = dofmap.cell_dofs(cells[e]);
                for (std::size_t i = 0; i < dofs.size(); ++i)
                    for (int k = 0; k < bs; ++k)
                        coeffs[e * cstride + offset + bs * i + k]
                            = ua[static_cast<std::size_t>(bs * dofs[i]) + k];
            }
        }

        const std::size_t npoints = expr.points().second[0];
        const std::size_t value_size = expr.value_size();
        const std::size_t per_entity = npoints * value_size;
        std::vector<T> values(num_cells * per_entity, 0);

        std::vector<T> values_local(per_entity);
        std::vector<T> cdofs_b(3 * ngeom);
        for (std::size_t e = 0; e < num_cells; ++e) {
            const std::int32_t c = cells[e];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(c, i) + k];

            std::ranges::fill(values_local, 0);
            expr.kernel()(values_local.data(),
                cstride ? coeffs.data() + e * cstride : nullptr,
                constants.empty() ? nullptr : constants.data(),
                cdofs_b.data(), nullptr, nullptr, nullptr);

            std::copy(values_local.begin(), values_local.end(),
                std::next(values.begin(), e * per_entity));
        }

        return values;
    }

    /// Evaluate an expression over (cell, local_facet) pairs, e.g. on
    /// the mesh boundary.
    ///
    /// @param[in] expr The expression.
    /// @param[in] mesh The mesh.
    /// @param[in] entities Flattened `(cell, local_facet)` pairs.
    /// @return Values, row-major with shape `(num_entities, num_points,
    /// value_size)`.
    template <std::floating_point T>
    std::vector<T> tabulate_expression_facets(const Expression<T>& expr,
        const mesh::Mesh<T>& mesh, std::span<const std::int32_t> entities)
    {
        if (entities.empty())
            return {};
        if (entities.size() % 2 != 0)
            throw std::runtime_error(
                "tabulate_expression_facets: entities must be (cell, facet) "
                "pairs.");

        const auto x_dofmap = mesh.geometry().dofmaps().front();
        std::span<const T> x = mesh.geometry().x();
        const std::size_t ngeom = x_dofmap.extent(1);
        const std::size_t num_entities = entities.size() / 2;

        // Pack coefficients (one block per (cell, facet) entity).
        std::vector<std::int32_t> offsets {0};
        for (const auto& c : expr.coefficients())
            offsets.push_back(offsets.back()
                + c->function_space()->element()->space_dimension());
        const std::size_t cstride = offsets.back();

        std::vector<T> constants;
        for (const auto& c : expr.constants())
            constants.insert(constants.end(), c->value.begin(), c->value.end());

        std::vector<T> coeffs(num_entities * cstride, 0);
        for (std::size_t e = 0; e < num_entities; ++e) {
            const std::int32_t c = entities[2 * e];
            for (std::size_t i = 0; i < expr.coefficients().size(); ++i) {
                const auto& u = *expr.coefficients()[i];
                const DofMap& dofmap = *u.function_space()->dofmap();
                std::span<const T> ua = u.x()->array();
                const int bs = dofmap.bs();
                const std::size_t offset = offsets[i];
                auto dofs = dofmap.cell_dofs(c);
                for (std::size_t j = 0; j < dofs.size(); ++j)
                    for (int k = 0; k < bs; ++k)
                        coeffs[e * cstride + offset + bs * j + k]
                            = ua[static_cast<std::size_t>(bs * dofs[j]) + k];
            }
        }

        const std::size_t npoints = expr.points().second[0];
        const std::size_t value_size = expr.value_size();
        const std::size_t per_entity = npoints * value_size;
        std::vector<T> values(num_entities * per_entity, 0);

        // Facet permutations are only needed for non-Lagrange elements;
        // request them if the mesh topology can provide them.
        std::vector<T> values_local(per_entity);
        std::vector<T> cdofs_b(3 * ngeom);
        for (std::size_t e = 0; e < num_entities; ++e) {
            const std::int32_t c = entities[2 * e];
            const std::int32_t lf = entities[2 * e + 1];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(c, i) + k];

            std::ranges::fill(values_local, 0);
            expr.kernel()(values_local.data(),
                cstride ? coeffs.data() + e * cstride : nullptr,
                constants.empty() ? nullptr : constants.data(),
                cdofs_b.data(), &lf, nullptr, nullptr);

            std::copy(values_local.begin(), values_local.end(),
                std::next(values.begin(), e * per_entity));
        }

        return values;
    }

} // namespace hellofem::fem
