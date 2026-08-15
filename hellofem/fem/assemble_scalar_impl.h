// hellofem::fem — scalar assembly kernels (single-process)
// SPDX-License-Identifier: MIT

#pragma once

#include "Form.h"
#include "kernel.h"
#include "mesh/Geometry.h"
#include "mesh/Mesh.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace hellofem::fem::impl {

    /// Execute the kernel over cells and accumulate a scalar.
    template <std::floating_point T>
    T assemble_cells_scalar(const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> cells, const kernel_t<T>& kernel,
        std::span<const T> constants, const T* coeffs, int cstride,
        std::span<T> cdofs_b)
    {
        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();
        assert(cdofs_b.size() >= 3 * ngeom);

        T value {0};
        for (std::size_t index = 0; index < cells.size(); ++index) {
            const std::int32_t c = cells[index];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(c, i) + k];

            kernel(&value, coeffs ? &coeffs[index * cstride] : nullptr,
                constants.data(), cdofs_b.data(), nullptr, nullptr, nullptr);
        }
        return value;
    }

    /// Execute the kernel over exterior-facet entities `(cell,
    /// local_facet)` and accumulate a scalar.
    template <std::floating_point T>
    T assemble_entities_scalar(const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> entities, const kernel_t<T>& kernel,
        std::span<const T> constants, const T* coeffs, int cstride,
        std::span<T> cdofs_b)
    {
        if (entities.empty())
            return 0;

        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();
        assert(cdofs_b.size() >= 3 * ngeom);

        T value {0};
        const std::size_t num_entities = entities.size() / 2;
        for (std::size_t f = 0; f < num_entities; ++f) {
            const std::int32_t cell = entities[2 * f];
            const std::int32_t lf = entities[2 * f + 1];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(cell, i) + k];

            kernel(&value, coeffs ? &coeffs[f * cstride] : nullptr,
                constants.data(), cdofs_b.data(), &lf, nullptr, nullptr);
        }
        return value;
    }

    /// Execute the kernel over interior-facet entities `(cell+, lf+,
    /// cell-, lf-)` and accumulate a scalar.
    template <std::floating_point T>
    T assemble_interior_facets_scalar(const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> facets, const kernel_t<T>& kernel,
        std::span<const T> constants, const T* coeffs, int cstride,
        std::span<T> cdofs_b)
    {
        if (facets.empty())
            return 0;

        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();
        assert(cdofs_b.size() >= 6 * ngeom);

        T value {0};
        const std::size_t num_facets = facets.size() / 4;
        for (std::size_t f = 0; f < num_facets; ++f) {
            const std::int32_t cell0 = facets[4 * f];
            const std::int32_t lf0 = facets[4 * f + 1];
            const std::int32_t cell1 = facets[4 * f + 2];
            const std::int32_t lf1 = facets[4 * f + 3];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(cell0, i) + k];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * (ngeom + i) + k] = x[3 * x_dofmap(cell1, i) + k];

            std::array<int, 2> lf {lf0, lf1};
            kernel(&value, coeffs ? &coeffs[f * cstride] : nullptr,
                constants.data(), cdofs_b.data(), lf.data(), nullptr, nullptr);
        }
        return value;
    }

} // namespace hellofem::fem::impl
