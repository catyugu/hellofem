// hellofem::fem — vector assembly kernels (single-process)
// SPDX-License-Identifier: MIT

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "Form.h"
#include "kernel.h"
#include "mesh/Geometry.h"
#include "mesh/Mesh.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace hellofem::fem::impl {

    /// Execute the kernel over cells and accumulate into a vector array.
    template <std::floating_point T>
    void assemble_cells_vector(
        const std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P0,
        std::span<T> b, const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> cells, const DofMap& dofmap,
        const kernel_t<T>& kernel, std::span<const T> constants,
        const T* coeffs, int cstride,
        std::span<const std::uint32_t> cell_info0, std::span<T> be_b,
        std::span<T> cdofs_b)
    {
        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();

        const int bs = dofmap.bs();
        const std::size_t ndofs = dofmap.map().extent(1);
        const std::size_t ndim = static_cast<std::size_t>(bs) * ndofs;
        assert(be_b.size() >= ndim);
        assert(cdofs_b.size() >= 3 * ngeom);

        for (std::size_t index = 0; index < cells.size(); ++index) {
            const std::int32_t c = cells[index];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(c, i) + k];

            std::ranges::fill(be_b, 0);
            std::span<T> be = be_b.first(ndim);
            kernel(be.data(), coeffs ? &coeffs[index * cstride] : nullptr,
                constants.data(), cdofs_b.data(), nullptr, nullptr, nullptr);
            P0(be, cell_info0, c, 1);

            // Scatter the cell vector into the global vector.
            auto dofs = dofmap.cell_dofs(c);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (int k = 0; k < bs; ++k)
                    b[static_cast<std::size_t>(bs * dofs[i] + k)]
                        += be[bs * i + k];
        }
    }

    /// Execute the kernel over exterior-facet entities `(cell,
    /// local_facet)` and accumulate into a vector array.
    template <std::floating_point T>
    void assemble_entities_vector(
        const std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P0,
        std::span<T> b, const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> entities, const DofMap& dofmap,
        const kernel_t<T>& kernel, std::span<const T> constants,
        const T* coeffs, int cstride,
        std::span<const std::uint32_t> cell_info0, std::span<T> be_b,
        std::span<T> cdofs_b)
    {
        if (entities.empty())
            return;

        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();

        const int bs = dofmap.bs();
        const std::size_t ndofs = dofmap.map().extent(1);
        const std::size_t ndim = static_cast<std::size_t>(bs) * ndofs;
        assert(be_b.size() >= ndim);
        assert(cdofs_b.size() >= 3 * ngeom);

        const std::size_t num_entities = entities.size() / 2;
        for (std::size_t f = 0; f < num_entities; ++f) {
            const std::int32_t cell = entities[2 * f];
            const std::int32_t lf = entities[2 * f + 1];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(cell, i) + k];

            std::ranges::fill(be_b, 0);
            std::span<T> be = be_b.first(ndim);
            kernel(be.data(), coeffs ? &coeffs[f * cstride] : nullptr,
                constants.data(), cdofs_b.data(), &lf, nullptr, nullptr);
            P0(be, cell_info0, cell, 1);

            auto dofs = dofmap.cell_dofs(cell);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (int k = 0; k < bs; ++k)
                    b[static_cast<std::size_t>(bs * dofs[i] + k)]
                        += be[bs * i + k];
        }
    }

    /// Execute the kernel over interior-facet entities `(cell+, lf+,
    /// cell-, lf-)` and accumulate into a vector array.
    template <std::floating_point T>
    void assemble_interior_facets_vector(
        const std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P0,
        std::span<T> b, const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> facets, const DofMap& dofmap,
        const kernel_t<T>& kernel, std::span<const T> constants,
        const T* coeffs, int cstride,
        std::span<const std::uint32_t> cell_info0, std::span<T> be_b,
        std::span<T> cdofs_b)
    {
        if (facets.empty())
            return;

        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();

        const int bs = dofmap.bs();
        const std::size_t ndofs = dofmap.map().extent(1);
        const std::size_t ndim = static_cast<std::size_t>(bs) * ndofs;
        assert(be_b.size() >= 2 * ndim);
        assert(cdofs_b.size() >= 6 * ngeom);

        const std::size_t num_facets = facets.size() / 4;
        for (std::size_t f = 0; f < num_facets; ++f) {
            const std::int32_t cell0 = facets[4 * f];
            const std::int32_t lf0 = facets[4 * f + 1];
            const std::int32_t cell1 = facets[4 * f + 2];
            const std::int32_t lf1 = facets[4 * f + 3];

            // Both cells' geometry concatenated.
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(cell0, i) + k];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * (ngeom + i) + k] = x[3 * x_dofmap(cell1, i) + k];

            std::ranges::fill(be_b, 0);
            std::span<T> be = be_b.first(2 * ndim);
            std::array<int, 2> lf {lf0, lf1};
            kernel(be.data(), coeffs ? &coeffs[f * cstride] : nullptr,
                constants.data(), cdofs_b.data(), lf.data(), nullptr, nullptr);

            // Apply dof transformations on each half (skip a missing side).
            std::span<T> be0 = be.first(ndim);
            std::span<T> be1 = be.subspan(ndim, ndim);
            if (cell0 >= 0)
                P0(be0, cell_info0, cell0, 1);
            if (cell1 >= 0)
                P0(be1, cell_info0, cell1, 1);

            // Scatter both halves.
            if (cell0 >= 0) {
                auto dofs0 = dofmap.cell_dofs(cell0);
                for (std::size_t i = 0; i < dofs0.size(); ++i)
                    for (int k = 0; k < bs; ++k)
                        b[static_cast<std::size_t>(bs * dofs0[i] + k)]
                            += be0[bs * i + k];
            }
            if (cell1 >= 0) {
                auto dofs1 = dofmap.cell_dofs(cell1);
                for (std::size_t i = 0; i < dofs1.size(); ++i)
                    for (int k = 0; k < bs; ++k)
                        b[static_cast<std::size_t>(bs * dofs1[i] + k)]
                            += be1[bs * i + k];
            }
        }
    }

} // namespace hellofem::fem::impl
