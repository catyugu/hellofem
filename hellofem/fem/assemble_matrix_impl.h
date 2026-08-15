// hellofem::fem — matrix assembly kernels (single-process)
// SPDX-License-Identifier: MIT

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "Form.h"
#include "kernel.h"
#include "la/utils.h"
#include "mesh/Geometry.h"
#include "mesh/Mesh.h"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace hellofem::fem::impl {

    /// True if any of the dofs of a cell are marked by `bc`. An empty
    /// `bc` (no boundary conditions) marks nothing.
    inline bool has_bc(std::span<const std::int32_t> dofs, int bs,
        std::span<const std::int8_t> bc)
    {
        if (bc.empty())
            return false;
        for (auto dof : dofs)
            for (int k = 0; k < bs; ++k)
                if (bc[static_cast<std::size_t>(bs * dof + k)])
                    return true;
        return false;
    }

    /// Zero the bs1 columns of column block `j` of the element tensor.
    inline void zero_col_block(std::span<double> Ae, std::size_t ndim0,
        std::size_t ndim1, std::size_t j, int bs1)
    {
        for (std::size_t i = 0; i < ndim0; ++i)
            for (int k = 0; k < bs1; ++k)
                Ae[i * ndim1 + j * bs1 + k] = 0;
    }

    /// Execute the kernel over cells and accumulate into a matrix.
    ///
    /// Each cell: gather geometry, zero the element tensor, run the
    /// kernel, apply dof transformations, zero BC rows/cols (unless
    /// lifting) and scatter through `mat_set`.
    ///
    /// @tparam LiftingMode When true, skip cells with no BC column dofs
    /// and do not zero BC rows/cols (used by apply_lifting).
    template <bool LiftingMode, std::floating_point T>
    void assemble_cells_matrix(
        la::MatSet<T> auto mat_set, const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> cells, const DofMap& dofmap0,
        std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P0,
        const DofMap& dofmap1,
        std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P1T,
        std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1,
        const kernel_t<T>& kernel, const T* coeffs, int cstride,
        std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
        std::span<const std::uint32_t> cell_info1, std::span<T> Ab,
        std::span<T> cdofs_b)
    {
        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();

        const int bs0 = dofmap0.bs();
        const int bs1 = dofmap1.bs();
        const std::size_t ndofs0 = dofmap0.map().extent(1);
        const std::size_t ndofs1 = dofmap1.map().extent(1);
        const std::size_t ndim0 = static_cast<std::size_t>(bs0) * ndofs0;
        const std::size_t ndim1 = static_cast<std::size_t>(bs1) * ndofs1;
        assert(Ab.size() >= ndim0 * ndim1);
        assert(cdofs_b.size() >= 3 * ngeom);

        for (std::size_t index = 0; index < cells.size(); ++index) {
            const std::int32_t c = cells[index];
            auto dofs0 = dofmap0.cell_dofs(c);
            auto dofs1 = dofmap1.cell_dofs(c);

            if (LiftingMode and not has_bc(dofs1, bs1, bc1))
                continue;

            // Gather the cell geometry into cdofs_b.
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(c, i) + k];

            // Zero the element tensor and run the kernel.
            std::ranges::fill(Ab, 0);
            std::span<T> Ae = Ab.first(ndim0 * ndim1);
            kernel(Ae.data(), coeffs ? &coeffs[index * cstride] : nullptr,
                constants.data(), cdofs_b.data(), nullptr, nullptr, nullptr);

            // Apply dof transformations: Ae = P0 * Ae * P1^T.
            P0(Ae, cell_info0, c, 1);
            P1T(Ae, cell_info1, c, 1);

            if (not LiftingMode) {
                // Zero rows marked by bc0 and columns marked by bc1.
                for (std::size_t i = 0; i < ndofs0; ++i)
                    if (has_bc(dofs0.subspan(i, 1), bs0, bc0))
                        for (std::size_t j = 0; j < ndim1; ++j)
                            Ae[i * ndim1 + j] = 0;
                for (std::size_t j = 0; j < ndofs1; ++j)
                    if (has_bc(dofs1.subspan(j, 1), bs1, bc1))
                        zero_col_block(Ae, ndim0, ndim1, j, bs1);
            }

            // Scatter the element tensor into the matrix.
            mat_set(dofs0, dofs1, Ae);
        }
    }

    /// Execute the kernel over exterior-facet entities `(cell,
    /// local_facet)` and accumulate into a matrix.
    template <bool LiftingMode, std::floating_point T>
    void assemble_entities_matrix(
        la::MatSet<T> auto mat_set, const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> entities, const DofMap& dofmap0,
        std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P0,
        const DofMap& dofmap1,
        std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P1T,
        std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1,
        const kernel_t<T>& kernel, const T* coeffs, int cstride,
        std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
        std::span<const std::uint32_t> cell_info1, std::span<T> Ab,
        std::span<T> cdofs_b)
    {
        if (entities.empty())
            return;

        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();

        const int bs0 = dofmap0.bs();
        const int bs1 = dofmap1.bs();
        const std::size_t ndofs0 = dofmap0.map().extent(1);
        const std::size_t ndofs1 = dofmap1.map().extent(1);
        const std::size_t ndim0 = static_cast<std::size_t>(bs0) * ndofs0;
        const std::size_t ndim1 = static_cast<std::size_t>(bs1) * ndofs1;
        assert(Ab.size() >= ndim0 * ndim1);
        assert(cdofs_b.size() >= 3 * ngeom);

        const std::size_t num_entities = entities.size() / 2;
        for (std::size_t f = 0; f < num_entities; ++f) {
            const std::int32_t cell = entities[2 * f];
            const std::int32_t lf = entities[2 * f + 1];
            auto dofs0 = dofmap0.cell_dofs(cell);
            auto dofs1 = dofmap1.cell_dofs(cell);

            if (LiftingMode and not has_bc(dofs1, bs1, bc1))
                continue;

            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(cell, i) + k];

            std::ranges::fill(Ab, 0);
            std::span<T> Ae = Ab.first(ndim0 * ndim1);
            kernel(Ae.data(), coeffs ? &coeffs[f * cstride] : nullptr,
                constants.data(), cdofs_b.data(), &lf, nullptr, nullptr);
            P0(Ae, cell_info0, cell, 1);
            P1T(Ae, cell_info1, cell, 1);

            if (not LiftingMode) {
                for (std::size_t i = 0; i < ndofs0; ++i)
                    if (has_bc(dofs0.subspan(i, 1), bs0, bc0))
                        for (std::size_t j = 0; j < ndim1; ++j)
                            Ae[i * ndim1 + j] = 0;
                for (std::size_t j = 0; j < ndofs1; ++j)
                    if (has_bc(dofs1.subspan(j, 1), bs1, bc1))
                        zero_col_block(Ae, ndim0, ndim1, j, bs1);
            }

            mat_set(dofs0, dofs1, Ae);
        }
    }

    /// Execute the kernel over interior-facet entities `(cell+, lf+,
    /// cell-, lf-)` and accumulate into a matrix. The element tensor is
    /// `(2*ndofs0, 2*ndofs1)` with blocks in row-major side0-side0,
    /// side0-side1, side1-side0, side1-side1 order.
    template <bool LiftingMode, std::floating_point T>
    void assemble_interior_facets_matrix(
        la::MatSet<T> auto mat_set, const mesh::Geometry<T>& geometry,
        std::span<const std::int32_t> facets, const DofMap& dofmap0,
        std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P0,
        const DofMap& dofmap1,
        std::function<void(std::span<T>, std::span<const std::uint32_t>,
            std::int32_t, int)>& P1T,
        std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1,
        const kernel_t<T>& kernel, const T* coeffs, int cstride,
        std::span<const T> constants, std::span<const std::uint32_t> cell_info0,
        std::span<const std::uint32_t> cell_info1, std::span<T> Ab,
        std::span<T> cdofs_b)
    {
        if (facets.empty())
            return;

        const auto x_dofmap = geometry.dofmaps().front();
        const std::size_t ngeom = x_dofmap.extent(1);
        std::span<const T> x = geometry.x();

        const int bs0 = dofmap0.bs();
        const int bs1 = dofmap1.bs();
        const std::size_t ndofs0 = dofmap0.map().extent(1);
        const std::size_t ndofs1 = dofmap1.map().extent(1);
        const std::size_t ndim0 = static_cast<std::size_t>(bs0) * ndofs0;
        const std::size_t ndim1 = static_cast<std::size_t>(bs1) * ndofs1;
        const std::size_t ndim0_2 = 2 * ndim0;
        const std::size_t ndim1_2 = 2 * ndim1;
        assert(Ab.size() >= ndim0_2 * ndim1_2);
        assert(cdofs_b.size() >= 6 * ngeom);

        const std::size_t num_facets = facets.size() / 4;
        for (std::size_t f = 0; f < num_facets; ++f) {
            const std::int32_t cell0 = facets[4 * f];
            const std::int32_t lf0 = facets[4 * f + 1];
            const std::int32_t cell1 = facets[4 * f + 2];
            const std::int32_t lf1 = facets[4 * f + 3];
            const std::array<std::int32_t, 2> cells {cell0, cell1};

            auto dofs0 = dofmap0.cell_dofs(cell0);
            auto dofs1 = dofmap1.cell_dofs(cell1);

            if (LiftingMode and not has_bc(dofs1, bs1, bc1))
                continue;

            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * i + k] = x[3 * x_dofmap(cell0, i) + k];
            for (std::size_t i = 0; i < ngeom; ++i)
                for (int k = 0; k < 3; ++k)
                    cdofs_b[3 * (ngeom + i) + k] = x[3 * x_dofmap(cell1, i) + k];

            std::ranges::fill(Ab, 0);
            std::span<T> Ae = Ab.first(ndim0_2 * ndim1_2);
            std::array<int, 2> lf {lf0, lf1};
            kernel(Ae.data(), coeffs ? &coeffs[f * cstride] : nullptr,
                constants.data(), cdofs_b.data(), lf.data(), nullptr, nullptr);

            // Apply dof transformations. The element tensor is a 2x2 block
            // matrix with blocks (cell0cell0, cell0cell1, cell1cell0,
            // cell1cell1), each of size (ndim0, ndim1). Cell0's test dofs
            // are the top ndim0 rows; cell1's the bottom ndim0 rows. The
            // trial dofs of cell0 are the left ndim1 columns; cell1's the
            // right ndim1 columns.
            if (cell0 >= 0)
                P0(Ae.first(ndim0 * ndim1_2), cell_info0, cell0, 1);
            if (cell1 >= 0)
                P0(Ae.subspan(ndim0 * ndim1_2, ndim0 * ndim1_2), cell_info0, cell1, 1);
            // P1T acts on the trial dofs: cell0's columns [0, ndim1) across
            // all rows, cell1's columns [ndim1, 2*ndim1) per row (not
            // contiguous in the joint block).
            if (cell0 >= 0)
                P1T(Ae, cell_info1, cell0, 1);
            if (cell1 >= 0) {
                for (std::size_t row = 0; row < ndim0_2; ++row)
                    P1T(Ae.subspan(row * ndim1_2 + ndim1, ndim1), cell_info1,
                        cell1, 1);
            }

            if (not LiftingMode) {
                // Zero BC rows/cols across both sides.
                for (std::size_t i = 0; i < 2 * ndofs0; ++i) {
                    const std::int32_t c = cells[i / ndofs0];
                    const std::size_t i_local = i % ndofs0;
                    auto cdofs = dofmap0.cell_dofs(c);
                    if (has_bc(cdofs.subspan(i_local, 1), bs0, bc0))
                        for (std::size_t j = 0; j < ndim1_2; ++j)
                            Ae[i * ndim1_2 + j] = 0;
                }
                for (std::size_t j = 0; j < 2 * ndofs1; ++j) {
                    const std::int32_t c = cells[j / ndofs1];
                    const std::size_t j_local = j % ndofs1;
                    auto cdofs = dofmap1.cell_dofs(c);
                    if (has_bc(cdofs.subspan(j_local, 1), bs1, bc1))
                        for (std::size_t i = 0; i < ndim0_2; ++i)
                            for (int k = 0; k < bs1; ++k)
                                Ae[i * ndim1_2 + j * bs1 + k] = 0;
                }
            }

            // Scatter the four blocks.
            std::array<std::span<const std::int32_t>, 2> ds {dofs0, dofs1};
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    // Build the block [a,b] explicitly.
                    std::vector<T> A_block(ndim0 * ndim1);
                    for (std::size_t i = 0; i < ndim0; ++i)
                        for (std::size_t j = 0; j < ndim1; ++j)
                            A_block[i * ndim1 + j]
                                = Ae[(a * ndim0 + i) * ndim1_2 + (b * ndim1 + j)];
                    mat_set(ds[a], ds[b], A_block);
                }
            }
        }
    }

} // namespace hellofem::fem::impl
