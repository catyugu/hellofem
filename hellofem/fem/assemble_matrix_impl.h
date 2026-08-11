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

} // namespace hellofem::fem::impl
