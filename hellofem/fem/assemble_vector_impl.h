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
        std::function<void(std::span<T>, std::span<const std::uint32_t>,
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

} // namespace hellofem::fem::impl
