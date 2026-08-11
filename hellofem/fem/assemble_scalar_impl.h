// hellofem::fem — scalar assembly kernels (single-process)
// SPDX-License-Identifier: MIT

#pragma once

#include "Form.h"
#include "kernel.h"
#include "mesh/Geometry.h"
#include "mesh/Mesh.h"

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

} // namespace hellofem::fem::impl
