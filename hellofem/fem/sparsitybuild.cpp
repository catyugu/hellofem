// hellofem::fem — build sparsity patterns from dofmaps
// SPDX-License-Identifier: MIT

#include "sparsitybuild.h"

#include "DofMap.h"
#include "la/SparsityPattern.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <span>

namespace hellofem::fem {

    void sparsitybuild::interior_facets(la::SparsityPattern& pattern,
        std::array<std::span<const std::int32_t>, 2> cells,
        std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
    {
        std::span<const std::int32_t> cells0 = cells[0];
        std::span<const std::int32_t> cells1 = cells[1];
        assert(cells0.size() == cells1.size());
        const DofMap& dofmap0 = dofmaps[0];
        const DofMap& dofmap1 = dofmaps[1];

        // Iterate over facets (pairs of cells). A missing side (negative
        // cell index) contributes nothing.
        for (std::size_t f = 0; f < cells0.size(); f += 2) {
            std::span<const std::int32_t> dofs00
                = cells0[f] >= 0 ? dofmap0.cell_dofs(cells0[f])
                                 : std::span<const std::int32_t>();
            std::span<const std::int32_t> dofs01
                = cells0[f + 1] >= 0 ? dofmap0.cell_dofs(cells0[f + 1])
                                     : std::span<const std::int32_t>();

            std::span<const std::int32_t> dofs10
                = cells1[f] >= 0 ? dofmap1.cell_dofs(cells1[f])
                                 : std::span<const std::int32_t>();
            std::span<const std::int32_t> dofs11
                = cells1[f + 1] >= 0 ? dofmap1.cell_dofs(cells1[f + 1])
                                     : std::span<const std::int32_t>();

            pattern.insert(dofs00, dofs10);
            pattern.insert(dofs00, dofs11);
            pattern.insert(dofs01, dofs10);
            pattern.insert(dofs01, dofs11);
        }
    }

} // namespace hellofem::fem
