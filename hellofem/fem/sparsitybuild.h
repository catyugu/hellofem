// hellofem::fem — build sparsity patterns from dofmaps
// SPDX-License-Identifier: MIT

#pragma once

#include "DofMap.h"
#include "la/SparsityPattern.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <ranges>
#include <span>
#include <utility>

namespace hellofem::fem {

    namespace sparsitybuild {

        /// Insert the non-zero pattern for a set of cells into a sparsity
        /// pattern, given a dofmap for each of the two (test, trial) spaces.
        template <std::ranges::input_range R0, std::ranges::input_range R1>
        void cells(la::SparsityPattern& pattern,
            const std::pair<R0, R1>& cells,
            std::array<std::reference_wrapper<const DofMap>, 2> dofmaps)
        {
            assert(cells.first.size() == cells.second.size());
            auto it0 = cells.first.begin();
            auto it1 = cells.second.begin();
            for (; it0 != cells.first.end(); ++it0, ++it1) {
                pattern.insert(dofmaps[0].get().cell_dofs(*it0),
                    dofmaps[1].get().cell_dofs(*it1));
            }
        }

        /// Insert the non-zero pattern for interior facets (pairs of cells
        /// sharing a facet) into a sparsity pattern.
        ///
        /// `cells[0]` and `cells[1]` hold the cells on each side of each
        /// facet, stored as interleaved pairs (cell0_0, cell1_0, cell0_1,
        /// cell1_1, ...). A negative cell index marks a missing (exterior)
        /// side, whose contribution is skipped.
        void interior_facets(la::SparsityPattern& pattern,
            std::array<std::span<const std::int32_t>, 2> cells,
            std::array<std::reference_wrapper<const DofMap>, 2> dofmaps);

    } // namespace sparsitybuild

} // namespace hellofem::fem
