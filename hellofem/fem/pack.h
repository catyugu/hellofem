// hellofem::fem — packing of coefficients and constants for kernels
// SPDX-License-Identifier: MIT

#pragma once

#include "Form.h"
#include "kernel.h"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace hellofem::fem {

    /// Concatenate all constant values of a form in order.
    template <std::floating_point T>
    std::vector<T> pack_constants(const Form<T>& form)
    {
        std::vector<T> constants;
        for (const auto& c : form.constants())
            constants.insert(constants.end(), c->value.begin(), c->value.end());
        return constants;
    }

    /// Allocate per-integral coefficient storage.
    ///
    /// @return For each `(type, idx)`, the coefficient buffer and the
    /// number of coefficients packed per entity (`cstride`). For
    /// `interior_facet` integrals `cstride` is `2 * total` (two sides per
    /// entity) and the buffer holds `num_entities * cstride` entries.
    template <std::floating_point T>
    std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>>
    allocate_coefficient_storage(const Form<T>& form)
    {
        std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>>
            storage;

        // Total coefficient size (sum of space dimensions) determines the
        // stride; each integral packs only its active coefficients.
        std::vector<int> offsets = form.coefficient_offsets();
        const int total = offsets.back();

        for (IntegralType type : form.integral_types()) {
            const int num = form.num_integrals(type);
            for (int idx = 0; idx < num; ++idx) {
                const std::size_t num_entities
                    = form.domain(type, idx, 0).size();
                const int cells_per_entity
                    = (type == IntegralType::interior_facet) ? 2 : 1;
                const std::size_t num_entity_blocks
                    = num_entities / cells_per_entity;
                const int stride = cells_per_entity * total;
                // Always create an entry so assemble() can look it up;
                // a form without coefficients stores an empty buffer and
                // the kernel receives a null pointer.
                std::vector<T> data(static_cast<std::size_t>(stride)
                    * num_entity_blocks);
                storage[{type, idx}] = {std::move(data), stride};
            }
        }

        return storage;
    }

    /// Pack the active coefficients of every integral of a form into the
    /// preallocated storage.
    ///
    /// The flattened entity encoding differs by integral type:
    /// - `cell`: one cell index per entity (`entities[e]`);
    /// - `exterior_facet`: `(cell, local_facet)` pairs, cell at `entities[2e]`;
    /// - `interior_facet`: `(cell+, lf+, cell-, lf-)` quadruples, cells at
    ///   `entities[4e]` and `entities[4e+2]`.
    ///
    /// Each cell contributes `total` coefficient entries (stride `2*total`
    /// per interior entity). Negative cell indices (missing sides) are
    /// skipped.
    ///
    /// @param[in] form The form.
    /// @param[in,out] storage Per-integral coefficient storage from
    /// @ref allocate_coefficient_storage.
    template <std::floating_point T>
    void pack_coefficients(const Form<T>& form,
        std::map<std::pair<IntegralType, int>,
            std::pair<std::vector<T>, int>>& storage)
    {
        const std::vector<int> offsets = form.coefficient_offsets();
        const int total = offsets.back();
        for (IntegralType type : form.integral_types()) {
            const int num = form.num_integrals(type);
            for (int idx = 0; idx < num; ++idx) {
                auto& [coeffs, cstride] = storage.at({type, idx});
                std::span<const std::int32_t> entities
                    = form.domain(type, idx, 0);

                // Cells per entity and flattened entries per entity. Cell
                // integrals store one cell per entity; exterior facets
                // store (cell, local_facet) pairs; interior facets store
                // (cell+, lf+, cell-, lf-) quadruples.
                const int cells_per_entity
                    = (type == IntegralType::interior_facet) ? 2 : 1;
                const int entries_per_entity
                    = (type == IntegralType::cell) ? 1
                    : (type == IntegralType::exterior_facet) ? 2
                                                             : 4;
                const std::size_t num_entities
                    = entities.size() / entries_per_entity;
                const int stride = cells_per_entity * total;

                for (int c : form.active_coeffs(type, idx)) {
                    const auto& u = *form.coefficients()[static_cast<std::size_t>(c)];
                    const DofMap& dofmap = *u.function_space()->dofmap();
                    std::span<const T> ua = u.x()->array();
                    const int bs = dofmap.bs();
                    const int offset = offsets[static_cast<std::size_t>(c)];

                    for (std::size_t e = 0; e < num_entities; ++e) {
                        for (int s = 0; s < cells_per_entity; ++s) {
                            // Flat position of the cell in `entities`:
                            // cell integrals carry no local index.
                            const std::size_t cell_pos
                                = (type == IntegralType::cell)
                                ? e
                                : e * entries_per_entity + 2 * static_cast<std::size_t>(s);
                            const std::int32_t cell = entities[cell_pos];
                            if (cell < 0)
                                continue;
                            auto dofs = dofmap.cell_dofs(cell);
                            const std::size_t pos_c
                                = e * static_cast<std::size_t>(stride)
                                + static_cast<std::size_t>(s * total + offset);
                            for (std::size_t i = 0; i < dofs.size(); ++i) {
                                const std::size_t pos_v
                                    = static_cast<std::size_t>(bs * dofs[i]);
                                for (int k = 0; k < bs; ++k)
                                    coeffs[pos_c + bs * i + k] = ua[pos_v + k];
                            }
                        }
                    }
                }
            }
        }
    }

} // namespace hellofem::fem
