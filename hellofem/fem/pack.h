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
    /// number of coefficients packed per entity (`cstride`).
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
                // Always create an entry so assemble() can look it up;
                // a form without coefficients stores an empty buffer and
                // the kernel receives a null pointer.
                std::vector<T> data(static_cast<std::size_t>(total)
                    * num_entities);
                storage[{type, idx}] = {std::move(data), total};
            }
        }

        return storage;
    }

    /// Pack the active coefficients of every integral of a form into the
    /// preallocated storage. Each entry `coeffs[e*cstride + offset + i]`
    /// holds the `i`-th dof value of the coefficient on entity `e`, for
    /// the coefficient's offset into the stride.
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
        for (IntegralType type : form.integral_types()) {
            const int num = form.num_integrals(type);
            for (int idx = 0; idx < num; ++idx) {
                auto& [coeffs, cstride] = storage.at({type, idx});
                std::span<const std::int32_t> entities
                    = form.domain(type, idx, 0);

                for (int c : form.active_coeffs(type, idx)) {
                    const auto& u = *form.coefficients()[static_cast<std::size_t>(c)];
                    const DofMap& dofmap = *u.function_space()->dofmap();
                    std::span<const T> ua = u.x()->array();
                    const int space_dim
                        = u.function_space()->element()->space_dimension();
                    const int bs = dofmap.bs();
                    const int offset = offsets[static_cast<std::size_t>(c)];

                    for (std::size_t e = 0; e < entities.size(); ++e) {
                        std::int32_t cell = entities[e];
                        if (cell < 0)
                            continue;
                        auto dofs = dofmap.cell_dofs(cell);
                        for (std::size_t i = 0; i < dofs.size(); ++i) {
                            const std::size_t pos_c
                                = static_cast<std::size_t>(e * cstride + offset)
                                + bs * i;
                            const std::size_t pos_v
                                = static_cast<std::size_t>(bs * dofs[i]);
                            for (int k = 0; k < bs; ++k)
                                coeffs[pos_c + k] = ua[pos_v + k];
                        }
                        (void)space_dim;
                    }
                }
            }
        }
    }

} // namespace hellofem::fem
