// hellofem::fem — a variational form
// SPDX-License-Identifier: MIT

#pragma once

#include "Constant.h"
#include "Function.h"
#include "FunctionSpace.h"
#include "kernel.h"
#include "mesh/Mesh.h"

#include <concepts>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace hellofem::fem {

    /// A variational form, e.g. a bilinear form `a(u, v)` or a linear
    /// form `L(v)`, defined by integral kernels over the mesh.
    ///
    /// A form is a list of integrals, one per (integral type, index)
    /// pair, each providing a kernel, the integration domain (cells or
    /// facets), and the set of coefficients the integral depends on.
    template <std::floating_point T, std::floating_point U = double>
    class Form {
    public:
        /// Geometry type.
        using geometry_type = U;

        /// Data of a single integral.
        struct integral_data {
            /// Kernel computing the element tensor.
            kernel_t<T, U> kernel;

            /// Integration domain. For `cell`: cell indices. For
            /// `exterior_facet`: flattened `(cell, local_facet)` pairs.
            /// For `interior_facet`: flattened
            /// `(cell+, local_facet+, cell-, local_facet-)` quadruples.
            std::vector<std::int32_t> entities;

            /// Indices (into `coefficients`) of the coefficients this
            /// integral depends on.
            std::vector<int> coeffs;
        };

        /// Create a form.
        /// @param[in] function_spaces Function spaces of the arguments
        /// (0 for a functional, 1 for a linear form, 2 for a bilinear).
        /// @param[in] integrals Integrals keyed by `(integral type, index)`.
        /// @param[in] mesh The mesh.
        /// @param[in] coefficients Coefficient functions.
        /// @param[in] constants Constants.
        Form(std::vector<std::shared_ptr<const FunctionSpace<T>>> function_spaces,
            std::map<std::pair<IntegralType, int>, std::vector<integral_data>>
                integrals,
            std::shared_ptr<const mesh::Mesh<T>> mesh,
            std::vector<std::shared_ptr<const Function<T>>> coefficients,
            std::vector<std::shared_ptr<const Constant<T>>> constants)
            : _function_spaces(std::move(function_spaces)), _integrals(std::move(integrals)), _mesh(std::move(mesh)), _coefficients(std::move(coefficients)), _constants(std::move(constants))
        {
            // Facet kernels in this build are written without permutation
            // handling; keep the flag off until interior-facet assembly
            // with permutations is added.
            _needs_facet_permutations = false;
        }

        /// Number of arguments (0, 1 or 2).
        int rank() const { return static_cast<int>(_function_spaces.size()); }

        /// The mesh the form is defined on.
        std::shared_ptr<const mesh::Mesh<T>> mesh() const { return _mesh; }

        /// Function spaces of the arguments.
        std::span<const std::shared_ptr<const FunctionSpace<T>>>
        function_spaces() const
        {
            return _function_spaces;
        }

        /// Coefficient functions.
        std::span<const std::shared_ptr<const Function<T>>> coefficients() const
        {
            return _coefficients;
        }

        /// Constants.
        std::span<const std::shared_ptr<const Constant<T>>> constants() const
        {
            return _constants;
        }

        /// Number of integrals of a given type.
        int num_integrals(IntegralType type) const
        {
            return static_cast<int>(_integrals.count({type, 0}));
        }

        /// Integral types present in the form.
        std::vector<IntegralType> integral_types() const
        {
            std::vector<IntegralType> types;
            for (const auto& [key, _] : _integrals)
                if (types.empty() or types.back() != key.first)
                    types.push_back(key.first);
            return types;
        }

        /// The kernel of integral `idx` of `type`, for cell type `ct`.
        const kernel_t<T, U>& kernel(IntegralType type, int idx,
            std::size_t ct) const
        {
            return _integrals.at({type, idx}).at(ct).kernel;
        }

        /// Integration domain of integral `idx` of `type`.
        std::span<const std::int32_t> domain(IntegralType type, int idx,
            std::size_t ct) const
        {
            return _integrals.at({type, idx}).at(ct).entities;
        }

        /// Coefficients of integral `idx` of `type`.
        std::span<const int> active_coeffs(IntegralType type, int idx) const
        {
            return _integrals.at({type, idx}).front().coeffs;
        }

        /// True if any integral needs facet permutations.
        bool needs_facet_permutations() const { return _needs_facet_permutations; }

        /// Cumulative space dimensions of the coefficients; the last
        /// entry is the total coefficient size.
        std::vector<int> coefficient_offsets() const
        {
            std::vector<int> offsets;
            offsets.reserve(_coefficients.size() + 1);
            offsets.push_back(0);
            for (const auto& c : _coefficients)
                offsets.push_back(offsets.back()
                    + c->function_space()->element()->space_dimension());
            return offsets;
        }

    private:
        // Argument function spaces.
        std::vector<std::shared_ptr<const FunctionSpace<T>>> _function_spaces;

        // Integrals per (type, index), one entry per cell type.
        std::map<std::pair<IntegralType, int>, std::vector<integral_data>>
            _integrals;

        // Mesh.
        std::shared_ptr<const mesh::Mesh<T>> _mesh;

        // Coefficients.
        std::vector<std::shared_ptr<const Function<T>>> _coefficients;

        // Constants.
        std::vector<std::shared_ptr<const Constant<T>>> _constants;

        // Whether facet permutations are needed.
        bool _needs_facet_permutations = false;
    };

} // namespace hellofem::fem
