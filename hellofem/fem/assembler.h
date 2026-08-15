// hellofem::fem — assembly of forms into matrices and vectors
// SPDX-License-Identifier: MIT

#pragma once

#include "Constant.h"
#include "DirichletBC.h"
#include "Form.h"
#include "Function.h"
#include "assemble_matrix_impl.h"
#include "assemble_scalar_impl.h"
#include "assemble_vector_impl.h"
#include "pack.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace hellofem::fem {

    /// Assemble the cell and facet integrals of a linear form into a vector.
    /// @param[in,out] b The vector to accumulate into.
    /// @param[in] L The linear form.
    template <typename V, std::floating_point T>
    void assemble_vector(V&& b, const Form<T>& L)
    {
        std::span<T> barr(b.array());
        auto constants = pack_constants(L);
        auto storage = allocate_coefficient_storage(L);
        pack_coefficients(L, storage);

        for (IntegralType type : L.integral_types()) {
            const int num = L.num_integrals(type);
            for (int idx = 0; idx < num; ++idx) {
                auto& [coeffs, cstride] = storage.at({type, idx});
                const auto& kernel = L.kernel(type, idx, 0);
                auto cells = L.domain(type, idx, 0);

                // Dof transformations (identity for Lagrange).
                const DofMap& dofmap = *L.function_spaces().front()->dofmap();
                auto P0 = L.function_spaces().front()->element()->template dof_transformation_fn<T>(
                    doftransform::standard);

                // Preallocate scratch.
                const auto x_dofmap = L.mesh()->geometry().dofmaps().front();
                const std::size_t ngeom = x_dofmap.extent(1);
                std::vector<T> be_b(static_cast<std::size_t>(2)
                    * dofmap.bs() * dofmap.map().extent(1));
                std::vector<T> cdofs_b(static_cast<std::size_t>(6) * ngeom);
                std::span<const std::uint32_t> empty_cell_info;

                switch (type) {
                case IntegralType::cell:
                    impl::assemble_cells_vector(P0, barr, L.mesh()->geometry(),
                        cells, dofmap, kernel, std::span<const T>(constants),
                        coeffs.data(), cstride, empty_cell_info, std::span(be_b),
                        std::span(cdofs_b));
                    break;
                case IntegralType::exterior_facet:
                    impl::assemble_entities_vector(P0, barr, L.mesh()->geometry(),
                        cells, dofmap, kernel, std::span<const T>(constants),
                        coeffs.data(), cstride, empty_cell_info, std::span(be_b),
                        std::span(cdofs_b));
                    break;
                case IntegralType::interior_facet:
                    impl::assemble_interior_facets_vector(P0, barr,
                        L.mesh()->geometry(), cells, dofmap, kernel,
                        std::span<const T>(constants), coeffs.data(), cstride,
                        empty_cell_info, std::span(be_b), std::span(cdofs_b));
                    break;
                }
            }
        }
    }

    /// Assemble the cell integrals of a bilinear form into a matrix,
    /// zeroing rows/cols marked by the boundary conditions.
    /// @param[in] mat_add Functor matching `la::MatSet` (e.g.
    /// `matrix.mat_add_values()`).
    /// @param[in] a The bilinear form.
    /// @param[in] bc0 Marker for BC rows.
    /// @param[in] bc1 Marker for BC columns.
    template <std::floating_point T>
    void assemble_matrix(la::MatSet<T> auto mat_add, const Form<T>& a,
        std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1)
    {
        auto constants = pack_constants(a);
        auto storage = allocate_coefficient_storage(a);
        pack_coefficients(a, storage);

        for (IntegralType type : a.integral_types()) {
            const int num = a.num_integrals(type);
            for (int idx = 0; idx < num; ++idx) {
                auto& [coeffs, cstride] = storage.at({type, idx});
                const auto& kernel = a.kernel(type, idx, 0);
                auto cells = a.domain(type, idx, 0);

                const DofMap& dofmap0 = *a.function_spaces()[0]->dofmap();
                const DofMap& dofmap1 = *a.function_spaces()[1]->dofmap();
                auto P0 = a.function_spaces()[0]->element()->template dof_transformation_fn<T>(
                    doftransform::standard);
                auto P1T = a.function_spaces()[1]->element()->template dof_transformation_right_fn<T>(
                    doftransform::transpose);

                const auto x_dofmap = a.mesh()->geometry().dofmaps().front();
                const std::size_t ngeom = x_dofmap.extent(1);
                std::vector<T> Ab(static_cast<std::size_t>(4)
                    * dofmap0.bs() * dofmap0.map().extent(1)
                    * static_cast<std::size_t>(dofmap1.bs())
                    * dofmap1.map().extent(1));
                std::vector<T> cdofs_b(static_cast<std::size_t>(6) * ngeom);
                std::span<const std::uint32_t> empty_cell_info;

                switch (type) {
                case IntegralType::cell:
                    impl::assemble_cells_matrix<false>(mat_add,
                        a.mesh()->geometry(), cells, dofmap0, P0, dofmap1, P1T,
                        bc0, bc1, kernel, coeffs.data(), cstride,
                        std::span<const T>(constants), empty_cell_info,
                        empty_cell_info, std::span(Ab), std::span(cdofs_b));
                    break;
                case IntegralType::exterior_facet:
                    impl::assemble_entities_matrix<false>(mat_add,
                        a.mesh()->geometry(), cells, dofmap0, P0, dofmap1, P1T,
                        bc0, bc1, kernel, coeffs.data(), cstride,
                        std::span<const T>(constants), empty_cell_info,
                        empty_cell_info, std::span(Ab), std::span(cdofs_b));
                    break;
                case IntegralType::interior_facet:
                    impl::assemble_interior_facets_matrix<false>(mat_add,
                        a.mesh()->geometry(), cells, dofmap0, P0, dofmap1, P1T,
                        bc0, bc1, kernel, coeffs.data(), cstride,
                        std::span<const T>(constants), empty_cell_info,
                        empty_cell_info, std::span(Ab), std::span(cdofs_b));
                    break;
                }
            }
        }
    }

    /// Assemble a bilinear form, building the BC markers from a list of
    /// boundary conditions.
    template <std::floating_point T>
    void assemble_matrix(la::MatSet<T> auto mat_add, const Form<T>& a,
        const std::vector<std::reference_wrapper<const DirichletBC<T>>>& bcs)
    {
        const std::size_t n = a.function_spaces()[0]->dofmap()->index_map_bs()
            * a.function_spaces()[0]->dofmap()->index_map->size_local();
        std::vector<std::int8_t> bc0(n, 0), bc1(n, 0);
        for (const auto& bc : bcs) {
            bc.get().mark_dofs(bc0);
            bc.get().mark_dofs(bc1);
        }
        assemble_matrix(mat_add, a, bc0, bc1);
    }

    /// Set diagonal entries of a matrix to `diagonal` for dofs marked by
    /// boundary conditions. Used after assembling with BCs (which zero
    /// BC rows) so the linear system is non-singular.
    template <std::floating_point T>
    void set_diagonal(la::MatSet<T> auto mat_set, const Form<T>& a,
        const std::vector<std::reference_wrapper<const DirichletBC<T>>>& bcs,
        T diagonal)
    {
        const std::size_t n = a.function_spaces()[0]->dofmap()->index_map_bs()
            * a.function_spaces()[0]->dofmap()->index_map->size_local();
        std::vector<std::int8_t> markers(n, 0);
        for (const auto& bc : bcs)
            bc.get().mark_dofs(markers);

        std::vector<std::int32_t> dofs;
        for (std::size_t i = 0; i < n; ++i)
            if (markers[i])
                dofs.push_back(static_cast<std::int32_t>(i));
        if (dofs.empty())
            return;

        // Set each diagonal entry individually.
        for (std::int32_t d : dofs) {
            std::array<std::int32_t, 1> one {d};
            std::array<T, 1> val {diagonal};
            mat_set(one, one, val);
        }
    }

    /// Apply the Dirichlet lifting to a vector: `b -= A[:,bc] * alpha *
    /// (g - x0)`.
    /// @param[in,out] b The vector to modify.
    /// @param[in] a The bilinear form (matrix).
    /// @param[in] bcs The boundary conditions.
    /// @param[in] x0 Previous value (may be nullopt).
    /// @param[in] alpha Scaling factor.
    template <typename V, std::floating_point T>
    void apply_lifting(V&& b, const Form<T>& a,
        const std::vector<std::reference_wrapper<const DirichletBC<T>>>& bcs,
        std::optional<std::span<const T>> x0, T alpha)
    {
        // Mark BC dofs.
        const std::size_t n = a.function_spaces()[0]->dofmap()->index_map_bs()
            * a.function_spaces()[0]->dofmap()->index_map->size_local();
        std::vector<std::int8_t> bc0(n, 0), bc1(n, 0);
        for (const auto& bc : bcs) {
            bc.get().mark_dofs(bc0);
            bc.get().mark_dofs(bc1);
        }

        // The BC value g - x0, extended to zero on non-BC dofs.
        std::vector<T> g(n, 0);
        for (const auto& bc : bcs) {
            std::vector<T> x(n);
            bc.get().set(std::span(x), x0, alpha);
            auto dofs = bc.get().dof_indices();
            for (std::int32_t d : dofs)
                g[static_cast<std::size_t>(d)] = x[static_cast<std::size_t>(d)];
        }

        // b -= A[:, bc] * g: assemble the matrix restricted to BC columns
        // and multiply.
        auto constants = pack_constants(a);
        auto storage = allocate_coefficient_storage(a);
        pack_coefficients(a, storage);

        // Build a column-extraction operator: multiply the assembled
        // matrix block by g. We assemble the full element tensors but
        // only keep BC columns, accumulating b -= A*g.
        for (IntegralType type : a.integral_types()) {
            const int num = a.num_integrals(type);
            for (int idx = 0; idx < num; ++idx) {
                auto& [coeffs, cstride] = storage.at({type, idx});
                const auto& kernel = a.kernel(type, idx, 0);
                auto cells = a.domain(type, idx, 0);

                const DofMap& dofmap0 = *a.function_spaces()[0]->dofmap();
                const DofMap& dofmap1 = *a.function_spaces()[1]->dofmap();
                auto P0 = a.function_spaces()[0]->element()->template dof_transformation_fn<T>(
                    doftransform::standard);
                auto P1T = a.function_spaces()[1]->element()->template dof_transformation_right_fn<T>(
                    doftransform::transpose);

                const auto x_dofmap = a.mesh()->geometry().dofmaps().front();
                const std::size_t ngeom = x_dofmap.extent(1);
                const int bs0 = dofmap0.bs();
                const int bs1 = dofmap1.bs();
                const std::size_t ndofs0 = dofmap0.map().extent(1);
                const std::size_t ndofs1 = dofmap1.map().extent(1);
                const std::size_t ndim0 = static_cast<std::size_t>(bs0) * ndofs0;
                const std::size_t ndim1 = static_cast<std::size_t>(bs1) * ndofs1;
                std::vector<T> Ab(ndim0 * ndim1);
                std::vector<T> cdofs_b(3 * ngeom);
                std::span<const std::uint32_t> empty_cell_info;

                impl::assemble_cells_matrix<true>(
                    [&](std::span<const std::int32_t> dofs0,
                        std::span<const std::int32_t> dofs1,
                        std::span<const T> Ae) {
                        // Accumulate b -= Ae[:, bc_cols] * g[bc_cols].
                        for (std::size_t i = 0; i < ndofs0; ++i)
                            for (int k0 = 0; k0 < bs0; ++k0) {
                                const std::size_t row
                                    = static_cast<std::size_t>(bs0 * dofs0[i]) + k0;
                                T acc = 0;
                                for (std::size_t j = 0; j < ndofs1; ++j)
                                    for (int k1 = 0; k1 < bs1; ++k1) {
                                        const std::size_t col
                                            = static_cast<std::size_t>(bs1 * dofs1[j]) + k1;
                                        if (bc1[col])
                                            acc += Ae[(bs0 * i + k0) * ndim1
                                                       + (bs1 * j + k1)]
                                                * g[col];
                                    }
                                b[row] -= acc;
                            }
                        return 0;
                    },
                    a.mesh()->geometry(), cells, dofmap0, P0, dofmap1, P1T,
                    bc0, bc1, kernel, coeffs.data(), cstride,
                    std::span<const T>(constants), empty_cell_info,
                    empty_cell_info, std::span(Ab), std::span(cdofs_b));
            }
        }
    }

    /// Set boundary condition values in a vector: `x[dof] = alpha *
    /// (g[dof] - x0[dof])`.
    template <std::floating_point T>
    void set_bc(std::span<T> x,
        const std::vector<std::reference_wrapper<const DirichletBC<T>>>& bcs,
        std::optional<std::span<const T>> x0 = std::nullopt, T alpha = 1.0)
    {
        for (const auto& bc : bcs)
            bc.get().set(x, x0, alpha);
    }

    /// Assemble a scalar-valued (functional) form.
    template <std::floating_point T>
    T assemble_scalar(const Form<T>& M)
    {
        auto constants = pack_constants(M);
        auto storage = allocate_coefficient_storage(M);
        pack_coefficients(M, storage);

        T value {0};
        for (IntegralType type : M.integral_types()) {
            const int num = M.num_integrals(type);
            for (int idx = 0; idx < num; ++idx) {
                auto& [coeffs, cstride] = storage.at({type, idx});
                const auto& kernel = M.kernel(type, idx, 0);
                auto cells = M.domain(type, idx, 0);
                const auto x_dofmap = M.mesh()->geometry().dofmaps().front();
                const std::size_t ngeom = x_dofmap.extent(1);
                std::vector<T> cdofs_b(static_cast<std::size_t>(6) * ngeom);

                switch (type) {
                case IntegralType::cell:
                    value += impl::assemble_cells_scalar(M.mesh()->geometry(),
                        cells, kernel, std::span<const T>(constants),
                        coeffs.data(), cstride, std::span(cdofs_b));
                    break;
                case IntegralType::exterior_facet:
                    value += impl::assemble_entities_scalar(M.mesh()->geometry(),
                        cells, kernel, std::span<const T>(constants),
                        coeffs.data(), cstride, std::span(cdofs_b));
                    break;
                case IntegralType::interior_facet:
                    value += impl::assemble_interior_facets_scalar(
                        M.mesh()->geometry(), cells, kernel,
                        std::span<const T>(constants), coeffs.data(), cstride,
                        std::span(cdofs_b));
                    break;
                }
            }
        }
        return value;
    }

} // namespace hellofem::fem
