// hellofem::fem — coordinate element (isoparametric geometry maps)
// SPDX-License-Identifier: MIT

#pragma once

#include "ElementDofLayout.h"
#include "basis/finite-element.h"
#include "common/math.h"
#include "mesh/cell_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>

namespace hellofem::fem {

    /// A CoordinateElement manages the coordinate map of an isoparametric
    /// cell, wrapping a basis Lagrange element on the reference cell.
    ///
    /// @tparam T Floating point type for geometry and basis evaluation.
    template <std::floating_point T>
    class CoordinateElement {
    public:
        /// Create from a basis finite element.
        explicit CoordinateElement(
            std::shared_ptr<const basis::FiniteElement<T>> element);

        /// Create a Lagrange coordinate element of given cell shape/degree.
        CoordinateElement(mesh::CellType celltype, int degree,
            basis::element::lagrange_variant type
            = basis::element::lagrange_variant::unset);

        /// Destructor
        ~CoordinateElement() = default;

        /// Cell shape.
        mesh::CellType cell_shape() const;

        /// Polynomial degree of the map.
        int degree() const;

        /// Dimension of the coordinate element space (number of basis
        /// functions).
        int dim() const;

        /// Lagrange variant of the element.
        basis::element::lagrange_variant variant() const;

        /// Element hash.
        std::size_t hash() const;

        /// Shape of the array @ref tabulate fills: (derivative index, point
        /// index, basis function index, basis component).
        std::array<std::size_t, 4> tabulate_shape(std::size_t nd,
            std::size_t num_points) const;

        /// Evaluate basis values and derivatives at a set of points.
        void tabulate(int nd, std::span<const T> X,
            std::array<std::size_t, 2> shape, std::span<T> basis) const;

        /// Permute the closure dofs of a sub-entity to match its mesh
        /// orientation (accounting for the parent cell's permutation info).
        void permute_subentity_closure(std::span<std::int32_t> d,
            std::uint32_t cell_info,
            mesh::CellType entity_type,
            int entity_index) const;

        /// Jacobian J (shape gdim x tdim) of the map, from basis
        /// derivatives `dphi` (tdim x num geometry nodes) and cell node
        /// coordinates `cell_geometry` (num nodes x gdim).
        template <typename U, typename V, typename W>
        static void compute_jacobian(const U& dphi, const V& cell_geometry, W&& J)
        {
            math::dot(cell_geometry, dphi, J, true);
        }

        /// Inverse Jacobian K (tdim x gdim). Pseudo-inverse when gdim != tdim.
        template <typename U, typename V>
        static void compute_jacobian_inverse(const U& J, V&& K)
        {
            if (J.extent(0) == K.extent(0))
                math::inv(J, K);
            else
                math::pinv(J, K);
        }

        /// Jacobian determinant. For gdim != tdim, uses sqrt(det(J^T J)).
        template <typename U>
        static double compute_jacobian_determinant(
            const U& J, std::span<typename U::value_type> w)
        {
            static_assert(U::rank() == 2, "Must be rank 2");
            if (J.extent(0) == J.extent(1))
                return math::det(J);
            else {
                assert(w.size() >= 2 * J.extent(0) * J.extent(1));
                using X = typename U::element_type;
                using mdspan2_t = md::mdspan<X, md::dextents<std::size_t, 2>>;
                mdspan2_t B(w.data(), J.extent(1), J.extent(0));
                mdspan2_t BA(w.data() + J.extent(0) * J.extent(1),
                    B.extent(0), J.extent(1));
                for (std::size_t i = 0; i < B.extent(0); ++i)
                    for (std::size_t j = 0; j < B.extent(1); ++j)
                        B(i, j) = J(j, i);

                std::fill_n(BA.data_handle(), BA.size(), 0);
                math::dot(B, J, BA);
                return std::sqrt(math::det(BA));
            }
        }

        /// Dof layout of this element.
        ElementDofLayout create_dof_layout() const;

        /// Push forward reference points to physical coordinates.
        template <typename U, typename V, typename W>
        static void push_forward(U&& x, const V& cell_geometry, const W& phi)
        {
            for (std::size_t i = 0; i < x.extent(0); ++i)
                for (std::size_t j = 0; j < x.extent(1); ++j)
                    x(i, j) = 0;
            math::dot(phi, cell_geometry, x);
        }

        /// Pull back physical points to reference coordinates for an affine
        /// map: X = K (x - x0).
        template <typename U, typename V, typename W>
        static void pull_back_affine(U&& X, const V& K, std::array<T, 3> x0,
            const W& x)
        {
            assert(X.extent(0) == x.extent(0));
            assert(X.extent(1) == K.extent(0));
            assert(x.extent(1) == K.extent(1));
            for (std::size_t i = 0; i < X.extent(0); ++i)
                for (std::size_t j = 0; j < X.extent(1); ++j)
                    X(i, j) = 0;

            for (std::size_t p = 0; p < x.extent(0); ++p)
                for (std::size_t i = 0; i < K.extent(0); ++i)
                    for (std::size_t j = 0; j < K.extent(1); ++j)
                        X(p, i) += K(i, j) * (x(p, j) - x0[j]);
        }

        /// mdspan rank-2 alias.
        template <typename X>
        using mdspan2_t = md::mdspan<X, md::dextents<std::size_t, 2>>;

        /// Pull back physical points to reference coordinates for a
        /// non-affine map via Newton iteration.
        void pull_back_nonaffine(mdspan2_t<T> X, mdspan2_t<const T> x,
            mdspan2_t<const T> cell_geometry,
            std::span<T> working_array, double tol,
            int maxit) const;

        /// Working array size required by @ref pull_back_nonaffine.
        std::size_t pull_back_working_size(std::size_t gdim) const;

        /// Permute cell-local dofs to match mesh orientation.
        void permute(std::span<std::int32_t> dofs, std::uint32_t cell_perm) const;

        /// Reverse of @ref permute.
        void permute_inv(std::span<std::int32_t> dofs,
            std::uint32_t cell_perm) const;

        /// True if geometry dofs need permuting on each cell (higher-order
        /// geometry with multiple dofs per sub-entity).
        bool needs_dof_permutations() const;

        /// True if the map is affine.
        bool is_affine() const noexcept { return _is_affine; }

    private:
        bool _is_affine;

        std::shared_ptr<const basis::FiniteElement<T>> _element;
    };

    /// Explicit instantiations.
    extern template class CoordinateElement<float>;
    extern template class CoordinateElement<double>;

} // namespace hellofem::fem
