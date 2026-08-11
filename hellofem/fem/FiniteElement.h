// hellofem::fem — model of a finite element
// SPDX-License-Identifier: MIT

#pragma once

#include "ElementDofLayout.h"
#include "basis/finite-element.h"
#include "basis/maps.h"
#include "mesh/cell_types.h"

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hellofem::fem {

    /// DOF transformation type.
    enum class doftransform : std::uint8_t {
        standard = 0, ///< Standard
        transpose = 1, ///< Transpose
        inverse = 2, ///< Inverse
        inverse_transpose = 3, ///< Transpose inverse
    };

    /// Model of a finite element. Provides the dof layout on a reference
    /// element, and methods for evaluating and transforming the basis.
    ///
    /// Wraps a `hellofem::basis::FiniteElement`, adding block and mixed
    /// structure plus a mesh-level cell type.
    template <std::floating_point T>
    class FiniteElement {
    public:
        /// Geometry type of the Mesh the element is defined on.
        using geometry_type = T;

        /// Create a finite element from a basis element.
        /// @param[in] element Basis finite element.
        /// @param[in] value_shape Value shape for a blocked element, e.g.
        /// `{3}` for a vector in 3D. Can only be set for a scalar base
        /// element.
        /// @param[in] symmetric Is the element a symmetric tensor? Only for
        /// rank-2 tensor blocked elements.
        FiniteElement(const basis::FiniteElement<geometry_type>& element,
            const std::optional<std::vector<std::size_t>>& value_shape
            = std::nullopt,
            bool symmetric = false);

        /// Create a mixed finite element from a list of finite elements.
        ///
        /// The ith sub-element can be accessed via `extract_sub_element`.
        /// Functions on mixed spaces cannot be interpolated into directly.
        /// @param[in] elements Finite elements composing the mixed element.
        FiniteElement(
            const std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>&
                elements);

        /// Create a quadrature element.
        /// @param[in] cell_type Cell type.
        /// @param[in] points Quadrature points.
        /// @param[in] pshape Shape of `points`.
        /// @param[in] value_shape Value shape for the element.
        /// @param[in] symmetric Is the element a symmetric tensor?
        FiniteElement(mesh::CellType cell_type,
            std::span<const geometry_type> points,
            std::array<std::size_t, 2> pshape,
            std::vector<std::size_t> value_shape = {}, bool symmetric = false);

        /// Copy constructor
        FiniteElement(const FiniteElement& element) = delete;

        /// Move constructor
        FiniteElement(FiniteElement&& element) = default;

        /// Destructor
        ~FiniteElement() = default;

        /// Copy assignment
        FiniteElement& operator=(const FiniteElement& element) = delete;

        /// Move assignment
        FiniteElement& operator=(FiniteElement&& element) = default;

        /// Equality. Only for non-mixed elements; mixed throws.
        bool operator==(const FiniteElement& e) const;

        /// Inequality. Only for non-mixed elements; mixed throws.
        bool operator!=(const FiniteElement& e) const;

        /// Cell shape the element is defined on.
        mesh::CellType cell_type() const noexcept;

        /// String identifying the finite element.
        std::string signature() const noexcept;

        /// Dimension of the function space (number of dofs). For blocked
        /// elements this is the full (physical) dimension.
        int space_dimension() const noexcept;

        /// Number of dofs collocated at each dof point (1 for non-blocked).
        int block_size() const noexcept;

        /// Number of components of the (physical) field.
        int value_size() const;

        /// Value shape of the field, e.g. `{}` scalar, `{2}` vector in 2D.
        std::span<const std::size_t> value_shape() const;

        /// Number of components of the base (reference) field.
        int reference_value_size() const;

        /// Value shape of the base (reference) field.
        std::span<const std::size_t> reference_value_shape() const;

        /// Dofs on each entity of the cell:
        /// `entity_dofs[entity_dim][entity_number] = [dof0, dof1, ...]`.
        const std::vector<std::vector<std::vector<int>>>&
        entity_dofs() const noexcept;

        /// Dofs on the closure of each entity of the cell.
        const std::vector<std::vector<std::vector<int>>>&
        entity_closure_dofs() const noexcept;

        /// True if the element is a symmetric tensor.
        bool symmetric() const;

        /// Compute basis values and derivatives at a set of points.
        void tabulate(std::span<geometry_type> values,
            std::span<const geometry_type> X,
            std::array<std::size_t, 2> shape, int order) const;

        /// Compute basis values and derivatives at a set of points.
        std::pair<std::vector<geometry_type>, std::array<std::size_t, 4>>
        tabulate(std::span<const geometry_type> X,
            std::array<std::size_t, 2> shape, int order) const;

        /// Number of sub-elements (block copies for blocked elements).
        int num_sub_elements() const noexcept;

        /// True if this is a mixed element (composed of distinct
        /// sub-elements rather than a single element).
        bool is_mixed() const noexcept;

        /// Sub-elements of a mixed or blocked element.
        const std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>&
        sub_elements() const noexcept;

        /// Extract a sub-element by a component list, one level per entry.
        std::shared_ptr<const FiniteElement<geometry_type>>
        extract_sub_element(const std::vector<int>& component) const;

        /// Underlying basis element. Throws for mixed/quadrature elements.
        const basis::FiniteElement<geometry_type>& basix_element() const;

        /// Map type of the underlying basis element.
        basis::maps::type map_type() const;

        /// True if the map is the identity (or element is quadrature).
        bool map_ident() const noexcept;

        /// True if the interpolation is the identity.
        bool interpolation_ident() const noexcept;

        /// Interpolation points (quadrature points if a quadrature element).
        std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>>
        interpolation_points() const;

        /// Interpolation operator (dofs -> coefficients).
        std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>>
        interpolation_operator() const;

        /// Interpolation operator from `from` into this element.
        std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>>
        create_interpolation_operator(const FiniteElement& from) const;

        /// True if dof transformations (not just permutations) are needed.
        bool needs_dof_transformations() const noexcept;

        /// True if dof permutations are needed.
        bool needs_dof_permutations() const noexcept;

        /// Permute dofs from reference to physical element ordering.
        void permute(std::span<std::int32_t> doflist,
            std::uint32_t cell_permutation) const;

        /// Apply the inverse permutation (physical to reference ordering).
        void permute_inv(std::span<std::int32_t> doflist,
            std::uint32_t cell_permutation) const;

        /// Return a function that applies a dof permutation to data.
        std::function<void(std::span<std::int32_t>, std::uint32_t)>
        dof_permutation_fn(bool inverse = false,
            bool scalar_element = false) const;

        /// ElementDofLayout for use with `fem::build_dofmap_data`.
        ElementDofLayout create_dof_layout() const;

        /// Return a function that applies a dof transformation operator to
        /// some data from the left (see `T_apply`).
        /// @param[in] ttype The transformation type.
        /// @param[in] scalar_element Whether to return the scalar
        /// transformations for a vector element.
        template <typename U>
        std::function<void(std::span<U>, std::span<const std::uint32_t>,
            std::int32_t, int)>
        dof_transformation_fn(doftransform ttype,
            bool scalar_element = false) const
        {
            if (!needs_dof_transformations()) {
                return [](std::span<U>, std::span<const std::uint32_t>,
                           std::int32_t, int) { };
            }

            if (!_sub_elements.empty()) {
                if (!_reference_value_shape) // Mixed element
                {
                    std::vector<std::function<void(std::span<U>,
                        std::span<const std::uint32_t>, std::int32_t, int)>>
                        sub_element_fns;
                    std::vector<int> dims;
                    for (std::size_t i = 0; i < _sub_elements.size(); ++i) {
                        sub_element_fns.push_back(
                            _sub_elements[i]
                                ->template dof_transformation_fn<U>(ttype));
                        dims.push_back(_sub_elements[i]->space_dimension());
                    }

                    return [dims = std::move(dims),
                               sub_element_fns = std::move(sub_element_fns)](
                               std::span<U> data,
                               std::span<const std::uint32_t> cell_info,
                               std::int32_t cell, int block_size) {
                        std::size_t offset = 0;
                        for (std::size_t e = 0; e < sub_element_fns.size(); ++e) {
                            const std::size_t width = dims[e] * block_size;
                            sub_element_fns[e](
                                data.subspan(offset, width), cell_info, cell,
                                block_size);
                            offset += width;
                        }
                    };
                }
                else if (!scalar_element) {
                    // Blocked element
                    std::function<void(std::span<U>,
                        std::span<const std::uint32_t>, std::int32_t, int)>
                        sub_fn
                        = _sub_elements.front()
                              ->template dof_transformation_fn<U>(ttype);
                    const int ebs = _bs;
                    return [ebs, sub_fn = std::move(sub_fn)](
                               std::span<U> data,
                               std::span<const std::uint32_t> cell_info,
                               std::int32_t cell, int data_block_size) {
                        sub_fn(data, cell_info, cell, ebs * data_block_size);
                    };
                }
            }

            switch (ttype) {
            case doftransform::inverse_transpose:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int block_size) {
                    Tt_inv_apply(data, cell_info[cell], block_size);
                };
            case doftransform::transpose:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int block_size) {
                    Tt_apply(data, cell_info[cell], block_size);
                };
            case doftransform::inverse:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int block_size) {
                    Tinv_apply(data, cell_info[cell], block_size);
                };
            case doftransform::standard:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int block_size) {
                    T_apply(data, cell_info[cell], block_size);
                };
            default:
                throw std::runtime_error("Unknown transformation type");
            }
        }

        /// Return a function that applies a dof transformation to transposed
        /// data from the right (see `T_apply_right`).
        template <typename U>
        std::function<void(std::span<U>, std::span<const std::uint32_t>,
            std::int32_t, int)>
        dof_transformation_right_fn(doftransform ttype,
            bool scalar_element = false) const
        {
            if (!needs_dof_transformations()) {
                return [](std::span<U>, std::span<const std::uint32_t>,
                           std::int32_t, int) { };
            }
            else if (!_sub_elements.empty()) {
                if (!_reference_value_shape) // Mixed element
                {
                    std::vector<std::function<void(std::span<U>,
                        std::span<const std::uint32_t>, std::int32_t, int)>>
                        sub_element_fns;
                    std::vector<int> dims;
                    for (std::size_t i = 0; i < _sub_elements.size(); ++i) {
                        sub_element_fns.push_back(_sub_elements[i]
                                ->template dof_transformation_right_fn<U>(
                                    ttype));
                        dims.push_back(_sub_elements[i]->space_dimension());
                    }

                    return [dims = std::move(dims),
                               sub_element_fns = std::move(sub_element_fns)](
                               std::span<U> data,
                               std::span<const std::uint32_t> cell_info,
                               std::int32_t cell, int block_size) {
                        std::size_t offset = 0;
                        for (std::size_t e = 0; e < sub_element_fns.size(); ++e) {
                            sub_element_fns[e](
                                data.subspan(offset, data.size() - offset),
                                cell_info, cell, block_size);
                            offset += dims[e];
                        }
                    };
                }
                else if (!scalar_element) {
                    // Blocked element. The left transformation can be used
                    // here as blocked elements use xyzxyzxyz ordering; applying
                    // the right transformation is equivalent to applying the
                    // left one to data in xxxyyyzzz ordering.
                    std::function<void(std::span<U>,
                        std::span<const std::uint32_t>, std::int32_t, int)>
                        sub_fn
                        = _sub_elements.front()
                              ->template dof_transformation_fn<U>(ttype);
                    return [this, sub_fn = std::move(sub_fn)](
                               std::span<U> data,
                               std::span<const std::uint32_t> cell_info,
                               std::int32_t cell, int data_block_size) {
                        const int ebs = block_size();
                        const std::size_t dof_count
                            = data.size() / data_block_size;
                        for (int block = 0; block < data_block_size; ++block) {
                            sub_fn(data.subspan(block * dof_count, dof_count),
                                cell_info, cell, ebs);
                        }
                    };
                }
            }

            switch (ttype) {
            case doftransform::inverse_transpose:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int n) {
                    Tt_inv_apply_right(data, cell_info[cell], n);
                };
            case doftransform::transpose:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int n) {
                    Tt_apply_right(data, cell_info[cell], n);
                };
            case doftransform::inverse:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int n) {
                    Tinv_apply_right(data, cell_info[cell], n);
                };
            case doftransform::standard:
                return [this](std::span<U> data,
                           std::span<const std::uint32_t> cell_info,
                           std::int32_t cell, int n) {
                    T_apply_right(data, cell_info[cell], n);
                };
            default:
                throw std::runtime_error("Unknown transformation type");
            }
        }

        /// Apply the dof transformation `phi = T phi_ref` to basis data.
        template <typename U>
        void T_apply(std::span<U> data, std::uint32_t cell_permutation,
            int n) const
        {
            assert(_element);
            _element->T_apply(data, n, cell_permutation);
        }

        /// Apply `v = T^{-T} u`.
        template <typename U>
        void Tt_inv_apply(std::span<U> data, std::uint32_t cell_permutation,
            int n) const
        {
            assert(_element);
            _element->Tt_inv_apply(data, n, cell_permutation);
        }

        /// Apply `u <- T^{T} u`.
        template <typename U>
        void Tt_apply(std::span<U> data, std::uint32_t cell_permutation,
            int n) const
        {
            assert(_element);
            _element->Tt_apply(data, n, cell_permutation);
        }

        /// Apply `v = T^{-1} u`.
        template <typename U>
        void Tinv_apply(std::span<U> data, std::uint32_t cell_permutation,
            int n) const
        {
            assert(_element);
            _element->Tinv_apply(data, n, cell_permutation);
        }

        /// Apply `v^{T} = u^{T} T`.
        template <typename U>
        void T_apply_right(std::span<U> data, std::uint32_t cell_permutation,
            int n) const
        {
            assert(_element);
            _element->T_apply_right(data, n, cell_permutation);
        }

        /// Apply `v^{T} = u^{T} T^{-1}`.
        template <typename U>
        void Tinv_apply_right(std::span<U> data,
            std::uint32_t cell_permutation, int n) const
        {
            assert(_element);
            _element->Tinv_apply_right(data, n, cell_permutation);
        }

        /// Apply `v^{T} = u^{T} T^{T}`.
        template <typename U>
        void Tt_apply_right(std::span<U> data, std::uint32_t cell_permutation,
            int n) const
        {
            assert(_element);
            _element->Tt_apply_right(data, n, cell_permutation);
        }

        /// Apply `v^{T} = u^{T} T^{-T}`.
        template <typename U>
        void Tt_inv_apply_right(std::span<U> data,
            std::uint32_t cell_permutation, int n) const
        {
            assert(_element);
            _element->Tt_inv_apply_right(data, n, cell_permutation);
        }

    private:
        // Value shape of the physical field (nullopt for mixed)
        std::optional<std::vector<std::size_t>> _value_shape;

        // Block size
        int _bs;

        // Mesh cell type
        mesh::CellType _cell_type;

        // Element signature
        std::string _signature;

        // Dimension of the function space
        int _space_dim;

        // Sub-elements (block copies for blocked elements)
        std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>
            _sub_elements;

        // Value shape of the base (reference) field (nullopt for mixed)
        std::optional<std::vector<std::size_t>> _reference_value_shape;

        // Underlying basis element (null for mixed/quadrature)
        std::unique_ptr<basis::FiniteElement<geometry_type>> _element;

        // Symmetry flag
        bool _symmetric;

        // Whether dof permutations / transformations are required
        bool _needs_dof_permutations;
        bool _needs_dof_transformations;

        // Dofs on each entity of the cell
        std::vector<std::vector<std::vector<int>>> _entity_dofs;
        std::vector<std::vector<std::vector<int>>> _entity_closure_dofs;

        // Quadrature points (0-dim for non-quadrature elements)
        std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>>
            _points;
    };

} // namespace hellofem::fem
