// hellofem::mesh — geometry imposed on a mesh
// SPDX-License-Identifier: MIT

#pragma once

#include "Topology.h"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "fem/ElementDofLayout.h"
#include "fem/dofmapbuilder.h"
#include "graph/AdjacencyList.h"
#include "graph/partition.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hellofem::mesh {

    /// Geometry stores the coordinates imposed on the entities of a
    /// mesh.
    ///
    /// The geometry degree-of-freedom map is a logically rectangular
    /// array (row-major) where each row corresponds to a cell and the
    /// columns are the indices in the coordinate array of the cell's
    /// coordinate degrees-of-freedom. Each cell type has its own dofmap.
    template <std::floating_point T>
    class Geometry {
    public:
        /// Value type.
        using value_type = T;

        /// Constructor of the geometry data holder.
        ///
        /// @param[in] index_map Index map of the geometry
        /// degrees-of-freedom.
        /// @param[in] dofmaps The geometry (point) dofmaps for each cell
        /// type, flattened row-major with `ndofs` columns each.
        /// @param[in] elements Coordinate elements, one per cell type
        /// (matching `dofmaps`).
        /// @param[in] x Point coordinates, flattened row-major with
        /// shape `(num_points, 3)`.
        /// @param[in] dim Geometric dimension (`0 < dim <= 3`).
        /// @param[in] input_global_indices Global index of each point,
        /// commonly from a mesh input file.
        template <typename U, typename V, typename W>
            requires std::is_convertible_v<std::remove_cvref_t<U>,
                         std::vector<std::vector<std::int32_t>>>
                         and std::is_convertible_v<std::remove_cvref_t<V>,
                             std::vector<T>>
                         and std::is_convertible_v<std::remove_cvref_t<W>,
                             std::vector<std::int64_t>>
        Geometry(std::shared_ptr<const common::IndexMap> index_map, U&& dofmaps,
            const std::vector<fem::CoordinateElement<
                typename std::remove_reference_t<typename V::value_type>>>&
                elements,
            V&& x, int dim, W&& input_global_indices)
            : _dim(dim), _dofmaps(std::forward<U>(dofmaps)), _index_map(std::move(index_map)), _cmaps(elements), _x(std::forward<V>(x)), _input_global_indices(std::forward<W>(input_global_indices))
        {
            assert(_x.size() % 3 == 0);
            if (_x.size() / 3 != _input_global_indices.size())
                throw std::runtime_error("Geometry size mismatch.");
            if (_dofmaps.size() != _cmaps.size())
                throw std::runtime_error(
                    "Geometry number of dofmaps not equal to the number of "
                    "coordinate elements.");
        }

        /// Copy constructor
        Geometry(const Geometry&) = default;
        /// Move constructor
        Geometry(Geometry&&) = default;
        /// Destructor
        ~Geometry() = default;

        // Copy assignment (deleted)
        Geometry& operator=(const Geometry&) = delete;
        /// Move assignment
        Geometry& operator=(Geometry&&) = default;

        /// Dimension of the Euclidean coordinate system.
        int dim() const { return _dim; }

        /// Dofmap for the geometry (single cell type). See dofmaps().
        [[deprecated("Use dofmaps().front() instead.")]]
        md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>
        dofmap() const
        {
            if (_dofmaps.size() != 1)
                throw std::runtime_error("Multiple dofmaps");
            const std::size_t ndofs = _cmaps.front().dim();
            return md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>(
                _dofmaps.front().data(), _dofmaps.front().size() / ndofs, ndofs);
        }

        /// Degree-of-freedom map for each coordinate element, with shape
        /// `(num_cells, dofs_per_cell)`.
        std::vector<md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>>
        dofmaps() const
        {
            std::vector<md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>>
                dms(_dofmaps.size());
            for (std::size_t i = 0; i < _dofmaps.size(); ++i) {
                const std::size_t ndofs = _cmaps.at(i).dim();
                dms[i] = md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>>(
                    _dofmaps.at(i).data(), _dofmaps.at(i).size() / ndofs, ndofs);
            }
            return dms;
        }

        /// Index map of the geometry degrees-of-freedom.
        std::shared_ptr<const common::IndexMap> index_map() const
        {
            return _index_map;
        }

        /// Geometry data, flattened row-major with shape `(num_points, 3)`.
        std::span<const value_type> x() const { return _x; }

        /// Geometry data (non-const version).
        std::span<value_type> x() { return _x; }

        /// Coordinate elements describing the geometry map.
        const std::vector<fem::CoordinateElement<value_type>>& cmaps() const
        {
            return _cmaps;
        }

        /// Global user indices of the geometry points.
        const std::vector<std::int64_t>& input_global_indices() const
        {
            return _input_global_indices;
        }

    private:
        // Geometric dimension.
        int _dim;

        // Map per cell for extracting coordinate data for each cmap.
        std::vector<std::vector<std::int32_t>> _dofmaps;

        // IndexMap for the geometry degrees-of-freedom.
        std::shared_ptr<const common::IndexMap> _index_map;

        // Coordinate elements.
        std::vector<fem::CoordinateElement<value_type>> _cmaps;

        // Coordinates of all points (row-major, 3 columns).
        std::vector<value_type> _x;

        // Global indices as provided on Geometry creation.
        std::vector<std::int64_t> _input_global_indices;
    };

    /// Template type deduction.
    template <typename U, typename V, typename W>
    Geometry(std::shared_ptr<const common::IndexMap>, U&&,
        const std::vector<fem::CoordinateElement<
            typename std::remove_reference_t<typename V::value_type>>>&,
        V&&, int, W&&)
        -> Geometry<typename std::remove_cvref_t<typename V::value_type>>;

    /// Build a Geometry from input data, given a topology, the
    /// coordinate elements, the cell geometry node (global) indices, the
    /// per-cell geometry dofmap (global indices), the node coordinates
    /// and the geometric dimension.
    ///
    /// @param[in] topology Mesh topology.
    /// @param[in] elements Coordinate elements, one per cell type.
    /// @param[in] nodes Sorted, unique geometry node global indices.
    /// @param[in] xdofs Geometry dofmap (global indices), one entry per
    /// (cell, local dof).
    /// @param[in] x Node coordinates, flattened row-major with shape
    /// `(num_nodes, dim)`.
    /// @param[in] dim Geometric dimension (1, 2 or 3).
    /// @param[in] reorder_fn Reordering function applied to the dofmap,
    /// or nullptr for none.
    /// @pre `nodes` must be sorted.
    template <typename U>
    Geometry<typename std::remove_reference_t<typename U::value_type>>
    create_geometry(const Topology& topology,
        const std::vector<fem::CoordinateElement<
            std::remove_reference_t<typename U::value_type>>>& elements,
        std::span<const std::int64_t> nodes,
        std::span<const std::int64_t> xdofs, const U& x, int dim,
        const std::function<std::vector<int>(
            const graph::AdjacencyList<std::int32_t>&)>& reorder_fn
        = nullptr)
    {
        spdlog::info("Create Geometry");

        assert(std::ranges::is_sorted(nodes));
        using T = typename std::remove_reference_t<typename U::value_type>;

        // Check elements match cell types in topology.
        const int tdim = topology.dim();
        const std::size_t num_cell_types = topology.entity_types(tdim).size();
        if (elements.size() != num_cell_types)
            throw std::runtime_error("Mismatch between topology and geometry.");

        std::vector<fem::ElementDofLayout> dof_layouts;
        dof_layouts.reserve(elements.size());
        for (auto& el : elements)
            dof_layouts.push_back(el.create_dof_layout());

        // Build the geometry dofmap on the topology.
        auto [dof_index_map, bs, dofmaps]
            = fem::build_dofmap_data(topology, dof_layouts, reorder_fn);
        auto dof_index_map_ptr
            = std::make_shared<common::IndexMap>(std::move(dof_index_map));

        // If the mesh has higher-order geometry, permute the dofmap.
        if (elements.front().needs_dof_permutations()) {
            const std::int32_t num_cells
                = topology.connectivity(topology.dim(), 0)->num_nodes();
            const std::vector<std::uint32_t>& cell_info
                = topology.get_cell_permutation_info();
            const int d = elements.front().dim();
            for (std::int32_t cell = 0; cell < num_cells; ++cell) {
                std::span dofs(dofmaps.front().data() + cell * d, d);
                elements.front().permute_inv(dofs, cell_info[cell]);
            }
        }

        // Local-to-global for dofs -> local-to-local for coordinates.
        std::vector<std::int32_t> all_dofmaps;
        for (auto q : dofmaps)
            all_dofmaps.insert(all_dofmaps.end(), q.begin(), q.end());

        const std::vector<std::int32_t> l2l
            = graph::build::compute_local_to_local(
                graph::build::compute_local_to_global(xdofs, all_dofmaps), nodes);

        // Allocate space for input global indices.
        std::vector<std::int64_t> igi(nodes.size());
        std::ranges::transform(l2l, igi.begin(),
            [&nodes](auto index) { return nodes[index]; });

        // Build coordinate dof array, copying coordinates to position.
        assert(x.size() % dim == 0);
        const std::size_t shape0 = x.size() / dim;
        const std::size_t shape1 = dim;
        std::vector<T> xg(3 * shape0, 0);
        for (std::size_t i = 0; i < shape0; ++i) {
            std::copy_n(std::next(x.begin(), shape1 * l2l[i]), shape1,
                std::next(xg.begin(), 3 * i));
        }

        return Geometry(dof_index_map_ptr, std::move(dofmaps), elements,
            std::move(xg), dim, std::move(igi));
    }

} // namespace hellofem::mesh
