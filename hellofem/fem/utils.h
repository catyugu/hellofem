// hellofem::fem — shared assembly utilities
// SPDX-License-Identifier: MIT

#pragma once

#include "ElementDofLayout.h"
#include "FiniteElement.h"
#include "FunctionSpace.h"
#include "graph/AdjacencyList.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace hellofem::fem {

    /// Permutation of one cell's dofs, as
    /// `FiniteElement::dof_permutation_fn` returns it.
    using DofPermutation
        = std::function<void(std::span<std::int32_t>, std::uint32_t)>;

    /// Reordering of a whole dofmap, as `build_dofmap_data` takes it.
    using DofmapReordering = std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>;

    /// Flattened `(cell, local_facet)` pairs for the exterior facets of a
    /// mesh. Each exterior facet is encoded as the single cell it is
    /// attached to together with the facet's local index in that cell.
    std::vector<std::int32_t> exterior_facet_entities(const mesh::Topology& topology);

    /// Flattened `(cell+, local_facet+, cell-, local_facet-)` quadruples
    /// for the interior facets of a mesh (every facet shared by exactly
    /// two cells).
    std::vector<std::int32_t> interior_facet_entities(const mesh::Topology& topology);

    /// The dofmap of a single element on a mesh: the mesh entities its dofs
    /// live on are created, the dofmap is built, and each cell's dofs are
    /// permuted into the element's reference ordering.
    ///
    /// @param[in] mesh The mesh.
    /// @param[in] layout The element's dof layout.
    /// @param[in] permute_inv Permutation of a cell's dofs, or nullptr for
    /// an element whose dof transformations are identity (P1 and P2
    /// Lagrange). An entity shared by two cells is traversed in opposite
    /// directions, so the dofs along it — two on a P3 edge, one on a P3
    /// face — are numbered differently in each cell unless the permutation
    /// puts them back into the element's reference order, and the two cells
    /// then disagree about which dof is which.
    /// @param[in] reorder_fn Graph reordering of the dofmap (or nullptr).
    DofMap create_dofmap(const mesh::Mesh<double>& mesh,
        const ElementDofLayout& layout, const DofPermutation& permute_inv,
        const DofmapReordering& reorder_fn = nullptr);

    /// The function space of an element on a mesh, its dofmap permuted as
    /// `create_dofmap` does.
    std::shared_ptr<FunctionSpace<double>> create_functionspace(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const FiniteElement<double>> element,
        const DofmapReordering& reorder_fn = nullptr);

} // namespace hellofem::fem
