// hellofem::fem — degree-of-freedom layout of a finite element on a cell
// SPDX-License-Identifier: MIT

#pragma once

#include <span>
#include <vector>

namespace hellofem::fem {

    /// Dof placement of a finite element on a reference cell.
    ///
    /// Dofs are associated with mesh entities: `entity_dofs[dim][entity]` is
    /// the list of cell-local dof indices sitting on that entity, and
    /// `entity_closure_dofs` extends each entity to the closure (the entity
    /// plus all lower-dimensional entities on its boundary). Sub-layouts
    /// describe views into a parent layout for mixed/vector elements.
    class ElementDofLayout {
    public:
        /// Constructor.
        ///
        /// @param[in] block_size Number of dofs co-located at each point.
        /// @param[in] entity_dofs Dofs per entity:
        /// `entity_dofs[dim][entity] = [dof0, dof1, ...]`.
        /// @param[in] entity_closure_dofs Dofs on each entity's closure.
        /// @param[in] parent_map Local-dof index map into the immediate
        /// parent layout (empty for a top-level layout).
        /// @param[in] sub_layouts Layouts of sub-elements.
        ElementDofLayout(
            int block_size,
            const std::vector<std::vector<std::vector<int>>>& entity_dofs,
            const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs,
            const std::vector<int>& parent_map,
            const std::vector<ElementDofLayout>& sub_layouts);

        /// Copy of the layout with any parent information discarded.
        ElementDofLayout copy() const;

        /// Copy constructor
        ElementDofLayout(const ElementDofLayout&) = default;
        /// Move constructor
        ElementDofLayout(ElementDofLayout&&) = default;
        /// Destructor
        ~ElementDofLayout() = default;
        /// Copy assignment
        ElementDofLayout& operator=(const ElementDofLayout&) = default;
        /// Move assignment
        ElementDofLayout& operator=(ElementDofLayout&&) = default;

        /// Equality on layout data (sub- and parent maps not compared).
        bool operator==(const ElementDofLayout& layout) const;

        /// Number of dofs on a cell.
        int num_dofs() const;

        /// Cell-local dof indices on entity (dim, entity_index).
        const std::vector<int>& entity_dofs(int dim, int entity_index) const;

        /// Cell-local dof indices on the closure of entity (dim, entity_index).
        const std::vector<int>& entity_closure_dofs(int dim,
            int entity_index) const;

        /// All entity dofs: `dof = _entity_dofs[dim][entity][i]`.
        const std::vector<std::vector<std::vector<int>>>& entity_dofs_all() const;

        /// All entity closure dofs.
        const std::vector<std::vector<std::vector<int>>>&
        entity_closure_dofs_all() const;

        /// Number of sub-layouts.
        int num_sub_dofmaps() const;

        /// Sub-layout selected by a component list (one level per index).
        const ElementDofLayout& sub_layout(std::span<const int> component) const;

        /// Cell-local dof indices of a sub-layout (selected by `component`)
        /// within this layout.
        std::vector<int> sub_view(std::span<const int> component) const;

        /// Block size.
        int block_size() const;

        /// True if this layout is a view into a parent layout.
        bool is_view() const;

    private:
        int _block_size;

        // Map of dofs to this layout's immediate parent
        std::vector<int> _parent_map;

        // Total number of dofs on a cell
        int _num_dofs;

        // Dofs per entity: dof = _entity_dofs[dim][entity][i]
        std::vector<std::vector<std::vector<int>>> _entity_dofs;

        // Dofs on entity closures
        std::vector<std::vector<std::vector<int>>> _entity_closure_dofs;

        // Sub-layouts
        std::vector<ElementDofLayout> _sub_dofmaps;
    };

} // namespace hellofem::fem
