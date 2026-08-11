// hellofem::fem — degree-of-freedom layout of a finite element on a cell
// SPDX-License-Identifier: MIT

#include "ElementDofLayout.h"

#include <cassert>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>

namespace hellofem::fem {

    ElementDofLayout::ElementDofLayout(
        int block_size,
        const std::vector<std::vector<std::vector<int>>>& entity_dofs,
        const std::vector<std::vector<std::vector<int>>>& entity_closure_dofs,
        const std::vector<int>& parent_map,
        const std::vector<ElementDofLayout>& sub_layouts)
        : _block_size(block_size), _parent_map(parent_map), _num_dofs(0), _entity_dofs(entity_dofs), _entity_closure_dofs(entity_closure_dofs), _sub_dofmaps(sub_layouts)
    {
        assert(entity_dofs.size() == _entity_closure_dofs.size());
        for (std::size_t dim = 0; dim < entity_dofs.size(); ++dim) {
            assert(!entity_dofs[dim].empty());
            assert(!_entity_closure_dofs[dim].empty());
            _num_dofs = std::accumulate(
                entity_dofs[dim].begin(), entity_dofs[dim].end(), _num_dofs,
                [](int a, const auto& b) { return a + b.size(); });
        }
    }

    ElementDofLayout ElementDofLayout::copy() const
    {
        ElementDofLayout layout(*this);
        layout._parent_map.clear();
        return layout;
    }

    bool ElementDofLayout::operator==(const ElementDofLayout& layout) const
    {
        return _num_dofs == layout._num_dofs
            and _entity_dofs == layout._entity_dofs
            and _entity_closure_dofs == layout._entity_closure_dofs;
    }

    int ElementDofLayout::num_dofs() const { return _num_dofs; }

    const std::vector<int>&
    ElementDofLayout::entity_dofs(int dim, int entity_index) const
    {
        return _entity_dofs.at(dim).at(entity_index);
    }

    const std::vector<int>&
    ElementDofLayout::entity_closure_dofs(int dim, int entity_index) const
    {
        return _entity_closure_dofs.at(dim).at(entity_index);
    }

    const std::vector<std::vector<std::vector<int>>>&
    ElementDofLayout::entity_dofs_all() const
    {
        return _entity_dofs;
    }

    const std::vector<std::vector<std::vector<int>>>&
    ElementDofLayout::entity_closure_dofs_all() const
    {
        return _entity_closure_dofs;
    }

    int ElementDofLayout::num_sub_dofmaps() const { return _sub_dofmaps.size(); }

    const ElementDofLayout&
    ElementDofLayout::sub_layout(std::span<const int> component) const
    {
        if (component.empty())
            throw std::runtime_error("No sub dofmap specified");
        std::reference_wrapper<const ElementDofLayout> current
            = _sub_dofmaps.at(component[0]);
        for (std::size_t i = 1; i < component.size(); ++i)
            current = _sub_dofmaps.at(component[i]);
        return current;
    }

    std::vector<int>
    ElementDofLayout::sub_view(std::span<const int> component) const
    {
        // Start from the full dof range, then pull out the parent dofs of
        // each sub-level.
        std::vector<int> dof_list(_num_dofs * _block_size);
        std::iota(dof_list.begin(), dof_list.end(), 0);

        const ElementDofLayout* current = this;
        for (int i : component) {
            if (i >= static_cast<int>(current->_sub_dofmaps.size()))
                throw std::runtime_error("Invalid component");
            current = &_sub_dofmaps.at(i);

            std::vector<int> dof_list_new(current->_num_dofs
                * current->_block_size);
            for (std::size_t j = 0; j < dof_list_new.size(); ++j)
                dof_list_new[j] = dof_list[current->_parent_map[j]];
            dof_list = std::move(dof_list_new);
        }

        return dof_list;
    }

    int ElementDofLayout::block_size() const { return _block_size; }

    bool ElementDofLayout::is_view() const { return !_parent_map.empty(); }

} // namespace hellofem::fem
