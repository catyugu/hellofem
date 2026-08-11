// hellofem::fem — Dirichlet boundary conditions
// SPDX-License-Identifier: MIT

#include "DirichletBC.h"

#include <algorithm>
#include <iterator>
#include <set>

namespace hellofem::fem {

    template <std::floating_point T>
    std::vector<std::int32_t> DirichletBC<T>::locate_dofs_topological(
        const mesh::Topology& topology, const DofMap& dofmap, int dim,
        std::span<const std::int32_t> entities)
    {
        // For each entity, the cell-local dofs on its closure, mapped
        // through the dofmap of the owning cell to global dofs.
        const ElementDofLayout& layout = dofmap.element_dof_layout();
        std::vector<std::int32_t> dofs;
        std::set<std::int32_t> seen;

        const int tdim = topology.dim();
        auto entity_to_cells = topology.connectivity(dim, tdim);
        auto cell_to_entities = topology.connectivity(tdim, dim);

        for (std::int32_t entity : entities) {
            auto cells = entity_to_cells->links(entity);
            for (std::int32_t cell : cells) {
                // Local index of the entity in the cell (position in the
                // cell's entity list).
                auto cell_entities = cell_to_entities->links(cell);
                auto it = std::find(cell_entities.begin(), cell_entities.end(),
                    entity);
                const std::int32_t local_index
                    = static_cast<std::int32_t>(it - cell_entities.begin());

                auto cell_dofs = dofmap.cell_dofs(cell);
                const auto& entity_dofs
                    = layout.entity_closure_dofs(dim, local_index);
                for (int d : entity_dofs) {
                    const std::int32_t global
                        = cell_dofs[static_cast<std::size_t>(d)];
                    if (seen.insert(global).second)
                        dofs.push_back(global);
                }
            }
        }

        std::sort(dofs.begin(), dofs.end());
        return dofs;
    }

    template <std::floating_point T>
    std::vector<std::int32_t> DirichletBC<T>::locate_dofs_geometrical(
        const FunctionSpace<T>& V, std::function<bool(std::span<const T>)> marker_fn)
    {
        // Coordinates of every dof, (num_dofs, gdim).
        auto coords = V.tabulate_dof_coordinates(false);
        const int gdim = V.mesh()->geometry().dim();
        std::vector<std::int32_t> dofs;
        const std::int64_t num_dofs
            = V.dofmap()->index_map_bs() * V.dofmap()->index_map->size_local();
        for (std::int64_t d = 0; d < num_dofs; ++d)
            if (marker_fn(std::span(coords).subspan(static_cast<std::size_t>(d) * gdim,
                    static_cast<std::size_t>(gdim))))
                dofs.push_back(static_cast<std::int32_t>(d));
        return dofs;
    }

    template class DirichletBC<double>;
    template class DirichletBC<float>;

} // namespace hellofem::fem
