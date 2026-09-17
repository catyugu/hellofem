// hellofem::fem — shared assembly utilities
// SPDX-License-Identifier: MIT

#include "utils.h"

#include "DofMap.h"
#include "common/IndexMap.h"
#include "dofmapbuilder.h"
#include "graph/AdjacencyList.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace hellofem::fem {

    namespace {

        /// Local index of entity `e` within cell `c`: the position of `e`
        /// in the cell's `c_to_e` list.
        int local_entity_index(const graph::AdjacencyList<std::int32_t>& c_to_e,
            std::int32_t c, std::int32_t e)
        {
            auto cell_entities = c_to_e.links(c);
            auto it = std::find(cell_entities.begin(), cell_entities.end(), e);
            return static_cast<int>(std::distance(cell_entities.begin(), it));
        }

    } // namespace

    std::vector<std::int32_t> exterior_facet_entities(const mesh::Topology& topology)
    {
        const int tdim = topology.dim();
        auto* mutable_topo = const_cast<mesh::Topology*>(&topology);
        mutable_topo->create_entities(tdim - 1);
        mutable_topo->create_connectivity(tdim - 1, tdim);
        mutable_topo->create_connectivity(tdim, tdim - 1);

        const std::vector<std::int32_t> facets
            = mesh::exterior_facet_indices(topology);
        auto e_to_c = topology.connectivity(tdim - 1, tdim);
        auto c_to_e = topology.connectivity(tdim, tdim - 1);

        std::vector<std::int32_t> entities;
        entities.reserve(2 * facets.size());
        for (std::int32_t f : facets) {
            auto cells = e_to_c->links(f);
            assert(!cells.empty());
            const std::int32_t c = cells.front();
            entities.push_back(c);
            entities.push_back(local_entity_index(*c_to_e, c, f));
        }
        return entities;
    }

    std::vector<std::int32_t> interior_facet_entities(const mesh::Topology& topology)
    {
        const int tdim = topology.dim();
        auto* mutable_topo = const_cast<mesh::Topology*>(&topology);
        mutable_topo->create_entities(tdim - 1);
        mutable_topo->create_connectivity(tdim - 1, tdim);
        mutable_topo->create_connectivity(tdim, tdim - 1);

        auto e_to_c = topology.connectivity(tdim - 1, tdim);
        auto c_to_e = topology.connectivity(tdim, tdim - 1);
        const std::int32_t num_facets = e_to_c->num_nodes();

        std::vector<std::int32_t> entities;
        for (std::int32_t f = 0; f < num_facets; ++f) {
            auto cells = e_to_c->links(f);
            if (cells.size() != 2)
                continue;
            entities.push_back(cells[0]);
            entities.push_back(local_entity_index(*c_to_e, cells[0], f));
            entities.push_back(cells[1]);
            entities.push_back(local_entity_index(*c_to_e, cells[1], f));
        }
        return entities;
    }

    DofMap create_dofmap(const mesh::Mesh<double>& mesh,
        const ElementDofLayout& layout, const DofPermutation& permute_inv,
        const DofmapReordering& reorder_fn)
    {
        auto topology = mesh.topology_mutable();
        const int tdim = topology->dim();

        // The entities a dof lives on have to exist before the dofmap can
        // number them.
        const auto& entity_dofs = layout.entity_dofs_all();
        for (int d = 1; d < tdim; ++d) {
            const int dofs_on_d = std::accumulate(entity_dofs[d].begin(),
                entity_dofs[d].end(), 0,
                [](int count, const auto& e) {
                    return count + static_cast<int>(e.size());
                });
            if (dofs_on_d > 0)
                topology->create_entities(d);
        }

        auto [index_map, bs, dofmaps]
            = build_dofmap_data(*topology, {layout}, reorder_fn);

        // An entity shared by two cells is traversed in opposite directions,
        // so the dofs along it are numbered differently in each cell; the
        // permutation puts them back into the element's reference order.
        if (permute_inv) {
            const std::int32_t num_cells
                = topology->connectivity(tdim, 0)->num_nodes();
            topology->create_entity_permutations();
            const std::vector<std::uint32_t>& cell_info
                = topology->get_cell_permutation_info();
            const int dim = layout.num_dofs();
            for (std::int32_t cell = 0; cell < num_cells; ++cell) {
                std::span dofs(
                    dofmaps.front().data() + cell * dim, dim);
                permute_inv(dofs, cell_info[cell]);
            }
        }

        return DofMap(layout,
            std::make_shared<common::IndexMap>(std::move(index_map)), bs,
            std::move(dofmaps.front()), bs);
    }

    std::shared_ptr<FunctionSpace<double>> create_functionspace(
        std::shared_ptr<const mesh::Mesh<double>> mesh,
        std::shared_ptr<const FiniteElement<double>> element,
        const DofmapReordering& reorder_fn)
    {
        DofPermutation permute_inv;
        if (element->needs_dof_permutations())
            permute_inv = element->dof_permutation_fn(/*inverse=*/false);
        // The dofmap is built from the element before either argument is
        // moved into the space: the order in which a call's arguments are
        // evaluated is unspecified, so `*mesh` and `element` must not be
        // read in the same call that moves them.
        auto dofmap = std::make_shared<const DofMap>(create_dofmap(*mesh,
            element->create_dof_layout(), permute_inv, reorder_fn));
        return std::make_shared<FunctionSpace<double>>(std::move(mesh),
            std::move(element), std::move(dofmap));
    }

} // namespace hellofem::fem
