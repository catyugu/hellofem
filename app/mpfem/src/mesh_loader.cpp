// hellofem::app — load a COMSOL .mphtxt mesh for the app layer
// SPDX-License-Identifier: MIT

#include "mesh_loader.h"

#include "io/mphtxt.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

namespace hellofem::app {
    namespace {

        /// The number of distinct ids the tags carry (COMSOL numbers a
        /// component's domains and boundaries from one, but a geometry
        /// operation may leave gaps, so the count is not the largest id).
        int tag_count(const std::shared_ptr<mesh::MeshTags<int>>& tags)
        {
            if (not tags)
                return 0;
            const auto values = tags->values();
            return static_cast<int>(
                std::set<int>(values.begin(), values.end()).size());
        }

    } // namespace

    LoadedMesh load_mphtxt_mesh(
        const std::filesystem::path& filename, double length_scale)
    {
        io::MphtxtMesh raw = io::read_mphtxt(filename);
        LoadedMesh out;
        out.order = raw.order;
        // The file's coordinates are in the model's geometry length unit;
        // the app works in SI.
        if (length_scale != 1.0)
            for (double& x : raw.mesh.geometry().x())
                x *= length_scale;
        out.mesh = std::make_shared<mesh::Mesh<double>>(
            raw.mesh.topology(), raw.mesh.geometry());

        // Domain ids: already 1-based, pass through.
        out.cell_tags = raw.cell_tags;
        out.num_domains = tag_count(out.cell_tags);

        // Boundary ids: file is 0-based -> +1 to match COMSOL selections.
        // MeshTags is immutable, so the normalized set is rebuilt.
        if (raw.facet_tags) {
            const auto idx = raw.facet_tags->indices();
            const auto val = raw.facet_tags->values();
            std::vector<std::int32_t> indices(idx.begin(), idx.end());
            std::vector<int> values;
            values.reserve(val.size());
            for (int v : val)
                values.push_back(v + 1);
            out.facet_tags = std::make_shared<mesh::MeshTags<int>>(
                out.mesh->topology(), raw.facet_tags->dim(), std::move(indices),
                std::move(values), "facet_tags");
            out.num_boundaries = tag_count(out.facet_tags);
        }
        return out;
    }

    std::set<int> boundary_domains(const LoadedMesh& mesh, int boundary)
    {
        return boundary_domains(mesh, std::set<int> {boundary});
    }

    std::set<int> boundary_domains(
        const LoadedMesh& mesh, const std::set<int>& boundaries)
    {
        std::set<int> domains;
        if (not mesh.facet_tags or not mesh.cell_tags)
            return domains;
        const int tdim = mesh.mesh->topology()->dim();
        auto topo = mesh.mesh->topology_mutable();
        topo->create_entities(tdim - 1);
        topo->create_connectivity(tdim - 1, tdim);
        auto f_to_c = topo->connectivity(tdim - 1, tdim);

        // Cell index -> domain id: the tags hold a (sorted) index and its
        // value per tagged cell, so the lookup is a binary search.
        const auto cell_indices = mesh.cell_tags->indices();
        const auto cell_values = mesh.cell_tags->values();
        auto domain_of = [&](std::int32_t cell) {
            const auto it = std::lower_bound(
                cell_indices.begin(), cell_indices.end(), cell);
            if (it == cell_indices.end() or *it != cell)
                throw std::runtime_error(
                    "boundary_domains: a cell carries no domain tag");
            return cell_values[static_cast<std::size_t>(it - cell_indices.begin())];
        };

        const auto& indices = mesh.facet_tags->indices();
        const auto& values = mesh.facet_tags->values();
        for (std::size_t i = 0; i < indices.size(); ++i)
            if (boundaries.contains(values[i]))
                for (std::int32_t c : f_to_c->links(indices[i]))
                    domains.insert(domain_of(c));
        return domains;
    }

} // namespace hellofem::app
