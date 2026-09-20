// hellofem::app — load a COMSOL .mphtxt mesh for the app layer
// SPDX-License-Identifier: MIT

#include "mesh_loader.h"

#include "io/mphtxt.h"
#include "units.h"

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
        const std::filesystem::path& filename, std::string_view length_unit)
    {
        io::MphtxtMesh raw = io::read_mphtxt(filename);
        LoadedMesh out;
        // The file's coordinates are in the model's geometry length unit;
        // the app works in SI.
        for (double& x : raw.mesh.geometry().x())
            x = to_si(x, length_unit);
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
        prepare_topology(*out.mesh);
        return out;
    }

    void prepare_topology(const mesh::Mesh<double>& mesh)
    {
        const int tdim = mesh.topology()->dim();
        if (mesh.topology()->connectivity(tdim - 1, tdim))
            return; // the facets are there, so the mesh is prepared
        auto topo = mesh.topology_mutable();
        // The facets and the cells around them, and the cells' vertices:
        // creating either creates the entities the other reads.
        topo->create_connectivity(tdim - 1, tdim);
        topo->create_connectivity(tdim, 0);
    }

} // namespace hellofem::app
