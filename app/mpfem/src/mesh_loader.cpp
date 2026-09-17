// hellofem::app — load a COMSOL .mphtxt mesh for the app layer
// SPDX-License-Identifier: MIT

#include "mesh_loader.h"

#include "io/mphtxt.h"

#include <set>
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

    LoadedMesh load_mphtxt_mesh(const std::filesystem::path& filename)
    {
        io::MphtxtMesh raw = io::read_mphtxt(filename);
        LoadedMesh out;
        out.order = raw.order;
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

} // namespace hellofem::app
