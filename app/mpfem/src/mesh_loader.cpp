// hellofem::app — load a COMSOL .mphtxt mesh for the app layer
// SPDX-License-Identifier: MIT

#include "mesh_loader.h"

#include "io/mphtxt.h"

#include <algorithm>

namespace hellofem::app {

LoadedMesh load_mphtxt_mesh(const std::filesystem::path& filename)
{
    io::MphtxtMesh raw = io::read_mphtxt(filename);
    LoadedMesh out;
    out.mesh = std::make_shared<mesh::Mesh<double>>(
        raw.mesh.topology(), raw.mesh.geometry());

    // Domain ids: already 1-based, pass through.
    out.cell_tags = raw.cell_tags;
    if (out.cell_tags) {
        int mn = 1000000, mx = 0;
        for (int v : out.cell_tags->values()) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        out.num_domains = mx;
    }

    // Boundary ids: file is 0-based -> +1 to match COMSOL selections.
    out.facet_tags = raw.facet_tags;
    if (out.facet_tags) {
        // Rebuild MeshTags with +1'd values (MeshTags is immutable).
        const auto idx = out.facet_tags->indices();
        const auto val = out.facet_tags->values();
        std::vector<std::int32_t> indices(idx.begin(), idx.end());
        std::vector<int> values;
        values.reserve(val.size());
        int mn = 1000000, mx = 0;
        for (int v : val) {
            const int b = v + 1;
            values.push_back(b);
            mn = std::min(mn, b);
            mx = std::max(mx, b);
        }
        out.facet_tags = std::make_shared<mesh::MeshTags<int>>(
            out.mesh->topology(), out.facet_tags->dim(), std::move(indices),
            std::move(values), "facet_tags");
        out.num_boundaries = mx;
    }
    return out;
}

} // namespace hellofem::app
