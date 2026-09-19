// hellofem::app — load a COMSOL .mphtxt mesh for the app layer
// SPDX-License-Identifier: MIT
#pragma once

#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"

#include <filesystem>
#include <memory>
#include <set>

namespace hellofem::app {

    /// Loaded mesh with normalized domain/boundary ids for the app layer.
    struct LoadedMesh {
        std::shared_ptr<mesh::Mesh<double>> mesh;
        /// Polynomial degree of the volume cells (1 or 2).
        int order = 1;
        /// Domain id (1-based COMSOL) per cell, by cell index.
        std::shared_ptr<mesh::MeshTags<int>> cell_tags;
        /// Boundary id (1-based COMSOL) per boundary facet, by facet index.
        std::shared_ptr<mesh::MeshTags<int>> facet_tags;
        /// Number of distinct domains.
        int num_domains = 0;
        /// Number of distinct boundaries.
        int num_boundaries = 0;
    };

    /// Read a COMSOL .mphtxt mesh and normalize it for the app layer:
    /// boundary (facet) entity indices are 0-based in the file and become
    /// 1-based (matching COMSOL boundary selection numbers); domain ids are
    /// already 1-based and pass through unchanged.
    ///
    /// The coordinates are scaled by `length_scale` — the model's geometry
    /// length unit in SI metres — because COMSOL exports the mesh in that
    /// unit while every material law and feature is stated in SI. The loaded
    /// mesh is therefore always in metres.
    LoadedMesh load_mphtxt_mesh(
        const std::filesystem::path& filename, double length_scale = 1.0);

} // namespace hellofem::app
