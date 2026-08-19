// hellofem::io — COMSOL .mphtxt mesh reader
// SPDX-License-Identifier: MIT

#pragma once

#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"

#include <filesystem>
#include <memory>

namespace hellofem::io {

    /// Mesh data read from a COMSOL `.mphtxt` file.
    struct MphtxtMesh {
        /// The mesh (affine P1 geometry, or isoparametric P2 for
        /// second-order elements).
        mesh::Mesh<double> mesh;
        /// Polynomial degree of the volume cells (1 or 2 for Lagrange).
        int order = 1;
        /// Tags on the boundary facets (dimension `tdim - 1`): each
        /// boundary element's geometric entity index from the file.
        /// COMSOL domain indices are 1-based and boundary entity indices
        /// 0-based; the raw file values are stored.
        std::shared_ptr<mesh::MeshTags<int>> facet_tags;
        /// Tags on the volume cells (dimension `tdim`): each volume
        /// element's geometric entity (domain) index from the file. COMSOL
        /// domain indices are already 1-based; raw file values are stored.
        std::shared_ptr<mesh::MeshTags<int>> cell_tags;
    };

    /// Read a COMSOL `.mphtxt` mesh file.
    ///
    /// Supports first- and second-order simplex cells (interval/triangle/
    /// tetrahedron) and quadrilateral/hexahedron; other cell shapes or
    /// mixed cell types are rejected. Boundary blocks (`vtx`/`edg`/`tri`/
    /// `quad` of co-dimension 1) produce the `facet_tags`.
    ///
    /// @param[in] filename Path to the `.mphtxt` file.
    MphtxtMesh read_mphtxt(const std::filesystem::path& filename);

} // namespace hellofem::io
