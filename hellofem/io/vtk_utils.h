// hellofem::io — VTK output of meshes and functions
// SPDX-License-Identifier: MIT

#pragma once

#include "io/cells.h"
#include "mesh/Mesh.h"
#include "mesh/cell_types.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace hellofem::fem {
    template <std::floating_point T>
    class FunctionSpace;
}

namespace hellofem::io {

    /// Cell connectivity in VTK node ordering from a geometry dofmap.
    ///
    /// @param[in] dofmap_x Geometry dofmap (num_cells x dofs_per_cell).
    /// @param[in] cell_type Cell shape.
    /// @return Connectivity in terms of geometry node indices, stored
    /// row-major with shape `(num_cells, nodes_per_cell)`.
    std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
    extract_vtk_connectivity(
        md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> dofmap_x,
        mesh::CellType cell_type);

    /// Node coordinates and cells (VTK ordering) for a (discontinuous)
    /// Lagrange function space.
    ///
    /// @pre `V` must be scalar (block size 1) and not mixed.
    ///
    /// @return (node coordinates (num_dofs x 3), coordinate shape, cells
    /// in VTK node ordering, cell shape).
    template <std::floating_point T>
    std::tuple<std::vector<T>, std::array<std::size_t, 2>,
        std::vector<std::int64_t>, std::array<std::size_t, 2>>
    vtk_mesh_from_space(const fem::FunctionSpace<T>& V);

    /// Named point-data array: `num_components` columns per point.
    struct PointData {
        std::string name; ///< Array name.
        std::vector<double> values; ///< Flattened `(num_points, num_components)`.
        int num_components = 1; ///< Number of columns.
    };

    /// Write an unstructured grid to a VTU file (ascii XML).
    ///
    /// @param[in] filename Output path.
    /// @param[in] x Node coordinates, flattened `(num_points, 3)`.
    /// @param[in] cells Cell connectivity in VTK node ordering, flattened
    /// with shape `cshape`.
    /// @param[in] cshape Shape of `cells`.
    /// @param[in] cell_type Cell shape.
    /// @param[in] point_data Named point-data arrays.
    void write_vtu(const std::filesystem::path& filename,
        std::span<const double> x, std::span<const std::int64_t> cells,
        std::array<std::size_t, 2> cshape, mesh::CellType cell_type,
        const std::vector<PointData>& point_data = {});

    /// Write a mesh (affine P1 geometry) to a VTU file.
    void write_vtu(const std::filesystem::path& filename,
        const mesh::Mesh<double>& mesh);

    // ------------------------------------------------------------------ //
    //                         Inline implementations                      //
    // ------------------------------------------------------------------ //

    template <std::floating_point T>
    std::tuple<std::vector<T>, std::array<std::size_t, 2>,
        std::vector<std::int64_t>, std::array<std::size_t, 2>>
    vtk_mesh_from_space(const fem::FunctionSpace<T>& V)
    {
        auto mesh = V.mesh();
        auto topology = mesh->topology();
        const int tdim = topology->dim();

        if (V.element()->is_mixed())
            throw std::runtime_error(
                "vtk_mesh_from_space: mixed elements are not supported.");
        if (V.dofmap()->bs() != 1)
            throw std::runtime_error(
                "vtk_mesh_from_space: only scalar (block size 1) spaces "
                "are supported.");

        // Node coordinates, one row per scalar dof.
        const int gdim = mesh->geometry().dim();
        std::vector<T> x = V.tabulate_dof_coordinates(false);
        const std::size_t num_nodes = x.size() / gdim;
        std::vector<T> x3(3 * num_nodes, 0);
        for (std::size_t i = 0; i < num_nodes; ++i)
            for (int j = 0; j < gdim; ++j)
                x3[3 * i + j] = x[gdim * i + j];
        const std::array<std::size_t, 2> xshape {num_nodes, 3};

        // Cell dofs re-ordered from the basix (dofmap) order to the VTK
        // node order.
        auto dofmap = V.dofmap();
        const std::uint32_t num_nodes_per_cell
            = V.element()->space_dimension();
        std::vector<std::uint16_t> vtkmap = cells::transpose(
            cells::perm_vtk(topology->cell_type(), num_nodes_per_cell));

        auto map = topology->index_map(tdim);
        const std::size_t num_cells = map->size_local();
        const std::array<std::size_t, 2> cshape {num_cells,
            num_nodes_per_cell};
        std::vector<std::int64_t> vtk_topology(num_cells * num_nodes_per_cell);
        for (std::size_t c = 0; c < num_cells; ++c) {
            auto dofs = dofmap->cell_dofs(static_cast<std::int32_t>(c));
            for (std::size_t i = 0; i < dofs.size(); ++i)
                vtk_topology[c * num_nodes_per_cell + i] = dofs[vtkmap[i]];
        }

        return {std::move(x3), xshape, std::move(vtk_topology), cshape};
    }

} // namespace hellofem::io
