// hellofem::io — VTK output of meshes and functions
// SPDX-License-Identifier: MIT

#include "vtk_utils.h"

#include "mesh/Topology.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <ostream>
#include <stdexcept>

using namespace hellofem;

namespace {

    /// VTK cell type identifier for a cell with `nodes_per_cell` nodes:
    /// linear types for the base cells, arbitrary-order Lagrange types
    /// (68-74) otherwise.
    std::int8_t vtk_cell_type(mesh::CellType cell_type,
        std::size_t nodes_per_cell)
    {
        const std::size_t linear = mesh::cell_num_entities(cell_type, 0);
        if (nodes_per_cell == linear)
            return io::cells::get_vtk_cell_type(cell_type);
        switch (cell_type) {
        case mesh::CellType::interval:
            return 68;
        case mesh::CellType::triangle:
            return 69;
        case mesh::CellType::quadrilateral:
            return 70;
        case mesh::CellType::tetrahedron:
            return 71;
        case mesh::CellType::hexahedron:
            return 72;
        default:
            throw std::runtime_error(
                "write_vtu: unsupported higher-order cell type.");
        }
    }

    /// Write a single ascii DataArray node (with range attributes).
    void write_data_array(std::ostream& out, std::string_view name,
        std::string_view type, std::span<const double> values,
        int num_components, int indent)
    {
        const std::string pad(indent, ' ');
        auto [min, max] = std::ranges::minmax_element(values);
        out << pad << "<DataArray type=\"" << type << "\" Name=\"" << name
            << "\" NumberOfComponents=\"" << num_components
            << "\" format=\"ascii\" RangeMin=\"" << *min << "\" RangeMax=\""
            << *max << "\">\n";
        out << pad << "  ";
        for (std::size_t i = 0; i < values.size(); ++i)
            out << values[i] << (i + 1 < values.size() ? " " : "\n");
        out << pad << "</DataArray>\n";
    }

} // namespace

std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
io::extract_vtk_connectivity(
    md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> dofmap_x,
    mesh::CellType cell_type)
{
    const std::size_t num_nodes = dofmap_x.extent(1);
    const std::vector<std::uint16_t> vtkmap
        = cells::transpose(cells::perm_vtk(cell_type, num_nodes));

    const std::size_t num_cells = dofmap_x.extent(0);
    std::vector<std::int64_t> cells(num_cells * num_nodes);
    for (std::size_t c = 0; c < num_cells; ++c)
        for (std::size_t i = 0; i < num_nodes; ++i)
            cells[c * num_nodes + i] = dofmap_x(c, vtkmap[i]);

    return {std::move(cells), {num_cells, num_nodes}};
}

void io::write_vtu(const std::filesystem::path& filename,
    std::span<const double> x, std::span<const std::int64_t> cells,
    std::array<std::size_t, 2> cshape, mesh::CellType cell_type,
    const std::vector<PointData>& point_data)
{
    const std::size_t num_points = x.size() / 3;
    const std::size_t num_cells = cshape[0];
    const std::size_t nodes_per_cell = cshape[1];

    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("write_vtu: cannot open file for writing.");
    out.precision(17);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
           "byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\""
        << num_cells << "\">\n";

    // Points.
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" Name=\"Points\" "
           "NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < num_points; ++i)
        out << "          " << x[3 * i] << " " << x[3 * i + 1] << " "
            << x[3 * i + 2] << "\n";
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Cells: connectivity, offsets and types.
    const std::int8_t type = vtk_cell_type(cell_type, nodes_per_cell);
    out << "      <Cells>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" "
           "format=\"ascii\">\n";
    for (std::size_t c = 0; c < num_cells; ++c) {
        out << "          ";
        for (std::size_t i = 0; i < nodes_per_cell; ++i)
            out << cells[c * nodes_per_cell + i]
                << (i + 1 < nodes_per_cell ? " " : "\n");
    }
    out << "        </DataArray>\n";
    out << "        <DataArray type=\"Int64\" Name=\"offsets\" "
           "format=\"ascii\">\n";
    out << "          ";
    for (std::size_t c = 0; c < num_cells; ++c)
        out << (c + 1) * nodes_per_cell
            << (c + 1 < num_cells ? " " : "\n");
    out << "        </DataArray>\n";
    out << "        <DataArray type=\"UInt8\" Name=\"types\" "
           "format=\"ascii\">\n";
    out << "          ";
    for (std::size_t c = 0; c < num_cells; ++c)
        out << static_cast<int>(type) << (c + 1 < num_cells ? " " : "\n");
    out << "        </DataArray>\n";
    out << "      </Cells>\n";

    // Point data.
    if (!point_data.empty()) {
        out << "      <PointData>\n";
        for (const PointData& pd : point_data)
            write_data_array(out, pd.name, "Float64", pd.values,
                pd.num_components, 8);
        out << "      </PointData>\n";
    }

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";
}

void io::write_vtu(const std::filesystem::path& filename,
    const mesh::Mesh<double>& mesh)
{
    auto topology = mesh.topology();
    auto [cells, cshape] = extract_vtk_connectivity(
        mesh.geometry().dofmaps().front(), topology->cell_type());
    write_vtu(filename, mesh.geometry().x(), cells, cshape,
        topology->cell_type());
}
