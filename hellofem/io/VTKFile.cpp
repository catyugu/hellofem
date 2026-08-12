// hellofem::io — VTK/ParaView file output
// SPDX-License-Identifier: MIT

#include "VTKFile.h"

#include "fem/Function.h"
#include "fem/FunctionSpace.h"

#include <fstream>
#include <stdexcept>

using namespace hellofem;

namespace {

    /// Path of the `i`-th per-step file: `<dir>/<stem>_<i>.vtu`.
    std::filesystem::path step_path(const std::filesystem::path& directory,
        const std::string& stem, std::size_t i)
    {
        return directory / (stem + "_" + std::to_string(i) + ".vtu");
    }

} // namespace

io::VTKFile::VTKFile(const std::filesystem::path& filename,
    std::string_view file_mode)
    : _filename(filename)
    , _directory(filename.parent_path())
    , _stem(filename.stem().string())
{
    if (file_mode != "w" and file_mode != "a")
        throw std::invalid_argument("VTKFile: file_mode must be 'w' or 'a'.");
    if (file_mode == "w")
        _steps.clear();
}

io::VTKFile::~VTKFile() { close(); }

void io::VTKFile::close()
{
    std::ofstream out(_filename);
    if (!out)
        throw std::runtime_error("VTKFile: cannot open .pvd file.");
    out.precision(17);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"0.1\" "
           "byte_order=\"LittleEndian\">\n";
    out << "  <Collection>\n";
    for (const auto& [t, file] : _steps)
        out << "    <DataSet timestep=\"" << t << "\" file=\"" << file
            << "\"/>\n";
    out << "  </Collection>\n";
    out << "</VTKFile>\n";
}

void io::VTKFile::write(const mesh::Mesh<double>& mesh, double t)
{
    const std::filesystem::path path = step_path(_directory, _stem, _counter);
    io::write_vtu(path, mesh);
    _steps.emplace_back(t, path.filename().string());
    ++_counter;
}

void io::VTKFile::write(
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
    double t)
{
    if (u.empty())
        return;

    // The first function's space provides the mesh and the node ordering.
    auto V0 = u.front().get().function_space();
    if (V0->dofmap()->bs() != 1)
        throw std::runtime_error(
            "VTKFile: only scalar (block size 1) functions are supported.");

    auto [x, xshape, cells, cshape] = vtk_mesh_from_space(*V0);

    std::vector<PointData> point_data;
    point_data.reserve(u.size());
    for (const auto& fu : u) {
        auto V = fu.get().function_space();
        if (V != V0)
            throw std::runtime_error(
                "VTKFile: all functions must share the same function space.");
        std::span<const double> values = fu.get().x()->array();
        if (values.size() != xshape[0])
            throw std::runtime_error(
                "VTKFile: function value count does not match the mesh nodes.");
        point_data.push_back(PointData {fu.get().name,
            std::vector<double>(values.begin(), values.end()), 1});
    }

    const std::filesystem::path path = step_path(_directory, _stem, _counter);
    io::write_vtu(path, x, cells, cshape, V0->mesh()->topology()->cell_type(),
        point_data);
    _steps.emplace_back(t, path.filename().string());
    ++_counter;
}
