// hellofem::io — VTK output tests
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "fem/DofMap.h"
#include "fem/FiniteElement.h"
#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "fem/dofmapbuilder.h"
#include "io/VTKFile.h"
#include "io/vtk_utils.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace hellofem;

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    std::shared_ptr<fem::FunctionSpace<double>> p1_space(int n)
    {
        auto mesh = mesh::create_unit_square(n);
        mesh->topology_mutable()->create_entities(1);
        mesh->topology_mutable()->create_connectivity(2, 1);
        mesh->topology_mutable()->create_connectivity(1, 2);
        auto fe = std::make_shared<fem::FiniteElement<double>>(
            basis::create_element<double>(
                B::P, C::triangle, 1, LV::equispaced, DV::unset, false));
        auto layout = fem::CoordinateElement<double>(
            mesh::CellType::triangle, 1, LV::equispaced)
                          .create_dof_layout();
        auto [imap, bs, dofmaps]
            = fem::build_dofmap_data(*mesh->topology(), {layout}, nullptr);
        auto dmap = std::make_shared<fem::DofMap>(layout,
            std::make_shared<common::IndexMap>(std::move(imap)), bs,
            std::move(dofmaps.front()), bs);
        return std::make_shared<fem::FunctionSpace<double>>(mesh, fe, dmap);
    }

    std::string read_file(const std::filesystem::path& path)
    {
        std::ifstream in(path);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    /// Count occurrences of a substring.
    std::size_t count(const std::string& s, const std::string& sub)
    {
        std::size_t n = 0, pos = 0;
        while ((pos = s.find(sub, pos)) != std::string::npos) {
            ++n;
            pos += sub.size();
        }
        return n;
    }

    /// Extract the integers of the connectivity DataArray from a VTU
    /// file (geometry-node indices are renumbered, so the exact values
    /// are not fixed; only their validity can be checked).
    std::vector<long long> connectivity_values(const std::string& s)
    {
        const std::string start
            = "type=\"Int64\" Name=\"connectivity\" format=\"ascii\">";
        const std::string end = "</DataArray>";
        const std::size_t b = s.find(start) + start.size();
        const std::size_t e = s.find(end, b);
        std::vector<long long> values;
        std::istringstream block(s.substr(b, e - b));
        long long v;
        while (block >> v)
            values.push_back(v);
        return values;
    }

} // namespace

TEST_CASE("write_vtu writes a unit square mesh", "[io][vtk]")
{
    const auto path = std::filesystem::temp_directory_path() / "hellofem_test.vtu";
    auto mesh = mesh::create_unit_square(4);
    io::write_vtu(path, *mesh);

    const std::string s = read_file(path);
    REQUIRE(s.find("<VTKFile type=\"UnstructuredGrid\"") != std::string::npos);
    REQUIRE(s.find("NumberOfPoints=\"25\"") != std::string::npos);
    REQUIRE(s.find("NumberOfCells=\"32\"") != std::string::npos);
    // VTK_TRIANGLE = 5 for every cell.
    REQUIRE(count(s, " 5") >= 32);

    // Every connectivity node index must reference a valid point.
    const auto values = connectivity_values(s);
    REQUIRE(values.size() == 32 * 3);
    for (long long v : values)
        REQUIRE(v >= 0);
    REQUIRE(*std::max_element(values.begin(), values.end()) < 25);
    // Each triangle references 3 distinct points.
    for (std::size_t c = 0; c < 32; ++c) {
        REQUIRE(values[3 * c] != values[3 * c + 1]);
        REQUIRE(values[3 * c] != values[3 * c + 2]);
        REQUIRE(values[3 * c + 1] != values[3 * c + 2]);
    }
}

TEST_CASE("write_vtu writes point data of a function", "[io][vtk]")
{
    const auto path
        = std::filesystem::temp_directory_path() / "hellofem_func.vtu";
    auto V = p1_space(2);
    fem::Function<double> f(V);
    f.name = "f";
    f.interpolate(
        [](std::span<const double> x, std::array<std::size_t, 2> shape) {
            std::vector<double> values(shape[0]);
            for (std::size_t i = 0; i < shape[0]; ++i)
                values[i] = x[shape[1] * i + 0] + 2 * x[shape[1] * i + 1];
            return std::pair(std::move(values),
                std::array<std::size_t, 2> {shape[0], 1});
        });

    std::vector<std::reference_wrapper<const fem::Function<double>>> u {f};
    const auto base = path;
    io::VTKFile file(base, "w");
    file.write(u, 0.0);

    // VTKFile derives the per-step name `<stem>_<i>.vtu`.
    const std::string s = read_file(
        base.parent_path() / (base.stem().string() + "_0.vtu"));
    REQUIRE(s.find("<PointData>") != std::string::npos);
    REQUIRE(s.find("Name=\"f\"") != std::string::npos);
    REQUIRE(s.find("RangeMin=\"0") != std::string::npos);
    // f(1,1) = 3 is the maximum on the unit square.
    REQUIRE(s.find("RangeMax=\"3") != std::string::npos);
}

TEST_CASE("VTKFile writes a time series (.pvd)", "[io][vtk]")
{
    const auto dir = std::filesystem::temp_directory_path();
    const auto pvd = dir / "hellofem_series.pvd";
    auto mesh = mesh::create_unit_square(2);

    io::VTKFile file(pvd, "w");
    file.write(*mesh, 0.0);
    file.write(*mesh, 0.5);
    file.close();

    const std::string s = read_file(pvd);
    REQUIRE(s.find("<VTKFile type=\"Collection\"") != std::string::npos);
    REQUIRE(count(s, "<DataSet timestep=") == 2);
    REQUIRE(s.find("timestep=\"0\"") != std::string::npos);
    REQUIRE(s.find("timestep=\"0.5\"") != std::string::npos);
    REQUIRE(s.find("hellofem_series_0.vtu") != std::string::npos);
    REQUIRE(s.find("hellofem_series_1.vtu") != std::string::npos);

    // The per-step files exist.
    REQUIRE(std::filesystem::exists(dir / "hellofem_series_0.vtu"));
    REQUIRE(std::filesystem::exists(dir / "hellofem_series_1.vtu"));
}
