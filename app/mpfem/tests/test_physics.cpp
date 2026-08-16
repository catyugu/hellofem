// hellofem::app — physics solver tests: electrostatics manufactured solution
// SPDX-License-Identifier: MIT

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "mesh/generation.h"
#include "mesh/utils.h"
#include "physics.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>

using hellofem::app::CellProperty;
using hellofem::app::ElectrostaticsSolver;
using hellofem::app::HeatTransferSolver;
using hellofem::app::SolidMechanicsSolver;
using Catch::Approx;

namespace {

    /// 1-based boundary id for a 3D box face by constant coordinate:
    /// 1=x-,2=x+,3=y-,4=y+,5=z-,6=z+.
    std::shared_ptr<hellofem::mesh::MeshTags<int>> make_boundary_tags(
        const hellofem::mesh::Mesh<double>& mesh)
    {
        auto topo = mesh.topology();
        auto c_to_v = topo->connectivity(3, 0);
        const auto [vc, shape] = hellofem::mesh::compute_vertex_coords(mesh);
        const std::size_t nv = shape[1];
        auto vert = [&](std::int32_t v, int d) { return vc[d * nv + v]; };

        auto topo_mut = mesh.topology_mutable();
        topo_mut->create_entities(2);
        topo_mut->create_connectivity(2, 3);
        auto f_to_c = topo_mut->connectivity(2, 3);
        auto f_to_v = topo_mut->connectivity(2, 0);
        std::vector<std::int32_t> idx;
        std::vector<int> vals;
        for (std::int32_t f = 0; f < f_to_v->num_nodes(); ++f) {
            if (f_to_c->num_links(f) != 1)
                continue; // interior
            auto vs = f_to_v->links(f);
            double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9, zmin = 1e9, zmax = -1e9;
            for (auto v : vs) {
                xmin = std::min(xmin, vert(v, 0)); xmax = std::max(xmax, vert(v, 0));
                ymin = std::min(ymin, vert(v, 1)); ymax = std::max(ymax, vert(v, 1));
                zmin = std::min(zmin, vert(v, 2)); zmax = std::max(zmax, vert(v, 2));
            }
            int id = 0;
            if (xmin == xmax) id = (xmin < 0.5) ? 1 : 2;
            else if (ymin == ymax) id = (ymin < 0.5) ? 3 : 4;
            else id = (zmin < 0.5) ? 5 : 6;
            idx.push_back(f);
            vals.push_back(id);
        }
        std::vector<std::size_t> order(idx.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return idx[a] < idx[b]; });
        std::vector<std::int32_t> sidx;
        std::vector<int> svals;
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (i == 0 or idx[order[i]] != idx[order[i - 1]]) {
                sidx.push_back(idx[order[i]]);
                svals.push_back(vals[order[i]]);
            }
        }
        return std::make_shared<hellofem::mesh::MeshTags<int>>(topo_mut, 2,
            std::move(sidx), std::move(svals), "boundary");
    }

    /// Build a 3D single-layer box mesh with the standard 6-face boundary
    /// tags and all cells on domain 1.
    struct Fixture {
        std::shared_ptr<hellofem::mesh::Mesh<double>> mesh;
        std::shared_ptr<hellofem::mesh::MeshTags<int>> boundary;
        std::shared_ptr<hellofem::mesh::MeshTags<int>> cells;
    };

    Fixture make_box_fixture(std::array<double, 3> lo, std::array<double, 3> hi,
        std::array<int, 3> n)
    {
        Fixture f;
        f.mesh = hellofem::mesh::create_box(lo, hi, n);
        f.boundary = make_boundary_tags(*f.mesh);
        const std::size_t nc = f.mesh->topology()->index_map(3)->size_local();
        std::vector<std::int32_t> cell_idx(nc);
        std::iota(cell_idx.begin(), cell_idx.end(), 0);
        f.cells = std::make_shared<hellofem::mesh::MeshTags<int>>(
            f.mesh->topology(), 3, std::move(cell_idx),
            std::vector<int>(nc, 1), "cells");
        return f;
    }

} // namespace

TEST_CASE("Electrostatics: -div(sigma grad V)=0 with V=V0 on x+, V=0 on x-", "[app][physics]")
{
    // 1x1x1 box, single layer in z. V solves Laplace; with V=1 on x+, 0 on x-,
    // the solution is V = x (linear), sigma = 1.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 4, 1});
    auto mesh = f.mesh;
    auto tags = f.boundary;
    auto cell_tags = f.cells;

    ElectrostaticsSolver es(mesh, tags, cell_tags, 1);
    auto sigma = std::make_shared<CellProperty>(mesh, cell_tags);
    sigma->set_domain(1, 1.0);
    es.set_conductivity(sigma);
    es.add_voltage_bc(2, 1.0);  // x+ : V=1
    es.add_voltage_bc(1, 0.0);  // x- : V=0

    es.solve_steady();
    auto V = es.solution();

    // V = x at every dof coordinate.
    auto coords = es.space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    double v_max = -1e9, v_min = 1e9;
    for (std::int32_t d = 0; d < es.space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        const double exact = x;
        const double val = V->x()->array()[static_cast<std::size_t>(d)];
        max_err = std::max(max_err, std::abs(val - exact));
        v_max = std::max(v_max, val);
        v_min = std::min(v_min, val);
    }
    INFO("max error = " << max_err << ", V range [" << v_min << "," << v_max << "]");
    REQUIRE(max_err < 1e-10);
    REQUIRE(v_max == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(v_min == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("HeatTransfer: steady -div(k grad T)=0 with Robin convection", "[app][physics]")
{
    // 1D rod along x in a 1x1x1 box (single y,z layer): -d/dx(k dT/dx)=0,
    // T=1 on x-, Robin h(T-Tinf)=h*T on x+ (Tinf=0). With k=1, h=1:
    // T = a + b·x, T(0)=1 => a=1; Robin: -k dT/dx = h·T(1) => -b = 1+b
    // => b = -1/2. Hence T(x) = 1 - x/2, T(1)=0.5.
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {4, 4, 1});
    HeatTransferSolver ht(f.mesh, f.boundary, f.cells, 1);
    auto k = std::make_shared<CellProperty>(f.mesh, f.cells);
    k->set_domain(1, 1.0);
    ht.set_conductivity(k);
    ht.add_temperature_bc(1, 1.0); // x- : T=1
    ht.add_convection(2, 1.0, 0.0); // x+ : h=1, Tinf=0

    ht.solve_steady();
    auto T = ht.solution();

    auto coords = ht.space()->tabulate_dof_coordinates(false);
    double max_err = 0;
    for (std::int32_t d = 0; d < ht.space()->dofmap()->index_map->size_local(); ++d) {
        const double x = coords[3 * d];
        const double exact = 1.0 - 0.5 * x;
        max_err = std::max(max_err,
            std::abs(T->x()->array()[static_cast<std::size_t>(d)] - exact));
    }
    INFO("heat max error = " << max_err);
    REQUIRE(max_err < 1e-8);
}

TEST_CASE("SolidMechanics: blocked assembly — no load with Fixed gives u=0", "[app][physics]")
{
    auto f = make_box_fixture({0, 0, 0}, {1, 1, 1}, {2, 2, 1});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1);
    auto E = std::make_shared<CellProperty>(f.mesh, f.cells);
    E->set_domain(1, 200e9);
    auto nu = std::make_shared<CellProperty>(f.mesh, f.cells);
    nu->set_domain(1, 0.3);
    sm.set_elastic(E, nu);
    sm.add_fixed_bc(1); // x- face: u=v=w=0

    sm.solve_steady();
    auto u = sm.solution();
    double max_mag = 0;
    for (double v : u->x()->array())
        max_mag = std::max(max_mag, std::abs(v));
    INFO("no-load solid max |u| = " << max_mag);
    REQUIRE(max_mag < 1e-10);
}

TEST_CASE("SolidMechanics: uniform thermal expansion of a clamped bar", "[app][physics]")
{
    // Slender bar along x: uniform DT=100, alpha=1e-5 -> free expansion would
    // be u_x = alpha*DT*x = 1e-3 at x=1. Fixed (u=v=w=0) on the x- face adds a
    // lateral-constraint boundary layer, so u_x(1) stays near alpha*DT*L and
    // the displacement is smooth and monotone.
    auto f = make_box_fixture({0, 0, 0}, {1, 0.3, 0.3}, {10, 3, 3});
    SolidMechanicsSolver sm(f.mesh, f.boundary, f.cells, 1);
    auto E = std::make_shared<CellProperty>(f.mesh, f.cells);
    E->set_domain(1, 200e9);
    auto nu = std::make_shared<CellProperty>(f.mesh, f.cells);
    nu->set_domain(1, 0.3);
    auto alpha = std::make_shared<CellProperty>(f.mesh, f.cells);
    alpha->set_domain(1, 1e-5);
    auto T = std::make_shared<CellProperty>(f.mesh, f.cells);
    T->set_domain(1, 393.15); // DT = 100 above Tref=293.15
    sm.set_elastic(E, nu);
    sm.set_thermal_expansion(T->function(), alpha, 293.15);
    sm.add_fixed_bc(1); // x- face clamped

    sm.solve_steady();
    auto u = sm.solution();
    const auto& xa = u->x()->array();

    // Blocked space: dof coordinates are per physical dof (bs*block + comp),
    // so vertex d lives at dofcoords[(3*d)*gdim + q] = coords[9*d+q], and the
    // solution component c of vertex d is xa[3*d+c].
    auto coords = sm.space()->tabulate_dof_coordinates(false);
    double ux_max = 0, ux_at_1 = -1e9;
    double max_lat = 0;
    for (std::int32_t d = 0; d < sm.space()->dofmap()->index_map->size_local(); ++d) {
        const std::size_t cd = static_cast<std::size_t>(9 * d);
        const double x = coords[cd];
        const double ux = xa[static_cast<std::size_t>(3 * d)];
        const double uy = xa[static_cast<std::size_t>(3 * d + 1)];
        const double uz = xa[static_cast<std::size_t>(3 * d + 2)];
        ux_max = std::max(ux_max, ux);
        if (std::abs(x - 1.0) < 1e-9)
            ux_at_1 = std::max(ux_at_1, ux);
        max_lat = std::max(max_lat, std::max(std::abs(uy), std::abs(uz)));
    }
    INFO("thermal bar ux_max = " << ux_max << ", ux(x=1)=" << ux_at_1
        << ", max lateral |u| = " << max_lat);
    REQUIRE(ux_max > 0.9e-3);
    REQUIRE(ux_max < 1.1e-3);
    REQUIRE(ux_at_1 > 0.9e-3); // end face essentially alpha*DT*L
    REQUIRE(max_lat < 2e-3);   // bounded lateral deformation
}
