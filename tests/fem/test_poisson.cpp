// hellofem::fem — Poisson solve end-to-end test
// SPDX-License-Identifier: MIT

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/IndexMap.h"
#include "fem/CoordinateElement.h"
#include "fem/DirichletBC.h"
#include "fem/DofMap.h"
#include "fem/FiniteElement.h"
#include "fem/Form.h"
#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "fem/assembler.h"
#include "fem/dofmapbuilder.h"
#include "fem/precompute.h"
#include "fem/sparsitybuild.h"
#include "la/KrylovSolver.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

using namespace hellofem;
using Catch::Approx;

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    /// Build a P1 FunctionSpace on an n-interval unit square with
    /// boundary facets created.
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

    /// All cell indices of a dofmap as a vector.
    std::vector<std::int32_t> all_cells(const fem::DofMap& dmap)
    {
        std::vector<std::int32_t> cells(dmap.map().extent(0));
        for (std::size_t c = 0; c < cells.size(); ++c)
            cells[c] = static_cast<std::int32_t>(c);
        return cells;
    }

    /// Poisson stiffness kernel.
    void stiffness_kernel(double* Ae, const fem::CellKernelData<double>& data)
    {
        const int nq = data.num_points;
        const int nd = data.num_dofs0;
        const int tdim = data.tdim;
        std::fill(Ae, Ae + nd * nd, 0);
        for (int q = 0; q < nq; ++q)
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j) {
                    double dot = 0;
                    for (int c = 0; c < tdim; ++c)
                        dot += data.dphi0[(q * nd + i) * tdim + c]
                            * data.dphi1[(q * nd + j) * tdim + c];
                    Ae[i * nd + j] += data.w[q] * data.detJ[q] * dot;
                }
    }

    /// Poisson load kernel: Ae[i] = sum_q w_q detJ phi_i f(q).
    void load_kernel(double* Ae, const fem::CellKernelData<double>& data)
    {
        const int nq = data.num_points;
        const int nd = data.num_dofs0;
        std::fill(Ae, Ae + nd, 0);
        for (int q = 0; q < nq; ++q)
            for (int i = 0; i < nd; ++i)
                Ae[i] += data.w[q] * data.detJ[q] * data.phi0[q * nd + i]
                    * data.coeffs[q];
    }

} // namespace

TEST_CASE("Poisson: -Delta u = f with homogeneous Dirichlet", "[fem]")
{
    // Manufactured solution u = sin(pi x) sin(pi y), so f = 2 pi^2 u,
    // with u = 0 on the boundary. Solve and compare dof values to the
    // exact solution at the interior nodes.
    for (int n : {4, 8}) {
        auto V = p1_space(n);
        auto mesh = V->mesh();
        auto coord = mesh->geometry().cmaps().front();
        auto cells = all_cells(*V->dofmap());

        // Stiffness matrix.
        fem::PrecomputeData<double> pre_stiff(mesh::CellType::triangle,
            *V->element(), *V->element(), {}, coord, 2);
        fem::Form<double>::integral_data stiff;
        stiff.kernel = fem::make_cell_kernel(pre_stiff, stiffness_kernel);
        stiff.entities = cells;
        stiff.coeffs = {};
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist_b {V, V};
        std::vector<std::shared_ptr<const fem::FunctionSpace<double>>> Vlist_l {V};
        std::map<std::pair<fem::IntegralType, int>,
            std::vector<fem::Form<double>::integral_data>>
            integrals;
        integrals[{fem::IntegralType::cell, 0}] = {stiff};
        fem::Form<double> a(Vlist_b, std::move(integrals), mesh, {}, {});

        // Boundary condition: u = 0 on all boundary facets.
        auto e_to_c = V->mesh()->topology()->connectivity(1, 2);
        std::vector<std::int32_t> boundary_edges;
        for (std::int32_t e = 0; e < e_to_c->num_nodes(); ++e)
            if (e_to_c->num_links(e) == 1)
                boundary_edges.push_back(e);
        auto bc_dofs = fem::DirichletBC<double>::locate_dofs_topological(
            *V->mesh()->topology(), *V->dofmap(), 1, boundary_edges);
        fem::DirichletBC<double> bc(0.0, bc_dofs, V);

        // Sparsity pattern and matrix (zero BC rows/cols during assembly).
        la::SparsityPattern pattern(V->dofmap()->index_map, 1);
        fem::sparsitybuild::cells(pattern, std::pair {cells, cells},
            {*V->dofmap(), *V->dofmap()});
        std::vector<std::int32_t> diag(V->dofmap()->index_map->size_local());
        for (std::int32_t d = 0; d < V->dofmap()->index_map->size_local(); ++d)
            diag[static_cast<std::size_t>(d)] = d;
        pattern.insert_diagonal(std::span(diag));
        pattern.finalize();
        la::MatrixCSR<double> A(pattern);
        fem::assemble_matrix(A.mat_add_values(), a, {std::cref(bc)});
        fem::set_diagonal(A.mat_set_values(), a, {std::cref(bc)}, 1.0);

        // Right-hand side coefficient f.
        fem::Function<double> f(V);
        f.interpolate(
            [](std::span<const double> X, std::array<std::size_t, 2> shape) {
                const std::size_t n = shape[0];
                const double pi = 3.14159265358979323846;
                std::vector<double> fv(n);
                for (std::size_t i = 0; i < n; ++i)
                    fv[i] = 2 * pi * pi * std::sin(pi * X[2 * i])
                        * std::sin(pi * X[2 * i + 1]);
                return std::make_pair(std::move(fv),
                    std::array<std::size_t, 2> {n, 1});
            });

        // Linear form L(v) = integral f v.
        fem::PrecomputeData<double> pre_load(mesh::CellType::triangle,
            *V->element(), *V->element(), {V->element().get()}, coord, 2);
        fem::Form<double>::integral_data load;
        load.kernel = fem::make_cell_kernel(pre_load, load_kernel);
        load.entities = cells;
        load.coeffs = {0};
        integrals.clear();
        integrals[{fem::IntegralType::cell, 0}] = {load};
        std::vector<std::shared_ptr<const fem::Function<double>>> coeffs {std::make_shared<fem::Function<double>>(f)};
        fem::Form<double> L(Vlist_l, std::move(integrals), mesh, coeffs, {});

        la::Vector<double> b(V->dofmap()->index_map,
            V->dofmap()->index_map_bs());
        fem::assemble_vector(b, L);

        // Lift and solve.
        fem::apply_lifting(b, a, {std::cref(bc)},
            std::optional<std::span<const double>> {}, 1.0);
        fem::set_bc(std::span(b.array()), {std::cref(bc)},
            std::optional<std::span<const double>> {}, 0.0);

        la::Vector<double> u(V->dofmap()->index_map, V->dofmap()->index_map_bs());
        la::KrylovSolver<double> solver;
        solver.set_operator(A);
        solver.set_solver_type("cg");
        solver.set_tolerances(1e-12, 1e-14, 1000);
        const int iters = solver.solve(u, b);
        REQUIRE(iters > 0);

        // Compare to the exact solution at the dof coordinates.
        auto coords = V->tabulate_dof_coordinates(false);
        double max_err = 0;
        for (std::int32_t d = 0; d < V->dofmap()->index_map->size_local(); ++d) {
            const double x = coords[2 * d];
            const double y = coords[2 * d + 1];
            const double exact = std::sin(3.14159265358979323846 * x)
                * std::sin(3.14159265358979323846 * y);
            const double err = std::abs(u.array()[static_cast<std::size_t>(d)] - exact);
            if (err > max_err)
                max_err = err;
        }
        // P1 on a regular mesh converges O(h^2); the error should shrink
        // by ~4x when n doubles.
        REQUIRE(max_err < 0.15 / (n / 4));
        REQUIRE(max_err > 0.0);
    }
}
