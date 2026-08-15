// hellofem::nls — Newton/JFNK solver tests
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
#include "la/LinearOperator.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/Mesh.h"
#include "mesh/Topology.h"
#include "mesh/cell_types.h"
#include "mesh/generation.h"
#include "nls/NewtonSolver.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

using namespace hellofem;

namespace {

    using B = basis::element::family;
    using C = basis::cell::type;
    using LV = basis::element::lagrange_variant;
    using DV = basis::element::dpc_variant;

    /// Nonlinear Poisson on the unit square: -Delta u + u^3 = 0 with
    /// homogeneous Dirichlet BC, P1. The only solution is u = 0, so
    /// convergence to zero is a clean check for both the matrix-free
    /// (JFNK) and assembled-Jacobian Newton paths.
    struct NlPoisson {
        using T = double;
        std::shared_ptr<fem::FunctionSpace<T>> V;
        std::vector<std::int32_t> cells;
        std::shared_ptr<const mesh::Mesh<T>> mesh;
        fem::Function<T> u;                    // coefficient (current iterate)
        std::vector<std::shared_ptr<const fem::Function<T>>> coeffs;
        std::vector<std::int32_t> bc_dofs;
        fem::DirichletBC<T> bc;

        // Reference-cell quadrature data. `make_cell_kernel` captures the
        // PrecomputeData by reference, so the kernels must outlive the
        // forms; hold them as members.
        std::shared_ptr<fem::PrecomputeData<T>> pre_stiff;
        std::shared_ptr<fem::PrecomputeData<T>> pre_dmass;
        std::shared_ptr<fem::PrecomputeData<T>> pre_mass3;

        // Stiffness kernel: Ae(i,j) += sum_q w detJ dphi_i.dphi_j.
        static void stiffness_kernel(T* Ae,
            const fem::CellKernelData<T>& data)
        {
            const int nq = data.num_points, nd = data.num_dofs0;
            const int tdim = data.tdim;
            std::fill(Ae, Ae + nd * nd, T(0));
            for (int q = 0; q < nq; ++q)
                for (int i = 0; i < nd; ++i)
                    for (int j = 0; j < nd; ++j) {
                        T dot = 0;
                        for (int c = 0; c < tdim; ++c)
                            dot += data.dphi0[(q * nd + i) * tdim + c]
                                * data.dphi1[(q * nd + j) * tdim + c];
                        Ae[i * nd + j] += data.w[q] * data.detJ[q] * dot;
                    }
        }

        // Jacobian of the u^3 term: Ae(i,j) += sum_q w detJ 3 u(q)^2 phi_i phi_j.
        static void d_mass_kernel(T* Ae, const fem::CellKernelData<T>& data)
        {
            const int nq = data.num_points, nd = data.num_dofs0;
            std::fill(Ae, Ae + nd * nd, T(0));
            for (int q = 0; q < nq; ++q) {
                const T fu = 3 * data.coeffs[q] * data.coeffs[q];
                for (int i = 0; i < nd; ++i)
                    for (int j = 0; j < nd; ++j)
                        Ae[i * nd + j] += data.w[q] * data.detJ[q] * fu
                            * data.phi0[q * nd + i] * data.phi1[q * nd + j];
            }
        }

        // Residual load: Ae(i) += sum_q w detJ u(q)^3 phi_i.
        static void mass3_kernel(T* Ae, const fem::CellKernelData<T>& data)
        {
            const int nq = data.num_points, nd = data.num_dofs0;
            std::fill(Ae, Ae + nd, T(0));
            for (int q = 0; q < nq; ++q) {
                const T uq3 = data.coeffs[q] * data.coeffs[q] * data.coeffs[q];
                for (int i = 0; i < nd; ++i)
                    Ae[i] += data.w[q] * data.detJ[q] * uq3
                        * data.phi0[q * nd + i];
            }
        }

        static std::shared_ptr<fem::FunctionSpace<T>> p1_space(int n)
        {
            auto m = mesh::create_unit_square(n);
            m->topology_mutable()->create_entities(1);
            m->topology_mutable()->create_connectivity(2, 1);
            m->topology_mutable()->create_connectivity(1, 2);
            auto fe = std::make_shared<fem::FiniteElement<T>>(
                basis::create_element<T>(B::P, C::triangle, 1, LV::equispaced,
                    DV::unset, false));
            auto layout = fem::CoordinateElement<T>(mesh::CellType::triangle, 1,
                LV::equispaced)
                              .create_dof_layout();
            auto [imap, bs, dofmaps] = fem::build_dofmap_data(
                *m->topology(), {layout}, nullptr);
            auto dmap = std::make_shared<fem::DofMap>(layout,
                std::make_shared<common::IndexMap>(std::move(imap)), bs,
                std::move(dofmaps.front()), bs);
            return std::make_shared<fem::FunctionSpace<T>>(m, fe, dmap);
        }

        static std::vector<std::int32_t> all_cells(const fem::DofMap& dmap)
        {
            std::vector<std::int32_t> c(dmap.map().extent(0));
            for (std::size_t i = 0; i < c.size(); ++i)
                c[i] = static_cast<std::int32_t>(i);
            return c;
        }

        /// Boundary dofs (edges with a single incident cell).
        static std::vector<std::int32_t> boundary_dofs(const NlPoisson& p)
        {
            auto e_to_c = p.mesh->topology()->connectivity(1, 2);
            std::vector<std::int32_t> edges;
            for (std::int32_t e = 0; e < e_to_c->num_nodes(); ++e)
                if (e_to_c->num_links(e) == 1)
                    edges.push_back(e);
            return fem::DirichletBC<T>::locate_dofs_topological(
                *p.mesh->topology(), *p.V->dofmap(), 1, edges);
        }

        explicit NlPoisson(int n)
            : V(p1_space(n)), cells(all_cells(*V->dofmap())), mesh(V->mesh()),
              u(V), bc(T(0), boundary_dofs(*this), V)
        {
            // The nonlinear coefficient is the iterate `u` itself; `coeffs`
            // aliases u's coefficient vector so assembly sees the current
            // Newton iterate (set_iterate writes into u).
            coeffs.push_back(std::make_shared<fem::Function<T>>(u));
            bc_dofs.assign(bc.dof_indices().begin(), bc.dof_indices().end());

            auto coord = mesh->geometry().cmaps().front();
            pre_stiff = std::make_shared<fem::PrecomputeData<T>>(
                mesh::CellType::triangle, *V->element(), *V->element(),
                std::vector<const fem::FiniteElement<T>*> {}, coord, 2);
            pre_dmass = std::make_shared<fem::PrecomputeData<T>>(
                mesh::CellType::triangle, *V->element(), *V->element(),
                std::vector<const fem::FiniteElement<T>*> {V->element().get()},
                coord, 4);
            pre_mass3 = pre_dmass;
        }

        /// Copy `x` into the coefficient function `u`.
        void set_iterate(const la::Vector<T>& x) const
        {
            std::copy(x.array().begin(), x.array().end(),
                u.x()->array().begin());
        }

        /// The grad-grad bilinear form (same for the residual matrix and
        /// the assembled Jacobian).
        fem::Form<T> stiffness_form() const
        {
            fem::Form<T>::integral_data stiff;
            stiff.kernel = fem::make_cell_kernel(*pre_stiff, stiffness_kernel);
            stiff.entities = cells;
            stiff.coeffs = {};
            std::vector<std::shared_ptr<const fem::FunctionSpace<T>>> Vlist {V, V};
            std::map<std::pair<fem::IntegralType, int>,
                std::vector<fem::Form<T>::integral_data>> integrals;
            integrals[{fem::IntegralType::cell, 0}] = {stiff};
            return fem::Form<T>(Vlist, std::move(integrals), mesh, {}, {});
        }

        /// The `3 u^2 v` Jacobian bilinear form (coefficient u).
        fem::Form<T> d_mass_form() const
        {
            fem::Form<T>::integral_data mass;
            mass.kernel = fem::make_cell_kernel(*pre_dmass, d_mass_kernel);
            mass.entities = cells;
            mass.coeffs = {0};
            std::vector<std::shared_ptr<const fem::FunctionSpace<T>>> Vlist {V, V};
            std::map<std::pair<fem::IntegralType, int>,
                std::vector<fem::Form<T>::integral_data>> integrals;
            integrals[{fem::IntegralType::cell, 0}] = {mass};
            return fem::Form<T>(Vlist, std::move(integrals), mesh, coeffs, {});
        }

        /// The `u^3 v` residual linear form (coefficient u).
        fem::Form<T> mass3_form() const
        {
            fem::Form<T>::integral_data mass;
            mass.kernel = fem::make_cell_kernel(*pre_mass3, mass3_kernel);
            mass.entities = cells;
            mass.coeffs = {0};
            std::vector<std::shared_ptr<const fem::FunctionSpace<T>>> Vlist {V};
            std::map<std::pair<fem::IntegralType, int>,
                std::vector<fem::Form<T>::integral_data>> integrals;
            integrals[{fem::IntegralType::cell, 0}] = {mass};
            return fem::Form<T>(Vlist, std::move(integrals), mesh, coeffs, {});
        }

        /// A matrix on the cell-coupling pattern of the spaces.
        la::MatrixCSR<T> make_matrix() const
        {
            la::SparsityPattern pattern(V->dofmap()->index_map, 1);
            fem::sparsitybuild::cells(pattern, std::pair {cells, cells},
                {*V->dofmap(), *V->dofmap()});
            std::vector<std::int32_t> diag(
                V->dofmap()->index_map->size_local());
            for (std::int32_t d = 0; d < V->dofmap()->index_map->size_local();
                 ++d)
                diag[static_cast<std::size_t>(d)] = d;
            pattern.insert_diagonal(std::span(diag));
            pattern.finalize();
            return la::MatrixCSR<T>(pattern);
        }

        /// Assemble the grad-grad matrix with homogeneous Dirichlet
        /// (BC rows/cols zeroed, diagonal 1).
        la::MatrixCSR<T> make_stiffness() const
        {
            la::MatrixCSR<T> K = make_matrix();
            auto a = stiffness_form();
            fem::assemble_matrix(K.mat_add_values(), a, {std::cref(bc)});
            fem::set_diagonal(K.mat_set_values(), a, {std::cref(bc)}, T(1));
            return K;
        }

        /// Residual F(u) = K u + integral u^3 v, with the BC dofs pinned
        /// (F_i = u_i on the boundary so the iterate is driven to zero).
        void residual(const la::Vector<T>& x, la::Vector<T>& out) const
        {
            set_iterate(x);
            la::MatrixCSR<T> K = make_stiffness();

            out.set(0);
            K.mult(x, out); // out += K x

            // Add integral u^3 v, zeroing BC entries.
            auto L = mass3_form();
            la::Vector<T> m3(V->dofmap()->index_map,
                V->dofmap()->index_map_bs());
            fem::assemble_vector(m3, L);
            for (std::int32_t d : bc_dofs)
                m3[static_cast<std::size_t>(d)] = 0;
            for (std::size_t i = 0; i < out.array().size(); ++i)
                out[i] += m3[i];
        }

        /// Assembled Jacobian J(u) = K + 3 integral u^2 v, with the same
        /// Dirichlet handling.
        void jacobian(const la::Vector<T>& x, la::MatrixCSR<T>& J) const
        {
            set_iterate(x);
            std::fill(J.values().begin(), J.values().end(), T(0));
            auto a = stiffness_form();
            auto am = d_mass_form();
            fem::assemble_matrix(J.mat_add_values(), a, {std::cref(bc)});
            fem::assemble_matrix(J.mat_add_values(), am, {std::cref(bc)});
            fem::set_diagonal(J.mat_set_values(), a, {std::cref(bc)}, T(1));
        }

        /// Interpolate a smooth field as the initial guess (BC satisfied).
        void set_initial_guess(la::Vector<T>& x)
        {
            fem::Function<T> g(V);
            g.interpolate(
                [](std::span<const T> X, std::array<std::size_t, 2> shape) {
                    const T pi = 3.14159265358979323846;
                    std::vector<T> v(shape[0]);
                    for (std::size_t i = 0; i < shape[0]; ++i)
                        v[i] = T(0.1) * std::sin(pi * X[2 * i])
                            * std::sin(pi * X[2 * i + 1]);
                    return std::make_pair(std::move(v),
                        std::array<std::size_t, 2> {shape[0], 1});
                });
            std::copy(g.x()->array().begin(), g.x()->array().end(),
                x.array().begin());
        }
    };

} // namespace

// Finite-difference Jacobian-vector product vs analytic, on a 1D problem.
TEST_CASE("JFNK finite-difference Jacobian matches analytic", "[nls]")
{
    constexpr int N = 20;
    const double h = 1.0 / (N + 1);
    auto residual = [&](const std::vector<double>& u) {
        std::vector<double> r(N, 0.0);
        for (int i = 0; i < N; ++i) {
            const double uim = i > 0 ? u[i - 1] : 0.0;
            const double uip = i < N - 1 ? u[i + 1] : 0.0;
            r[i] = -(uim - 2 * u[i] + uip) / (h * h)
                + u[i] * u[i] * u[i];
        }
        return r;
    };
    std::vector<double> u(N);
    for (int i = 0; i < N; ++i)
        u[i] = 0.1 * std::sin(3.14159265358979323846 * (i + 1) / (N + 1));
    const auto Fu = residual(u);
    std::vector<double> v(N);
    for (int i = 0; i < N; ++i)
        v[i] = std::sin(2.0 * 3.14159265358979323846 * (i + 1) / (N + 1));

    const double sigma = 1e-6;
    std::vector<double> up(N);
    for (int i = 0; i < N; ++i)
        up[i] = u[i] + sigma * v[i];
    const auto Fup = residual(up);
    std::vector<double> Jv(N);
    for (int i = 0; i < N; ++i)
        Jv[i] = (Fup[i] - Fu[i]) / sigma;

    // Analytic J = -D2 + 3 u^2 I applied to v.
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double Jav = (2.0 / (h * h) + 3 * u[i] * u[i]) * v[i];
        if (i > 0)
            Jav += (-1.0 / (h * h)) * v[i - 1];
        if (i < N - 1)
            Jav += (-1.0 / (h * h)) * v[i + 1];
        max_err = std::max(max_err, std::abs(Jv[i] - Jav));
    }
    REQUIRE(max_err < 1e-3);
}

// End-to-end: matrix-free JFNK Newton solves the nonlinear Poisson.
TEST_CASE("JFNK matrix-free Newton solves nonlinear Poisson", "[nls]")
{
    NlPoisson nl(8);
    la::Vector<double> x(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());
    nl.set_initial_guess(x);
    la::Vector<double> b(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());

    nls::NewtonSolver<double> solver;
    solver.set_residual(
        [&](const la::Vector<double>& xx, la::Vector<double>& bb) {
            nl.residual(xx, bb);
        },
        b);
    solver.max_it = 20;
    solver.rtol = 1e-8;
    solver.atol = 1e-12;
    solver.linear_solver = "gmres";
    solver.linear_max_iter = 1000;

    const auto [iters, converged] = solver.solve(x);
    REQUIRE(converged);
    REQUIRE(iters > 0);
    REQUIRE(iters <= 15);

    double maxu = 0.0;
    for (const double xi : x.array())
        maxu = std::max(maxu, std::abs(xi));
    REQUIRE(maxu < 1e-6);
    REQUIRE(solver.krylov_iterations() > 0);
}

// End-to-end: assembled-Jacobian Newton solves the same problem and agrees
// with the matrix-free path.
TEST_CASE("Assembled-Jacobian Newton matches matrix-free", "[nls]")
{
    NlPoisson nl(8);
    la::Vector<double> x(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());
    nl.set_initial_guess(x);
    la::Vector<double> b(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());
    la::MatrixCSR<double> J = nl.make_matrix();

    nls::NewtonSolver<double> solver;
    solver.set_residual(
        [&](const la::Vector<double>& xx, la::Vector<double>& bb) {
            nl.residual(xx, bb);
        },
        b);
    solver.set_jacobian(
        [&](const la::Vector<double>& xx, la::MatrixCSR<double>& Jmat) {
            nl.jacobian(xx, Jmat);
        },
        J);
    solver.max_it = 20;
    solver.rtol = 1e-8;
    solver.atol = 1e-12;
    solver.linear_solver = "gmres";

    const auto [iters, converged] = solver.solve(x);
    REQUIRE(converged);
    REQUIRE(iters > 0);

    double maxu = 0.0;
    for (const double xi : x.array())
        maxu = std::max(maxu, std::abs(xi));
    REQUIRE(maxu < 1e-6);
}
