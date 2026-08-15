// hellofem::nls — Anderson-accelerated Picard solver tests
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
#include "nls/AndersonPicard.h"

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

    /// Nonlinear Poisson -Delta u + u^3 = 0 with homogeneous Dirichlet,
    /// P1, solved by Anderson-accelerated Picard. The fixed-point map is
    /// G(u) = K^-1 (b - M(u^3)) where K is the stiffness matrix and M the
    /// u^3 load, so the Picard iteration drives u to 0.
    struct PicardPoisson {
        using T = double;
        std::shared_ptr<fem::FunctionSpace<T>> V;
        std::vector<std::int32_t> cells;
        std::shared_ptr<const mesh::Mesh<T>> mesh;
        fem::Function<T> u;
        std::vector<std::shared_ptr<const fem::Function<T>>> coeffs;
        std::vector<std::int32_t> bc_dofs;
        fem::DirichletBC<T> bc;

        std::shared_ptr<fem::PrecomputeData<T>> pre_stiff;
        std::shared_ptr<fem::PrecomputeData<T>> pre_mass3;

        static void stiffness_kernel(T* Ae, const fem::CellKernelData<T>& d)
        {
            const int nq = d.num_points, nd = d.num_dofs0;
            const int tdim = d.tdim;
            std::fill(Ae, Ae + nd * nd, T(0));
            for (int q = 0; q < nq; ++q)
                for (int i = 0; i < nd; ++i)
                    for (int j = 0; j < nd; ++j) {
                        T dot = 0;
                        for (int c = 0; c < tdim; ++c)
                            dot += d.dphi0[(q * nd + i) * tdim + c]
                                * d.dphi1[(q * nd + j) * tdim + c];
                        Ae[i * nd + j] += d.w[q] * d.detJ[q] * dot;
                    }
        }

        // Load of u^3: Ae(i) += sum_q w detJ u(q)^3 phi_i.
        static void mass3_kernel(T* Ae, const fem::CellKernelData<T>& d)
        {
            const int nq = d.num_points, nd = d.num_dofs0;
            std::fill(Ae, Ae + nd, T(0));
            for (int q = 0; q < nq; ++q) {
                const T uq3 = d.coeffs[q] * d.coeffs[q] * d.coeffs[q];
                for (int i = 0; i < nd; ++i)
                    Ae[i] += d.w[q] * d.detJ[q] * uq3
                        * d.phi0[q * nd + i];
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

        static std::vector<std::int32_t> boundary_dofs(const PicardPoisson& p)
        {
            auto e_to_c = p.mesh->topology()->connectivity(1, 2);
            std::vector<std::int32_t> edges;
            for (std::int32_t e = 0; e < e_to_c->num_nodes(); ++e)
                if (e_to_c->num_links(e) == 1)
                    edges.push_back(e);
            return fem::DirichletBC<T>::locate_dofs_topological(
                *p.mesh->topology(), *p.V->dofmap(), 1, edges);
        }

        explicit PicardPoisson(int n)
            : V(p1_space(n)), cells(all_cells(*V->dofmap())), mesh(V->mesh()),
              u(V), bc(T(0), boundary_dofs(*this), V)
        {
            // The nonlinear coefficient is the iterate `u` itself; `coeffs`
            // aliases u's coefficient vector so assembly sees the current
            // Picard iterate (set_iterate writes into u).
            coeffs.push_back(std::make_shared<fem::Function<T>>(u));
            bc_dofs.assign(bc.dof_indices().begin(), bc.dof_indices().end());

            auto coord = mesh->geometry().cmaps().front();
            pre_stiff = std::make_shared<fem::PrecomputeData<T>>(
                mesh::CellType::triangle, *V->element(), *V->element(),
                std::vector<const fem::FiniteElement<T>*> {}, coord, 2);
            pre_mass3 = std::make_shared<fem::PrecomputeData<T>>(
                mesh::CellType::triangle, *V->element(), *V->element(),
                std::vector<const fem::FiniteElement<T>*> {V->element().get()},
                coord, 4);
        }

        void set_iterate(const la::Vector<T>& x) const
        {
            std::copy(x.array().begin(), x.array().end(),
                u.x()->array().begin());
        }

        /// Stiffness matrix with homogeneous Dirichlet.
        la::MatrixCSR<T> stiffness() const
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
            la::MatrixCSR<T> K(pattern);

            fem::Form<T>::integral_data stiff;
            stiff.kernel = fem::make_cell_kernel(*pre_stiff, stiffness_kernel);
            stiff.entities = cells;
            stiff.coeffs = {};
            std::vector<std::shared_ptr<const fem::FunctionSpace<T>>> Vlist {V, V};
            std::map<std::pair<fem::IntegralType, int>,
                std::vector<fem::Form<T>::integral_data>> integrals;
            integrals[{fem::IntegralType::cell, 0}] = {stiff};
            fem::Form<T> a(Vlist, std::move(integrals), mesh, {}, {});
            fem::assemble_matrix(K.mat_add_values(), a, {std::cref(bc)});
            fem::set_diagonal(K.mat_set_values(), a, {std::cref(bc)}, T(1));
            return K;
        }

        /// The `u^3 v` load, computed at the current iterate.
        la::Vector<T> mass3() const
        {
            fem::Form<T>::integral_data mass;
            mass.kernel = fem::make_cell_kernel(*pre_mass3, mass3_kernel);
            mass.entities = cells;
            mass.coeffs = {0};
            std::vector<std::shared_ptr<const fem::FunctionSpace<T>>> Vlist {V};
            std::map<std::pair<fem::IntegralType, int>,
                std::vector<fem::Form<T>::integral_data>> integrals;
            integrals[{fem::IntegralType::cell, 0}] = {mass};
            fem::Form<T> L(Vlist, std::move(integrals), mesh, coeffs, {});

            la::Vector<T> b(V->dofmap()->index_map,
                V->dofmap()->index_map_bs());
            fem::assemble_vector(b, L);
            for (std::int32_t d : bc_dofs)
                b[static_cast<std::size_t>(d)] = T(0);
            return b;
        }

        void set_initial_guess(la::Vector<T>& x) const
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

// Unit: the Anderson mixer falls back to Picard during warmup, mixes after,
// and respects the depth bound.
TEST_CASE("AndersonMixer warmup, mixing and depth", "[nls]")
{
    nls::AndersonMixer<double> mixer(3, 2, 0.8, 1.5, true);
    auto imap = std::make_shared<common::IndexMap>(0, 4);
    la::Vector<double> x(imap, 1), G(imap, 1);
    for (int i = 0; i < 4; ++i) {
        x[i] = static_cast<double>(i + 1);
        G[i] = 0.5 * static_cast<double>(i + 1);
    }

    // Warmup: no history -> plain Picard.
    REQUIRE(not mixer.step(x, G).has_value());
    mixer.push(x, G);

    // Still within warmup (iter_count == 1 < 2).
    REQUIRE(not mixer.step(x, G).has_value());
    mixer.push(x, G);

    // Now mixing is active (iter_count == 2 >= warmup).
    REQUIRE(mixer.step(x, G).has_value());
    REQUIRE(mixer.has_history());
}

// Unit: on a linear contraction the mixer converges to the fixed point.
TEST_CASE("AndersonMixer converges on a linear contraction", "[nls]")
{
    nls::AndersonMixer<double> mixer(5, 2, 0.8, 1.5, true);
    auto imap = std::make_shared<common::IndexMap>(0, 8);
    la::Vector<double> x(imap, 1), G(imap, 1);
    x.set(0.0);
    for (int it = 0; it < 200; ++it) {
        for (int i = 0; i < 8; ++i)
            G[i] = 1.0 + 0.5 * x[i]; // contraction to 2.0
        const auto prop = mixer.step(x, G);
        if (prop)
            x = *prop;
        else
            for (int i = 0; i < 8; ++i)
                x[i] += 0.8 * (G[i] - x[i]);
        mixer.push(x, G);
    }
    double err = 0;
    for (int i = 0; i < 8; ++i)
        err = std::max(err, std::abs(x[i] - 2.0));
    REQUIRE(err < 1e-8);
}

// End-to-end: Anderson-accelerated Picard solves the nonlinear Poisson.
TEST_CASE("Anderson-accelerated Picard solves nonlinear Poisson", "[nls]")
{
    PicardPoisson nl(8);
    la::Vector<double> x(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());
    nl.set_initial_guess(x);

    nls::AndersonConfig cfg;
    cfg.max_iterations = 60;
    cfg.relative_tolerance = 1e-8;
    cfg.absolute_tolerance = 1e-12;

    auto result = nls::anderson_picard<double>(
        [&](const la::Vector<double>& xx)
            -> std::pair<la::MatrixCSR<double>, la::Vector<double>> {
            nl.set_iterate(xx);
            la::MatrixCSR<double> A = nl.stiffness();
            la::Vector<double> b = nl.mass3();
            for (std::size_t i = 0; i < b.array().size(); ++i)
                b[i] = -b[i]; // G = A^-1 b is the fixed-point map
            return {std::move(A), std::move(b)};
        },
        x, cfg);

    REQUIRE(result.converged);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.iterations <= 40);
    REQUIRE(result.krylov_iterations > 0);

    double maxu = 0.0;
    for (const double xi : x.array())
        maxu = std::max(maxu, std::abs(xi));
    REQUIRE(maxu < 1e-6);
}
