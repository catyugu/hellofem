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

    /// Vector type of the iterate.
    using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;

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
            : V(p1_space(n)), cells(all_cells(*V->dofmap())), mesh(V->mesh()), u(V), bc(T(0), boundary_dofs(*this), V)
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
                std::vector<fem::Form<T>::integral_data>>
                integrals;
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
                std::vector<fem::Form<T>::integral_data>>
                integrals;
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

    /// The frozen system of the fixture at `x`: `K u = -M(u^3)`, whose
    /// solution is the fixed-point image `G(x)`.
    std::pair<la::MatrixCSR<double>, la::Vector<double>> picard_system(
        const PicardPoisson& nl, const la::Vector<double>& x)
    {
        nl.set_iterate(x);
        la::MatrixCSR<double> A = nl.stiffness();
        la::Vector<double> b = nl.mass3();
        for (std::size_t i = 0; i < b.array().size(); ++i)
            b[i] = -b[i];
        return {std::move(A), std::move(b)};
    }

    /// A linear contraction towards `(1, 5/9)`, at two different rates.
    Vec contract(const Vec& x)
    {
        Vec G(2);
        G << 0.5 + 0.5 * x[0], 0.5 + 0.1 * x[1];
        return G;
    }

} // namespace

// The configuration takes COMSOL's Anderson acceleration settings.
TEST_CASE("Anderson configuration defaults are COMSOL's", "[nls]")
{
    const nls::AndersonConfig cfg;
    REQUIRE(cfg.dimension == 5);
    REQUIRE(cfg.mixing == Catch::Approx(0.9));
    REQUIRE(cfg.delay == 0);
    REQUIRE(cfg.threshold == Catch::Approx(10.0));
}

// Unit: the mixer takes plain steps during the iteration delay, mixes after
// it, and keeps at most `dimension` pairs.
TEST_CASE("AndersonMixer delays the mixing and bounds its history", "[nls]")
{
    nls::AndersonConfig cfg;
    cfg.dimension = 2;
    cfg.delay = 2;
    nls::AndersonMixer<double> mixer(cfg);

    Vec x = Vec::Zero(2);
    for (int it = 0; it < 6; ++it) {
        const Vec G = contract(x);
        const Vec f = G - x;
        const Vec s = mixer.step(x, G);

        if (it < cfg.delay)
            REQUIRE((s - f).cwiseAbs().maxCoeff() == 0.0);
        if (it == cfg.delay)
            REQUIRE((s - f).cwiseAbs().maxCoeff() > 1e-3);
        REQUIRE(mixer.history_size() == std::min(it + 1, cfg.dimension));
        x += s;
    }
}

// Unit: COMSOL's threshold for the Anderson step falls back to the plain
// step when the mixed one is far longer than the previous one.
TEST_CASE("AndersonMixer rejects a step past the threshold", "[nls]")
{
    const auto second_step = [](double threshold) {
        nls::AndersonConfig cfg;
        cfg.threshold = threshold;
        nls::AndersonMixer<double> mixer(cfg);
        Vec x = Vec::Zero(2);
        mixer.step(x, contract(x)); // plain: no history yet
        x = contract(x);
        return mixer.step(x, contract(x));
    };

    Vec f(2);
    f << 0.25, 0.05; // the plain step of the second call
    REQUIRE((second_step(10.0) - f).cwiseAbs().maxCoeff() > 1e-3);
    REQUIRE((second_step(0.1) - f).cwiseAbs().maxCoeff() < 1e-15);
}

// Unit: on a linear contraction the mixing converges to the fixed point.
TEST_CASE("AndersonMixer converges on a linear contraction", "[nls]")
{
    nls::AndersonMixer<double> mixer(nls::AndersonConfig {});
    Vec x = Vec::Zero(8);
    for (int it = 0; it < 200; ++it) {
        const Vec G = Vec::Constant(8, 1.0) + 0.5 * x;
        x += mixer.step(x, G);
    }
    REQUIRE((x - Vec::Constant(8, 2.0)).cwiseAbs().maxCoeff() < 1e-8);
}

// Contract: an initial guess that already solves the frozen system is
// returned without an iteration or an inner solve.
TEST_CASE("anderson_picard returns a converged initial guess at once", "[nls]")
{
    PicardPoisson nl(8);
    la::Vector<double> x(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());
    x.set(0.0); // u = 0 solves -Delta u + u^3 = 0 with homogeneous Dirichlet

    la::LinearSolver<double> inner;
    const auto result = nls::anderson_picard<double>(
        [&](const la::Vector<double>& xx) { return picard_system(nl, xx); }, x,
        inner, nls::AndersonConfig {});

    REQUIRE(result.converged);
    REQUIRE(result.iterations == 0);
    REQUIRE(result.krylov_iterations == 0);
}

// Contract: an operator that does not read the iterate has one frozen
// system, which a full mixing parameter solves in the first step.
TEST_CASE("anderson_picard solves a frozen operator in one iteration", "[nls]")
{
    PicardPoisson nl(8);
    la::Vector<double> x(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());
    nl.set_initial_guess(x);
    la::Vector<double> frozen(nl.V->dofmap()->index_map,
        nl.V->dofmap()->index_map_bs());
    frozen.set(0.5);

    nls::AndersonConfig cfg;
    cfg.mixing = 1.0;

    la::LinearSolver<double> inner;
    const auto result = nls::anderson_picard<double>(
        [&](const la::Vector<double>&) { return picard_system(nl, frozen); },
        x, inner, cfg);

    REQUIRE(result.converged);
    REQUIRE(result.iterations == 1);
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

    la::LinearSolver<double> inner;
    auto result = nls::anderson_picard<double>(
        [&](const la::Vector<double>& xx) { return picard_system(nl, xx); }, x,
        inner, cfg);

    REQUIRE(result.converged);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.iterations <= 40);
    REQUIRE(result.krylov_iterations > 0);

    double maxu = 0.0;
    for (const double xi : x.array())
        maxu = std::max(maxu, std::abs(xi));
    REQUIRE(maxu < 1e-6);
}

// Unit: the mixing accelerates a stiff linear contraction. Plain Picard
// converges at the slowest rate of the map; mixing five increments resolves
// it in a handful of steps.
TEST_CASE("AndersonMixer accelerates a stiff linear contraction", "[nls]")
{
    const Vec d = (Vec(4) << 0.5, 0.9, 0.99, 0.995).finished();
    const Vec target = Vec::Constant(4, 1.0);
    const auto iterations = [&](int dimension) {
        nls::AndersonConfig cfg;
        cfg.dimension = dimension;
        nls::AndersonMixer<double> mixer(cfg);
        Vec x = Vec::Zero(4);
        int it = 0;
        for (; it < 3000; ++it) {
            const Vec G = d.cwiseProduct(x) + (target - d);
            if ((G - x).cwiseAbs().maxCoeff() <= 1e-8)
                break;
            x += mixer.step(x, G);
        }
        return it;
    };
    REQUIRE(iterations(0) > 1000); // plain Picard, at the slowest rate
    REQUIRE(iterations(5) <= 20); // mixed, in a handful of steps
}
