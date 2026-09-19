// hellofem::app — field solver base: function space, solution, linearization
// SPDX-License-Identifier: MIT
#pragma once

#include "property.h"
#include "solver.h"
#include "time_scheme.h"

#include "fem/DirichletBC.h"
#include "fem/Form.h"
#include "fem/Function.h"
#include "fem/FunctionSpace.h"
#include "fem/facet_precompute.h"
#include "fem/precompute.h"
#include "la/MatrixCSR.h"
#include "la/SparsityPattern.h"
#include "la/Vector.h"
#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace hellofem::app {

    /// Base for a single-physics field solver: owns the function space, the
    /// solution, the sparsity pattern of the linearized system and the mesh
    /// topology queries the physics needs.
    class FieldSolver {
    public:
        FieldSolver(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order,
            int value_dim);
        virtual ~FieldSolver() = default;

        /// Evaluate the model data that depends on time or on the solution
        /// (material properties, boundary values, source terms) at time `t`,
        /// taking the current solution as the field state. Call before
        /// assembling a system.
        virtual void refresh(double t) = 0;

        /// Assemble the linearized steady system `A u = b` at the current
        /// state, with the Dirichlet conditions imposed.
        virtual void assemble_steady(la::MatrixCSR<double>& A,
            la::Vector<double>& b) const = 0;

        /// Solve the steady system at time `t`, starting from the current
        /// solution (the previous level, or the previous linearization — both
        /// are good initial guesses). A solution-dependent material law makes
        /// the system nonlinear, and the solve iterates it to convergence.
        /// @return Iterations used (0 for the single solve of a linear one).
        int solve_steady(double t);

        /// Impose the Dirichlet data of time `t` on the current solution. A
        /// state that is not the result of a solve (the initial one of a
        /// transient run) must satisfy the pointwise constraints.
        virtual void constrain_solution(double t) = 0;

        std::shared_ptr<fem::Function<double>> solution() const { return u_; }

        std::shared_ptr<fem::FunctionSpace<double>> space() const { return V_; }

        /// Sparsity pattern of the linearized system.
        const la::SparsityPattern& pattern() const { return *pattern_; }

        /// All cells, the assembly range.
        std::vector<std::int32_t> cells() const;

        /// The linear solver of this field's systems: the app's defaults,
        /// changed by a physics whose operator a direct factorization suits
        /// better (see `solid.cpp`).
        const la::LinearSettings& linear_settings() const { return linear_; }

        /// The solve of this field's systems. It is the same object at every
        /// level, so it keeps what it built for an operator across them (see
        /// `la::LinearSolver`).
        la::LinearSolver<double>& linear_solver() { return linear_solver_; }

    protected:
        /// Dofs on the facets carrying the given 1-based boundary ids.
        std::vector<std::int32_t> boundary_dofs(const std::set<int>& ids) const;

        /// `(cell, local facet)` pairs of the given 1-based boundary ids.
        ///
        /// A facet two cells share is emitted once, paired with one of them,
        /// when `interior` is set: the integral's support is the facet itself
        /// and the two cells share the dofs on it, so the one cell's scatter
        /// carries the facet's whole contribution. That is what a condition
        /// on an *interior* boundary — a thin layer imprinted inside a
        /// domain — needs. A boundary flux or a Dirichlet condition takes the
        /// exterior facets alone, which is the default.
        std::vector<std::int32_t> boundary_facets(const std::set<int>& ids,
            bool interior = false) const;

        /// Dirichlet conditions of `(boundary id -> value)` at time `t`.
        std::vector<fem::DirichletBC<double>> make_bcs(
            const std::map<int, ScalarExpression>& values, double t) const;

        /// The Functions a weak form reads, in the order its kernel expects
        /// them (material properties, then fields).
        using Coefficients
            = std::vector<std::shared_ptr<const fem::Function<double>>>;
        /// Constants a weak form reads after its coefficients.
        using Constants
            = std::vector<std::shared_ptr<const fem::Constant<double>>>;
        /// Dirichlet marker per physical dof, as `DirichletBC::mark_dofs`
        /// writes it.
        using DirichletRows = std::vector<std::int8_t>;

        /// The marker of the given conditions.
        DirichletRows marked_rows(
            const std::vector<fem::DirichletBC<double>>& bcs) const;

        /// Reference data of a cell integral whose coefficients are `coeffs`,
        /// in the order the kernel reads them.
        std::shared_ptr<const fem::PrecomputeData<double>> cell_precompute(
            const Coefficients& coeffs) const;

        /// Facet counterpart of `cell_precompute`.
        std::shared_ptr<const fem::FacetPrecomputeData<double>> facet_precompute(
            const Coefficients& coeffs) const;

        /// Add the assembled form `a` to `A`, masking the Dirichlet rows.
        void assemble_into(la::MatrixCSR<double>& A, const fem::Form<double>& a,
            const DirichletRows& bc_rows) const;

        /// Add the integral of the weak form `w`, whose coefficients are
        /// `coeffs` in order, to the matrix `A` with the rows in `bc_rows`
        /// masked — over every cell, or over the exterior facets of
        /// `boundary`. Returns the form, which names the spaces that
        /// imposing the Dirichlet data needs.
        fem::Form<double> add_operator(la::MatrixCSR<double>& A,
            const DirichletRows& bc_rows, const Coefficients& coeffs,
            fem::cell_kernel_weak_fn_t<double> w) const;
        fem::Form<double> add_operator(la::MatrixCSR<double>& A,
            const DirichletRows& bc_rows, const Coefficients& coeffs,
            fem::facet_kernel_weak_fn_t<double> w,
            const std::set<int>& boundaries, bool interior = false) const;

        /// Add the integral of the weak form `w` (coefficients `coeffs`,
        /// constants after them) to the load `b`.
        void add_load(la::Vector<double>& b, const Coefficients& coeffs,
            fem::cell_kernel_weak_fn_t<double> w, Constants constants = {}) const;
        void add_load(la::Vector<double>& b, const Coefficients& coeffs,
            fem::facet_kernel_weak_fn_t<double> w,
            const std::set<int>& boundaries, bool interior = false) const;

        /// Impose the Dirichlet data of `bcs` (dof marker `marked`) on the
        /// assembled system `a u = b`: the known boundary-column
        /// contribution moves to the right-hand side, those columns are
        /// zeroed and their diagonal set, so the boundary rows read `u = g`.
        void impose_dirichlet(la::MatrixCSR<double>& A, la::Vector<double>& b,
            const fem::Form<double>& a,
            const std::vector<fem::DirichletBC<double>>& bcs,
            const DirichletRows& marked) const;

        /// Write the Dirichlet values of `bcs` into the solution `x`.
        void impose_values(std::span<double> x,
            const std::vector<fem::DirichletBC<double>>& bcs) const;

        std::shared_ptr<const mesh::Mesh<double>> mesh_;
        std::shared_ptr<const mesh::MeshTags<int>> facet_tags_;
        std::shared_ptr<const mesh::MeshTags<int>> cell_tags_;
        std::shared_ptr<fem::FunctionSpace<double>> V_;
        std::shared_ptr<fem::Function<double>> u_;
        std::shared_ptr<la::SparsityPattern> pattern_;
        la::LinearSettings linear_;
        la::LinearSolver<double> linear_solver_;
        double t_ = 0.0; // time of the last refresh
        int order_ = 1; // element order of the field, sizes the quadrature
    };

    /// A field with a time derivative: it assembles one step of the time
    /// discretization of its own equation, `M u' + K u = f`, at the new
    /// level. The weights and the solution history come from the time
    /// stepping driver.
    class TimeDependentField : public FieldSolver {
    public:
        TimeDependentField(std::shared_ptr<const mesh::Mesh<double>> mesh,
            std::shared_ptr<const mesh::MeshTags<int>> facet_tags,
            std::shared_ptr<const mesh::MeshTags<int>> cell_tags, int order,
            int value_dim)
            : FieldSolver(std::move(mesh), std::move(facet_tags),
                  std::move(cell_tags), order, value_dim)
        {
        }

        /// Assemble one time step of the field's equation with the scheme
        /// weights of `level`; the Dirichlet data is taken at the level
        /// time.
        virtual void assemble_step(la::MatrixCSR<double>& A,
            la::Vector<double>& b, const TimeLevel& level) const = 0;
    };

} // namespace hellofem::app
