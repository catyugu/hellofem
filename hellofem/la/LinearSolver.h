// hellofem::la — the solve of a linear system, kept across calls
// SPDX-License-Identifier: MIT

#pragma once

#include "KrylovSolver.h"
#include "MatrixCSR.h"
#include "Vector.h"
#include "direct.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hellofem::la {

    /// Settings of a linear solve: the method, its preconditioner and its
    /// tolerances.
    ///
    /// `solver_type` names a Krylov method ("cg", "gmres", "bicgstab") or a
    /// direct factorization ("direct"), which ignores the preconditioner.
    struct LinearSettings {
        std::string solver_type = "cg";
        std::string preconditioner_type = "amg";
        double rtol = 1e-12;
        double atol = 1e-14;
        int max_iterations = 2000;

        bool operator==(const LinearSettings&) const = default;
    };

    /// The solve of `A x = b`, keeping what it built for `A`: the
    /// factorization of a direct solve, the preconditioner of an iterative
    /// one.
    ///
    /// A transient run solves an operator that barely moves from one level to
    /// the next, and a factorization costs orders of magnitude more than the
    /// solve it enables; keeping it is what makes those repeated solves cost
    /// the factorization once instead of once each. The operator is taken as
    /// new only when it differs from the one the solve was built for (see
    /// `operator_changed`).
    ///
    /// @tparam T Scalar type of the system.
    template <typename T>
    class LinearSolver {
    public:
        /// Solve `A x = b`, taking `x` as the initial guess when
        /// `warm_start`.
        /// @param[in,out] x On entry the initial guess, on exit the solution.
        /// @return Iterations used (0 for a direct solve).
        int solve(const MatrixCSR<T>& A, Vector<T>& x, const Vector<T>& b,
            bool warm_start, const LinearSettings& settings)
        {
            if (operator_changed(A, settings))
                build(A, settings);
            if constexpr (std::is_same_v<T, double>) {
                if (direct_) {
                    direct_->solve(x, b);
                    return 0;
                }
            }
            krylov_->set_initial_guess(warm_start);
            return krylov_->solve(x, b);
        }

    private:
        /// Build the solve for `A`.
        void build(const MatrixCSR<T>& A, const LinearSettings& settings)
        {
            direct_.reset();
            krylov_.reset();
            if (settings.solver_type == "direct") {
                if constexpr (std::is_same_v<T, double>)
                    direct_ = std::make_unique<DirectSolver>(A);
                else
                    throw std::runtime_error(
                        "a direct solve is real double only.");
            }
            else {
                krylov_ = std::make_unique<KrylovSolver<T>>();
                krylov_->set_operator(A);
                krylov_->set_solver_type(settings.solver_type);
                if (settings.preconditioner_type != "none")
                    krylov_->set_preconditioner_type(
                        settings.preconditioner_type);
            }
            settings_ = settings;
            values_.assign(A.values().begin(), A.values().end());
        }

        /// Whether `A` calls for a new build.
        ///
        /// Re-assembling one operator is not bit-for-bit reproducible — a
        /// parallel assembly sums the contributions of its elements in a
        /// different order — so comparing exactly would rebuild on every
        /// solve. The two cases are orders of magnitude apart, though: an
        /// unchanged operator moves by a few ulps of its largest entry, and a
        /// changed one by far more. An operator that moved by less than the
        /// tolerance below is the operator the solve was built for, and one
        /// that moved by more is taken as new — the tolerance errs towards
        /// rebuilding, which costs a factorization rather than an answer.
        bool operator_changed(
            const MatrixCSR<T>& A, const LinearSettings& settings) const
        {
            if (not direct_ and not krylov_)
                return true;
            if (not(settings == settings_))
                return true;
            const auto& v = A.values();
            if (v.size() != values_.size())
                return true;

            double scale = 0.0;
            double change = 0.0;
            for (std::size_t i = 0; i < v.size(); ++i) {
                scale = std::max(scale,
                    static_cast<double>(std::abs(values_[i])));
                change = std::max(change,
                    static_cast<double>(std::abs(v[i] - values_[i])));
            }
            return change > operator_change_tolerance * scale;
        }

        /// An operator that moved by less than this, relative to its largest
        /// entry, is the operator the solve was built for.
        static constexpr double operator_change_tolerance = 1e-13;

        std::unique_ptr<DirectSolver> direct_;
        std::unique_ptr<KrylovSolver<T>> krylov_;
        LinearSettings settings_;

        /// The values of the operator the solve was built for.
        std::vector<T> values_;
    };

} // namespace hellofem::la
