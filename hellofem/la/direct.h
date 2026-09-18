// hellofem::la — direct sparse solve of an assembled system
// SPDX-License-Identifier: MIT

#pragma once

#include "MatrixCSR.h"
#include "Vector.h"

#include <memory>

namespace hellofem::la {

    /// Direct solve of `A x = b` by a sparse factorization.
    ///
    /// The factorization is MKL PARDISO where the build has MKL, and Eigen's
    /// SparseLU otherwise; a caller asks for a direct solve and does not name
    /// a library. It is the backend for the systems a factorization suits
    /// better than an iteration — an operator that is expensive to
    /// precondition, or a solve so small that an iteration is all overhead —
    /// and it is what a subdomain solve of the additive Schwarz preconditioner
    /// uses.
    class DirectSolver {
    public:
        /// Factorize `A`.
        explicit DirectSolver(const MatrixCSR<double>& A);

        ~DirectSolver();

        DirectSolver(const DirectSolver&) = delete;
        DirectSolver& operator=(const DirectSolver&) = delete;

        /// Solve `A x = b` with the factorization. `x` is overwritten.
        void solve(Vector<double>& x, const Vector<double>& b) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace hellofem::la
