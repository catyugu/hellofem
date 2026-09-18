// hellofem::la — direct sparse factorization (MKL PARDISO)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HELLOFEM_WITH_PARDISO

#include "MatrixCSR.h"
#include "Vector.h"

#include <memory>

namespace hellofem::la {

    /// Direct factorization of `A` by MKL PARDISO, on the scalar expansion of
    /// a blocked matrix. This is the la layer's direct backend: where a
    /// factorization is wanted (see `KrylovSolver`), it replaces the Eigen
    /// factorizations.
    ///
    /// The factorization is the one the matrix calls for, and the caller names
    /// none of them. An assembled system whose boundary conditions are imposed
    /// by zeroing the constrained rows and columns and setting their diagonal
    /// is symmetric, and its principal submatrix is what the physics makes it —
    /// definite for the diffusion, mass and elasticity operators — so it is
    /// factorized as the symmetric positive definite LLT of one triangle.
    /// Anything else is factorized as the general LU of the whole matrix.
    class PardisoSolver {
    public:
        /// Analyze the pattern of `A` and factorize it.
        explicit PardisoSolver(const MatrixCSR<double>& A);

        ~PardisoSolver();

        PardisoSolver(const PardisoSolver&) = delete;
        PardisoSolver& operator=(const PardisoSolver&) = delete;

        /// Solve `A x = b` with the factorization.
        void solve(Vector<double>& x, const Vector<double>& b) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace hellofem::la

#endif // HELLOFEM_WITH_PARDISO
