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
    /// The matrix is read as the unsymmetric CSR it is. An assembled system
    /// with Dirichlet rows is not symmetric — those rows are zeroed and their
    /// diagonal set, while the columns of the neighbouring rows keep their
    /// entries — so the factorization is the general LU, which reads the whole
    /// matrix. A symmetric factorization reads one triangle only and is
    /// therefore both cheaper and a correctness hazard here.
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
