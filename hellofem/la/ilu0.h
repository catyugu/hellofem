// hellofem::la — incomplete LU (ILU(0)) preconditioner (MKL)
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HELLOFEM_WITH_PARDISO

#include "Vector.h"
#include "preconditioner.h"

#include <memory>

namespace hellofem::la {

    /// Incomplete LU (ILU(0)) preconditioner by MKL's `dcsrilu0`, operating on
    /// the scalar expansion of the matrix. This is the la layer's ILU where the
    /// build has MKL — the same factorization as `IluPreconditioner`, computed
    /// by the backend the la layer otherwise prefers — and `IluPreconditioner`
    /// (Eigen's `IncompleteLUT`) is what `KrylovSolver` falls back to without it.
    class Ilu0Preconditioner : public Preconditioner<double> {
    public:
        /// Build the incomplete factorization of `A`.
        explicit Ilu0Preconditioner(const MatrixCSR<double>& A);

        ~Ilu0Preconditioner() override;

        Ilu0Preconditioner(const Ilu0Preconditioner&) = delete;
        Ilu0Preconditioner& operator=(const Ilu0Preconditioner&) = delete;

        /// Apply `y = ILU^{-1} x`.
        void apply(const Vector<double>& x, Vector<double>& y) const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace hellofem::la

#endif // HELLOFEM_WITH_PARDISO
