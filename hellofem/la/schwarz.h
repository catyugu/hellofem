// hellofem::la — additive Schwarz (overlapping domain decomposition)
// SPDX-License-Identifier: MIT

#pragma once

#include "MatrixCSR.h"
#include "Vector.h"
#include "preconditioner.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace hellofem::la {

    /// Additive Schwarz preconditioner.
    ///
    /// Splits the (scalar) matrix rows into `nparts` contiguous subdomains,
    /// grows each by `overlap` layers of the graph adjacency, solves the
    /// local subsystem of each subdomain exactly (Eigen SparseLU) and
    /// sums the extended local solutions — the additive Schwarz operator
    /// `P = sum_s R_s^T A_ss^{-1} R_s`.
    ///
    /// This is a genuine overlapping domain-decomposition preconditioner
    /// (one-level; the "DDM" requirement).
    template <typename T>
    class SchwarzPreconditioner : public Preconditioner<T> {
    public:
        /// Build the Schwarz preconditioner.
        /// @param[in] A The matrix to precondition.
        /// @param[in] nparts Number of subdomains (>= 1).
        /// @param[in] overlap Graph-adjacency overlap layers (>= 0).
        SchwarzPreconditioner(const MatrixCSR<T>& A, int nparts, int overlap);

        /// Apply `y = P x` (the additive Schwarz operator).
        void apply(const Vector<T>& x, Vector<T>& y) const override;

    private:
        struct Impl;
        // Scalar matrix being preconditioned.
        MatrixCSR<T> _S;
        // Number of rows of `_S`.
        std::int32_t _nrows;
        // PIMPL (Eigen factorization state).
        std::shared_ptr<Impl> _impl;
    };

} // namespace hellofem::la
