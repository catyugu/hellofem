// hellofem::la — preconditioners
// SPDX-License-Identifier: MIT

#pragma once

#include "MatrixCSR.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hellofem::la {

    /// Abstract preconditioner: `y := P x`, where P is an approximate
    /// inverse of the operator.
    template <typename T>
    class Preconditioner {
    public:
        /// Destructor
        virtual ~Preconditioner() = default;

        /// Apply the preconditioner.
        /// @param[in] x Input vector.
        /// @param[out] y Result `P x`.
        virtual void apply(const Vector<T>& x, Vector<T>& y) const = 0;
    };

    /// Jacobi preconditioner: pointwise (scalar) for bs == {1,1}, or
    /// block-Jacobi (per-block diagonal inverse) for blocked matrices.
    ///
    /// `y[i] = x[i] / A(i,i)` for scalar matrices; for blocked matrices
    /// each block row is multiplied by the inverse of its diagonal block.
    template <typename T>
    class JacobiPreconditioner : public Preconditioner<T> {
    public:
        /// Build from a matrix.
        /// @param[in] A The matrix.
        explicit JacobiPreconditioner(const MatrixCSR<T>& A)
        {
            const auto [bs0, bs1] = A.block_size();
            if (bs0 == 1 and bs1 == 1) {
                _bs = 1;
                const std::int32_t nrows = A.num_owned_rows();
                _inv_diag.resize(nrows);
                const auto& row_ptr = A.row_ptr();
                const auto& cols = A.cols();
                const auto& values = A.values();
                for (std::int32_t r = 0; r < nrows; ++r) {
                    auto cit0 = cols.begin() + row_ptr[r];
                    auto cit1 = cols.begin() + row_ptr[r + 1];
                    auto it = std::lower_bound(cit0, cit1, r);
                    if (it == cit1 or *it != r)
                        throw std::runtime_error("Matrix has no diagonal entry.");
                    const std::size_t d = std::ranges::distance(cols.begin(), it);
                    _inv_diag[r] = T(1) / values[d];
                }
            }
            else if (bs0 == bs1) {
                // Block-Jacobi: invert each bs x bs diagonal block.
                _bs = bs0;
                const auto S = A.to_scalar();
                const std::int32_t nrows_b = S.num_owned_rows();
                const std::int32_t nblocks = nrows_b / bs0;
                _inv_diag.resize(static_cast<std::size_t>(nblocks) * bs0 * bs1,
                    0);
                const auto& row_ptr = S.row_ptr();
                const auto& cols = S.cols();
                const auto& values = S.values();
                for (std::int32_t b = 0; b < nblocks; ++b) {
                    // Diagonal block entries: S[b*bs+i0][b*bs+i1].
                    std::vector<T> D(static_cast<std::size_t>(bs0 * bs1), 0);
                    bool ok = true;
                    for (int i0 = 0; i0 < bs0; ++i0) {
                        const std::int32_t r = b * bs0 + i0;
                        auto cit0 = cols.begin() + row_ptr[r];
                        auto cit1 = cols.begin() + row_ptr[r + 1];
                        for (int i1 = 0; i1 < bs1; ++i1) {
                            auto it = std::lower_bound(cit0, cit1, b * bs1 + i1);
                            if (it == cit1 or *it != b * bs1 + i1) {
                                ok = false;
                                break;
                            }
                            const std::size_t d
                                = std::ranges::distance(cols.begin(), it);
                            D[static_cast<std::size_t>(i0 * bs1 + i1)] = values[d];
                        }
                        if (!ok)
                            break;
                    }
                    if (!ok)
                        throw std::runtime_error(
                            "Blocked matrix has no diagonal block.");
                    // Invert the bs x bs block (generic 1/2/3).
                    std::vector<T> Din(static_cast<std::size_t>(bs0 * bs1), 0);
                    _inv_block(D, bs0, Din);
                    for (int i0 = 0; i0 < bs0; ++i0)
                        for (int i1 = 0; i1 < bs1; ++i1)
                            _inv_diag[static_cast<std::size_t>(b) * bs0 * bs1
                                + i0 * bs1 + i1]
                                = Din[static_cast<std::size_t>(i0 * bs1 + i1)];
                }
            }
            else {
                throw std::runtime_error(
                    "Jacobi preconditioner requires a square block size.");
            }
        }

        /// Apply `y = inv(diag(A)) * x`.
        void apply(const Vector<T>& x, Vector<T>& y) const override
        {
            const auto& xa = x.array();
            auto& ya = y.array();
            if (_bs == 1) {
                assert(ya.size() >= _inv_diag.size());
                for (std::size_t i = 0; i < _inv_diag.size(); ++i)
                    ya[i] = _inv_diag[i] * xa[i];
            }
            else {
                // Block-Jacobi: y[block*i0] = sum_i1 Din[i0][i1] x[block*i1].
                // `_inv_diag` holds one bs*bs block inverse per scalar block.
                const std::size_t nblocks = _inv_diag.size() / (_bs * _bs);
                for (std::size_t b = 0; b < nblocks; ++b)
                    for (int i0 = 0; i0 < _bs; ++i0) {
                        T acc {0};
                        for (int i1 = 0; i1 < _bs; ++i1)
                            acc += _inv_diag[b * static_cast<std::size_t>(_bs * _bs)
                                      + static_cast<std::size_t>(i0 * _bs + i1)]
                                * xa[b * static_cast<std::size_t>(_bs) + i1];
                        ya[b * static_cast<std::size_t>(_bs) + i0] = acc;
                    }
            }
        }

    private:
        /// Invert a small dense matrix (bs x bs) into `Din`.
        static void _inv_block(const std::vector<T>& D, int n,
            std::vector<T>& Din)
        {
            switch (n) {
            case 1:
                Din[0] = T(1) / D[0];
                break;
            case 2: {
                const T det = D[0] * D[3] - D[1] * D[2];
                Din[0] = D[3] / det;
                Din[1] = -D[1] / det;
                Din[2] = -D[2] / det;
                Din[3] = D[0] / det;
                break;
            }
            case 3: {
                // Inverse via cofactors (only used for bs=3 blocks).
                const T a = D[0], b = D[1], c = D[2];
                const T d = D[3], e = D[4], f = D[5];
                const T g = D[6], h = D[7], i = D[8];
                const T det = a * (e * i - f * h) - b * (d * i - f * g)
                    + c * (d * h - e * g);
                Din[0] = (e * i - f * h) / det;
                Din[1] = (c * h - b * i) / det;
                Din[2] = (b * f - c * e) / det;
                Din[3] = (f * g - d * i) / det;
                Din[4] = (a * i - c * g) / det;
                Din[5] = (c * d - a * f) / det;
                Din[6] = (d * h - e * g) / det;
                Din[7] = (b * g - a * h) / det;
                Din[8] = (a * e - b * d) / det;
                break;
            }
            default:
                throw std::runtime_error(
                    "Block-Jacobi supports block sizes up to 3.");
            }
        }

        // Block size (1 for pointwise Jacobi)
        int _bs = 1;

        // Inverse diagonal: pointwise (bs=1) or per-block (bs>1), stored
        // as [block][i0][i1] row-major.
        std::vector<T> _inv_diag;
    };

    /// Incomplete LU (ILU(0)) preconditioner via Eigen's `IncompleteLUT`,
    /// operating on the scalar expansion of the matrix.
    template <typename T>
    class IluPreconditioner : public Preconditioner<T> {
    public:
        /// Build the ILU factorization from a matrix.
        /// @param[in] A The matrix to precondition.
        explicit IluPreconditioner(const MatrixCSR<T>& A)
        {
            const auto S = A.to_scalar();
            const std::int32_t nrows = S.num_owned_rows();

            // Fill an Eigen sparse matrix from the scalar CSR.
            Eigen::SparseMatrix<T> E(nrows, nrows);
            std::vector<Eigen::Triplet<T>> triplets;
            triplets.reserve(static_cast<std::size_t>(S.cols().size()));
            for (std::int32_t r = 0; r < nrows; ++r)
                for (std::int64_t j = S.row_ptr()[r]; j < S.row_ptr()[r + 1]; ++j)
                    triplets.emplace_back(r, S.cols()[static_cast<std::size_t>(j)],
                        S.values()[static_cast<std::size_t>(j)]);
            E.setFromTriplets(triplets.begin(), triplets.end());

            _ilu.compute(E);
        }

        /// Apply `y = ILU^{-1} x`.
        void apply(const Vector<T>& x, Vector<T>& y) const override
        {
            Eigen::Map<const Eigen::VectorX<T>> xe(
                const_cast<T*>(x.array().data()), static_cast<Eigen::Index>(x.array().size()));
            Eigen::Map<Eigen::VectorX<T>> ye(y.array().data(),
                static_cast<Eigen::Index>(y.array().size()));
            ye = _ilu.solve(xe);
        }

    private:
        // Eigen ILU factorization
        Eigen::IncompleteLUT<T> _ilu;
    };

    /// Algebraic multigrid preconditioner via amgcl. The matrix is
    /// copied into a scalar CSR for amgcl; blocked matrices are expanded
    /// to scalar entries first.
    template <typename T>
    class AmgPreconditioner : public Preconditioner<T> {
    public:
        /// Build the AMG hierarchy from a matrix.
        /// @param[in] A The matrix to precondition.
        explicit AmgPreconditioner(const MatrixCSR<T>& A)
        {
            using Backend = amgcl::backend::builtin<T>;
            using Matrix = typename Backend::matrix;

            const auto [bs0, bs1] = A.block_size();
            if (bs0 == 1 and bs1 == 1) {
                // Scalar matrix: hand amgcl the CSR arrays directly
                // (amgcl's crs view constructor deep-copies).
                auto M = std::make_shared<Matrix>(
                    static_cast<std::size_t>(A.num_owned_rows()),
                    static_cast<std::size_t>(A.index_map(1)->size_local()),
                    A.row_ptr(), A.cols(), A.values());
                _amg = std::make_shared<AMG>(M);
            }
            else {
                // Blocked matrix: expand blocks to scalar CSR.
                const std::size_t nrows = static_cast<std::size_t>(
                    A.num_owned_rows());
                const std::size_t nrows_b = nrows * bs0;
                const std::size_t ncols_b = static_cast<std::size_t>(
                                                A.index_map(1)->size_local())
                    * bs1;
                std::vector<std::int64_t> row_ptr2(nrows_b + 1, 0);
                std::vector<std::int32_t> cols2;
                std::vector<T> values2;
                cols2.reserve(A.cols().size() * bs1);
                values2.reserve(A.values().size());

                for (std::size_t r = 0; r < nrows; ++r) {
                    for (int i0 = 0; i0 < bs0; ++i0) {
                        row_ptr2[r * bs0 + i0 + 1] = row_ptr2[r * bs0 + i0];
                        for (std::int64_t j = A.row_ptr()[r]; j < A.row_ptr()[r + 1];
                            ++j) {
                            for (int i1 = 0; i1 < bs1; ++i1) {
                                cols2.push_back(A.cols()[j] * bs1 + i1);
                                values2.push_back(A.values()[j * bs0 * bs1
                                    + i0 * bs1 + i1]);
                                ++row_ptr2[r * bs0 + i0 + 1];
                            }
                        }
                    }
                }

                auto M = std::make_shared<Matrix>(nrows_b, ncols_b, row_ptr2,
                    cols2, values2);
                _amg = std::make_shared<AMG>(M);
            }
        }

        /// Apply one AMG V-cycle: `y = AMG(x)` (clears y internally).
        void apply(const Vector<T>& x, Vector<T>& y) const override
        {
            _amg->apply(x.array(), y.array());
        }

    private:
        using Backend = amgcl::backend::builtin<T>;
        using AMG = amgcl::amg<Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::damped_jacobi>;

        // The AMG hierarchy
        std::shared_ptr<AMG> _amg;
    };

} // namespace hellofem::la
