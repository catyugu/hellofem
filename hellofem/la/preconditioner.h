// hellofem::la — preconditioners
// SPDX-License-Identifier: MIT

#pragma once

#include "MatrixCSR.h"

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

    /// Pointwise (scalar) Jacobi preconditioner: `y[i] = x[i] / A(i,i)`.
    /// Requires a scalar matrix (block size `{1,1}`).
    template <typename T>
    class JacobiPreconditioner : public Preconditioner<T> {
    public:
        /// Build from a matrix.
        /// @param[in] A The matrix; must have block size `{1,1}`.
        explicit JacobiPreconditioner(const MatrixCSR<T>& A)
        {
            if (A.block_size() != std::array<int, 2> {1, 1}) {
                throw std::runtime_error(
                    "Jacobi preconditioner requires a scalar matrix.");
            }

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

        /// Apply `y = inv(diag(A)) * x`.
        void apply(const Vector<T>& x, Vector<T>& y) const override
        {
            const auto& xa = x.array();
            auto& ya = y.array();
            assert(ya.size() >= _inv_diag.size());
            for (std::size_t i = 0; i < _inv_diag.size(); ++i)
                ya[i] = _inv_diag[i] * xa[i];
        }

    private:
        // Inverse of the diagonal
        std::vector<T> _inv_diag;
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
