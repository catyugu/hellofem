// hellofem::la — direct sparse solve of an assembled system
// SPDX-License-Identifier: MIT

#include "direct.h"

#ifdef HELLOFEM_WITH_PARDISO
#include "pardiso.h"
#else
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hellofem::la {

#ifdef HELLOFEM_WITH_PARDISO

    struct DirectSolver::Impl {
        PardisoSolver pardiso;

        explicit Impl(const MatrixCSR<double>& A)
            : pardiso(A)
        {
        }
    };

#else

    struct DirectSolver::Impl {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;

        explicit Impl(const MatrixCSR<double>& A)
        {
            const MatrixCSR<double> S = A.to_scalar();
            const std::int32_t nrows = S.num_owned_rows();
            Eigen::SparseMatrix<double> E(nrows, nrows);
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(static_cast<std::size_t>(S.cols().size()));
            for (std::int32_t r = 0; r < nrows; ++r)
                for (std::int64_t j = S.row_ptr()[r]; j < S.row_ptr()[r + 1]; ++j)
                    triplets.emplace_back(r,
                        S.cols()[static_cast<std::size_t>(j)],
                        S.values()[static_cast<std::size_t>(j)]);
            E.setFromTriplets(triplets.begin(), triplets.end());
            lu.compute(E);
            if (lu.info() != Eigen::Success)
                throw std::runtime_error("direct: the factorization failed.");
        }
    };

#endif

    DirectSolver::DirectSolver(const MatrixCSR<double>& A)
        : impl_(std::make_unique<Impl>(A))
    {
    }

    DirectSolver::~DirectSolver() = default;

    void DirectSolver::solve(Vector<double>& x, const Vector<double>& b) const
    {
#ifdef HELLOFEM_WITH_PARDISO
        impl_->pardiso.solve(x, b);
#else
        const std::size_t n = b.array().size();
        if (x.array().size() != n)
            throw std::runtime_error("direct: the vector size does not match "
                                     "the factorized matrix.");
        Eigen::Map<const Eigen::VectorXd> be(b.array().data(),
            static_cast<Eigen::Index>(n));
        Eigen::Map<Eigen::VectorXd> xe(x.array().data(),
            static_cast<Eigen::Index>(n));
        xe = impl_->lu.solve(be);
        if (impl_->lu.info() != Eigen::Success)
            throw std::runtime_error("direct: the solve failed.");
#endif
    }

} // namespace hellofem::la
