// hellofem::la — direct sparse solve of an assembled system
// SPDX-License-Identifier: MIT

#include "direct.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HELLOFEM_WITH_PARDISO
#include <mkl_pardiso.h>
#include <spdlog/spdlog.h>
#else
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

namespace hellofem::la {

#ifdef HELLOFEM_WITH_PARDISO
    namespace {

        [[noreturn]] void fail(const char* phase, MKL_INT error)
        {
            throw std::runtime_error(std::string("direct: PARDISO ") + phase
                + " failed with error " + std::to_string(error));
        }

        /// A matrix in the arrays PARDISO reads: zero-based row pointers and
        /// column indices, values in the same order.
        struct Csr {
            std::vector<MKL_INT> ia;
            std::vector<MKL_INT> ja;
            std::vector<double> a;

            std::size_t size() const { return ia.empty() ? 0 : ia.size() - 1; }
        };

        Csr to_csr(const MatrixCSR<double>& S)
        {
            Csr c;
            c.ia.assign(S.row_ptr().begin(), S.row_ptr().end());
            c.ja.assign(S.cols().begin(), S.cols().end());
            c.a.assign(S.values().begin(), S.values().end());
            return c;
        }

        /// Whether `c` is symmetric, structurally and numerically.
        ///
        /// PARDISO's symmetric factorization reads one triangle and
        /// reconstructs the other, so a wrong verdict does not fail — it
        /// silently factorizes half the system. The test therefore compares
        /// every entry with its transpose rather than a cheaper proxy.
        ///
        /// The comparison allows for the round-off an assembly leaves behind:
        /// the two halves of an element tensor are evaluated by different
        /// arithmetic, so they agree to a few ulps of the largest entry and
        /// not bit for bit. The allowance sits orders of magnitude above that
        /// and orders of magnitude below the asymmetry of an operator that is
        /// genuinely unsymmetric, so the two are not confusable.
        bool is_symmetric(const Csr& c)
        {
            const std::size_t n = c.size();
            const std::size_t nnz = c.ja.size();
            if (nnz == 0)
                return true;

            // The transpose, by counting sort of the column indices.
            std::vector<MKL_INT> tia(n + 1, 0);
            for (const MKL_INT j : c.ja)
                ++tia[static_cast<std::size_t>(j) + 1];
            for (std::size_t i = 0; i < n; ++i)
                tia[i + 1] += tia[i];
            std::vector<MKL_INT> tja(nnz);
            std::vector<double> ta(nnz);
            std::vector<MKL_INT> next(tia.begin(), tia.end() - 1);
            for (std::size_t i = 0; i < n; ++i)
                for (MKL_INT k = c.ia[i]; k < c.ia[i + 1]; ++k) {
                    const std::size_t t = static_cast<std::size_t>(
                        next[static_cast<std::size_t>(c.ja[k])]++);
                    tja[t] = static_cast<MKL_INT>(i);
                    ta[t] = c.a[static_cast<std::size_t>(k)];
                }

            double scale = 0;
            for (const double v : c.a)
                scale = std::max(scale, std::abs(v));
            const double tolerance
                = 64.0 * std::numeric_limits<double>::epsilon() * scale;

            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t len0
                    = static_cast<std::size_t>(c.ia[i + 1] - c.ia[i]);
                const std::size_t len1
                    = static_cast<std::size_t>(tia[i + 1] - tia[i]);
                if (len0 != len1)
                    return false;
                for (std::size_t k = 0; k < len0; ++k) {
                    const std::size_t k0 = static_cast<std::size_t>(c.ia[i]) + k;
                    const std::size_t k1 = static_cast<std::size_t>(tia[i]) + k;
                    if (c.ja[k0] != tja[k1]
                        or std::abs(c.a[k0] - ta[k1]) > tolerance)
                        return false;
                }
            }
            return true;
        }

        /// The upper triangle (column >= row) of a CSR with sorted columns,
        /// which is the triangle PARDISO's symmetric factorization reads.
        Csr upper_triangle(const Csr& c)
        {
            Csr t;
            const std::size_t n = c.size();
            t.ia.assign(n + 1, 0);
            for (std::size_t i = 0; i < n; ++i) {
                for (MKL_INT k = c.ia[i]; k < c.ia[i + 1]; ++k)
                    if (c.ja[static_cast<std::size_t>(k)]
                        >= static_cast<MKL_INT>(i)) {
                        t.ja.push_back(c.ja[static_cast<std::size_t>(k)]);
                        t.a.push_back(c.a[static_cast<std::size_t>(k)]);
                    }
                t.ia[i + 1] = static_cast<MKL_INT>(t.ja.size());
            }
            return t;
        }

        /// Direct factorization of a scalar CSR by MKL PARDISO.
        ///
        /// The factorization is the one the matrix calls for: a symmetric
        /// matrix is factorized from its upper triangle alone as the LLT
        /// (mtype 2), anything else whole as the LU (mtype 11). Both triangles
        /// must not be handed to the symmetric factorization — it rejects that
        /// input rather than ignoring the duplicate half.
        class Pardiso {
        public:
            explicit Pardiso(const MatrixCSR<double>& A)
            {
                // iparm[0] = 1 takes the values below rather than PARDISO's own
                // defaults, so every setting that matters is named here.
                // iparm[1] is the fill-in reducing reordering, iparm[34] the
                // zero-based CSR of MatrixCSR, iparm[9] the pivot perturbation
                // of the LU.
                iparm_[0] = 1;
                iparm_[1] = 2; // nested dissection (METIS)
                iparm_[9] = 13;
                iparm_[34] = 1; // zero-based row and column indices

                const Csr full = to_csr(A.to_scalar());
                n_ = static_cast<MKL_INT>(full.size());
                const bool symmetric = is_symmetric(full);
                csr_ = symmetric ? upper_triangle(full) : full;
                mtype_ = symmetric ? 2 : 11;

                spdlog::debug("direct: pardiso n = {}, nnz = {}, mtype = {}",
                    n_, static_cast<long long>(csr_.ja.size()),
                    static_cast<long long>(mtype_));

                // Analysis of the pattern, then the numerical factorization.
                if (const MKL_INT error = run(11, nullptr, nullptr))
                    fail("analysis", error);
                if (const MKL_INT error = run(22, nullptr, nullptr))
                    fail("factorization", error);
            }

            ~Pardiso()
            {
                // Release the internal memory, then the handle itself.
                run(-1, nullptr, nullptr);
                run(0, nullptr, nullptr);
            }

            Pardiso(const Pardiso&) = delete;
            Pardiso& operator=(const Pardiso&) = delete;

            void solve(Vector<double>& x, const Vector<double>& b)
            {
                if (static_cast<MKL_INT>(x.array().size()) != n_)
                    throw std::runtime_error("direct: the vector size does not "
                                             "match the factorized matrix.");
                // The right-hand side and the solution are separate arrays: the
                // solve keeps the right-hand side unchanged (iparm[6] = 0) and
                // aliasing the two corrupts the result.
                std::vector<double> rhs(b.array().begin(), b.array().end());
                if (const MKL_INT error
                    = run(33, rhs.data(), x.array().data()))
                    fail("solve", error);
            }

        private:
            /// Run one phase over the current arrays. The right-hand side is
            /// only read in the solve phase; the other phases get the same
            /// array, which the interface requires to be a valid pointer.
            MKL_INT run(MKL_INT phase, double* b, double* x)
            {
                MKL_INT error = 0;
                pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase, &n_, csr_.a.data(),
                    csr_.ia.data(), csr_.ja.data(), nullptr, &nrhs_, iparm_,
                    &msglvl_, b, x, &error);
                return error;
            }

            // PARDISO's handle and parameter block. Both are zero-initialized
            // instead of taken from `pardisoinit`: the parameter block is
            // filled explicitly above, and a zeroed handle is what the first
            // call needs.
            _MKL_DSS_HANDLE_t pt_[64] = {};
            MKL_INT iparm_[64] = {};
            MKL_INT mtype_ = 11;
            MKL_INT maxfct_ = 1;
            MKL_INT mnum_ = 1;
            MKL_INT nrhs_ = 1;
            MKL_INT msglvl_ = 0;
            MKL_INT n_ = 0;

            /// The matrix PARDISO reads, held for the lifetime of the handle:
            /// the factorization phases take it by pointer, not by value.
            Csr csr_;
        };

    } // namespace

    struct DirectSolver::Impl {
        Pardiso pardiso;

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
