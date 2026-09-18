// hellofem::la — direct sparse factorization (MKL PARDISO)
// SPDX-License-Identifier: MIT

#include "pardiso.h"

#ifdef HELLOFEM_WITH_PARDISO

#include <mkl_pardiso.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hellofem::la {
    namespace {

        [[noreturn]] void fail(const char* phase, MKL_INT error)
        {
            throw std::runtime_error(std::string("pardiso: ") + phase
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
                    const std::size_t t
                        = static_cast<std::size_t>(next[static_cast<std::size_t>(c.ja[k])]++);
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

    } // namespace

    struct PardisoSolver::Impl {
        // PARDISO's handle and parameter block. Both are zero-initialized
        // instead of taken from `pardisoinit`: the parameter block is filled
        // explicitly below, and a zeroed handle is what the first call needs.
        _MKL_DSS_HANDLE_t pt[64] = {};
        MKL_INT iparm[64] = {};
        MKL_INT mtype = 11;
        MKL_INT maxfct = 1;
        MKL_INT mnum = 1;
        MKL_INT nrhs = 1;
        MKL_INT msglvl = 0;
        MKL_INT n = 0;

        /// The matrix PARDISO reads, held for the lifetime of the handle: the
        /// factorization phases take it by pointer, not by value.
        Csr csr;

        /// Run one phase over the current arrays. The right-hand side is only
        /// read in the solve phase; the other phases get the same array, which
        /// the interface requires to be a valid pointer.
        MKL_INT run(MKL_INT phase, double* b, double* x)
        {
            MKL_INT error = 0;
            pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, csr.a.data(),
                csr.ia.data(), csr.ja.data(), nullptr, &nrhs, iparm, &msglvl,
                b, x, &error);
            return error;
        }
    };

    PardisoSolver::PardisoSolver(const MatrixCSR<double>& A)
        : impl_(std::make_unique<Impl>())
    {
        Impl& s = *impl_;

        // iparm[0] = 1 takes the values below rather than PARDISO's own
        // defaults, so every setting that matters is named here. iparm[1] is
        // the fill-in reducing reordering, iparm[34] the zero-based CSR of
        // MatrixCSR, iparm[9] the pivot perturbation of the LU.
        s.iparm[0] = 1;
        s.iparm[1] = 2; // nested dissection (METIS)
        s.iparm[9] = 13;
        s.iparm[34] = 1; // zero-based row and column indices

        const Csr full = to_csr(A.to_scalar());
        s.n = static_cast<MKL_INT>(full.size());
        const bool symmetric = is_symmetric(full);

        // A symmetric matrix is factorized from its upper triangle alone, so
        // the factorization reads and stores half the matrix; an unsymmetric
        // one is factorized whole. Both triangles of a symmetric matrix must
        // not be handed to the symmetric factorization: it rejects that input
        // rather than ignoring the duplicate half.
        s.csr = symmetric ? upper_triangle(full) : full;
        s.mtype = symmetric ? 2 : 11;

        spdlog::debug("pardiso: n = {}, nnz = {}, mtype = {}", s.n,
            static_cast<long long>(s.csr.ja.size()),
            static_cast<long long>(s.mtype));

        // Analysis of the pattern, then the numerical factorization.
        if (const MKL_INT error = s.run(11, nullptr, nullptr))
            fail("analysis", error);
        if (const MKL_INT error = s.run(22, nullptr, nullptr))
            fail("factorization", error);
    }

    PardisoSolver::~PardisoSolver()
    {
        // Release the internal memory, then the handle itself.
        if (impl_) {
            impl_->run(-1, nullptr, nullptr);
            impl_->run(0, nullptr, nullptr);
        }
    }

    void PardisoSolver::solve(Vector<double>& x, const Vector<double>& b) const
    {
        if (static_cast<MKL_INT>(x.array().size()) != impl_->n)
            throw std::runtime_error("pardiso: the vector size does not match "
                                     "the factorized matrix.");
        // The right-hand side and the solution are separate arrays: the
        // solve keeps the right-hand side unchanged (iparm[6] = 0) and
        // aliasing the two corrupts the result.
        std::vector<double> rhs(b.array().begin(), b.array().end());
        if (const MKL_INT error
            = impl_->run(33, rhs.data(), x.array().data()))
            fail("solve", error);
    }

} // namespace hellofem::la

#endif // HELLOFEM_WITH_PARDISO
