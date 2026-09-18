// hellofem::la — direct sparse factorization (MKL PARDISO)
// SPDX-License-Identifier: MIT

#include "pardiso.h"

#ifdef HELLOFEM_WITH_PARDISO

#include <mkl_pardiso.h>

#include <algorithm>
#include <cstring>
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

    } // namespace

    struct PardisoSolver::Impl {
        // PARDISO's handle and parameter block. Both are zero-initialized
        // instead of taken from `pardisoinit`: the parameter block is filled
        // explicitly below, and a zeroed handle is what the first call needs.
        _MKL_DSS_HANDLE_t pt[64] = {};
        MKL_INT iparm[64] = {};
        MKL_INT mtype = 11; // real unsymmetric: LU factorization
        MKL_INT maxfct = 1;
        MKL_INT mnum = 1;
        MKL_INT nrhs = 1;
        MKL_INT msglvl = 0;
        MKL_INT n = 0;

        // The matrix in the CSR PARDISO reads: zero-based row and column
        // indices (iparm[34]), values in `a`.
        std::vector<MKL_INT> ia;
        std::vector<MKL_INT> ja;
        std::vector<double> a;

        /// Run one phase over the current arrays. The right-hand side is only
        /// read in the solve phase; the other phases get the same array, which
        /// the interface requires to be a valid pointer.
        MKL_INT run(MKL_INT phase, double* b, double* x)
        {
            MKL_INT error = 0;
            pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a.data(), ia.data(),
                ja.data(), nullptr, &nrhs, iparm, &msglvl, b, x, &error);
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

        const MatrixCSR<double> S = A.to_scalar();
        s.n = S.num_owned_rows();
        s.ia.assign(S.row_ptr().begin(), S.row_ptr().end());
        s.ja.assign(S.cols().begin(), S.cols().end());
        s.a.assign(S.values().begin(), S.values().end());

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
