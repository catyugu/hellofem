// hellofem::la — incomplete LU (ILU(0)) preconditioner (MKL)
// SPDX-License-Identifier: MIT

#include "ilu0.h"

#ifdef HELLOFEM_WITH_PARDISO

#include <mkl_rci.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace hellofem::la {

    struct Ilu0Preconditioner::Impl {
        MKL_INT n = 0;

        // MKL's incomplete factorization takes the matrix in the Fortran
        // convention — one-based row and column indices — so it is held in
        // that form here. `alu` carries the factors over the same pattern:
        // L below the diagonal with a unit diagonal implied, U on and above.
        std::vector<MKL_INT> ia;
        std::vector<MKL_INT> ja;
        std::vector<double> alu;

        /// `y = ILU^{-1} x`, a forward sweep of L and a backward sweep of U.
        void apply(const double* x, double* y) const
        {
            for (MKL_INT i = 0; i < n; ++i) {
                double acc = x[i];
                for (MKL_INT k = ia[static_cast<std::size_t>(i)] - 1;
                    k < ia[static_cast<std::size_t>(i) + 1] - 1; ++k) {
                    const MKL_INT j = ja[static_cast<std::size_t>(k)] - 1;
                    if (j < i)
                        acc -= alu[static_cast<std::size_t>(k)] * y[j];
                }
                y[i] = acc;
            }
            for (MKL_INT i = n - 1; i >= 0; --i) {
                double acc = y[i];
                for (MKL_INT k = ia[static_cast<std::size_t>(i)] - 1;
                    k < ia[static_cast<std::size_t>(i) + 1] - 1; ++k) {
                    const MKL_INT j = ja[static_cast<std::size_t>(k)] - 1;
                    if (j > i)
                        acc -= alu[static_cast<std::size_t>(k)] * y[j];
                }
                for (MKL_INT k = ia[static_cast<std::size_t>(i)] - 1;
                    k < ia[static_cast<std::size_t>(i) + 1] - 1; ++k)
                    if (ja[static_cast<std::size_t>(k)] - 1 == i)
                        acc /= alu[static_cast<std::size_t>(k)];
                y[i] = acc;
            }
        }
    };

    Ilu0Preconditioner::Ilu0Preconditioner(const MatrixCSR<double>& A)
        : impl_(std::make_unique<Impl>())
    {
        const MatrixCSR<double> S = A.to_scalar();
        Impl& s = *impl_;
        s.n = static_cast<MKL_INT>(S.num_owned_rows());

        s.ia.resize(static_cast<std::size_t>(s.n) + 1);
        for (MKL_INT i = 0; i <= s.n; ++i)
            s.ia[static_cast<std::size_t>(i)] = static_cast<MKL_INT>(
                S.row_ptr()[static_cast<std::size_t>(i)]) + 1;
        s.ja.resize(S.cols().size());
        for (std::size_t k = 0; k < s.ja.size(); ++k)
            s.ja[k] = S.cols()[k] + 1;
        s.alu.assign(S.values().size(), 0.0);

        MKL_INT ipar[128] = {};
        double dpar[128] = {};
        MKL_INT ierr = 0;
        dcsrilu0(&s.n, S.values().data(), s.ia.data(), s.ja.data(), s.alu.data(),
            ipar, dpar, &ierr);
        if (ierr)
            throw std::runtime_error("ilu: the MKL ILU(0) factorization failed "
                                     "with error "
                + std::to_string(ierr));
    }

    Ilu0Preconditioner::~Ilu0Preconditioner() = default;

    void Ilu0Preconditioner::apply(
        const Vector<double>& x, Vector<double>& y) const
    {
        if (y.array().size() != x.array().size())
            throw std::runtime_error("ilu: the vector size does not match the "
                                     "factorized matrix.");
        impl_->apply(x.array().data(), y.array().data());
    }

} // namespace hellofem::la

#endif // HELLOFEM_WITH_PARDISO
