// hellofem::la — incomplete LU (ILU(0)) preconditioner
// SPDX-License-Identifier: MIT

#include "preconditioner.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HELLOFEM_WITH_PARDISO
#include <mkl_rci.h>
#endif

namespace hellofem::la {

#ifdef HELLOFEM_WITH_PARDISO
    namespace {

        /// ILU(0) by MKL's `dcsrilu0`, on the scalar expansion of the matrix.
        ///
        /// MKL's incomplete factorization reads the matrix in the Fortran
        /// convention — one-based row and column indices — and rejects
        /// zero-based input with `ierr = -106`, so the scalar CSR is held
        /// one-based. `alu` carries the factors over the same pattern: L below
        /// the diagonal with a unit diagonal implied, U on and above it.
        class MklIlu0 : public Preconditioner<double> {
        public:
            explicit MklIlu0(const MatrixCSR<double>& A)
            {
                const MatrixCSR<double> S = A.to_scalar();
                n_ = static_cast<MKL_INT>(S.num_owned_rows());

                ia_.resize(static_cast<std::size_t>(n_) + 1);
                for (MKL_INT i = 0; i <= n_; ++i)
                    ia_[static_cast<std::size_t>(i)] = static_cast<MKL_INT>(
                        S.row_ptr()[static_cast<std::size_t>(i)]) + 1;
                ja_.resize(S.cols().size());
                for (std::size_t k = 0; k < ja_.size(); ++k)
                    ja_[k] = S.cols()[k] + 1;
                alu_.assign(S.values().size(), 0.0);

                MKL_INT ipar[128] = {};
                double dpar[128] = {};
                MKL_INT ierr = 0;
                dcsrilu0(&n_, S.values().data(), ia_.data(), ja_.data(),
                    alu_.data(), ipar, dpar, &ierr);
                if (ierr)
                    throw std::runtime_error(
                        "ilu: the MKL ILU(0) factorization failed with error "
                        + std::to_string(ierr));
            }

            /// Apply `y = ILU^{-1} x`: a forward sweep of L, then a backward
            /// sweep of U. MKL declares no triangular solve in this MKL's
            /// headers (the routine itself is in the library), and the two
            /// sweeps are what it would do.
            void apply(const Vector<double>& x, Vector<double>& y) const override
            {
                if (y.array().size() != x.array().size())
                    throw std::runtime_error("ilu: the vector size does not "
                                             "match the factorized matrix.");
                const double* xv = x.array().data();
                double* yv = y.array().data();
                for (MKL_INT i = 0; i < n_; ++i) {
                    double acc = xv[i];
                    for (MKL_INT k = ia_[static_cast<std::size_t>(i)] - 1;
                        k < ia_[static_cast<std::size_t>(i) + 1] - 1; ++k) {
                        const MKL_INT j = ja_[static_cast<std::size_t>(k)] - 1;
                        if (j < i)
                            acc -= alu_[static_cast<std::size_t>(k)] * yv[j];
                    }
                    yv[i] = acc;
                }
                for (MKL_INT i = n_ - 1; i >= 0; --i) {
                    double acc = yv[i];
                    for (MKL_INT k = ia_[static_cast<std::size_t>(i)] - 1;
                        k < ia_[static_cast<std::size_t>(i) + 1] - 1; ++k) {
                        const MKL_INT j = ja_[static_cast<std::size_t>(k)] - 1;
                        if (j > i)
                            acc -= alu_[static_cast<std::size_t>(k)] * yv[j];
                    }
                    for (MKL_INT k = ia_[static_cast<std::size_t>(i)] - 1;
                        k < ia_[static_cast<std::size_t>(i) + 1] - 1; ++k)
                        if (ja_[static_cast<std::size_t>(k)] - 1 == i)
                            acc /= alu_[static_cast<std::size_t>(k)];
                    yv[i] = acc;
                }
            }

        private:
            MKL_INT n_ = 0;
            std::vector<MKL_INT> ia_;
            std::vector<MKL_INT> ja_;
            std::vector<double> alu_;
        };

    } // namespace
#endif

    std::shared_ptr<Preconditioner<double>> make_ilu_preconditioner(
        const MatrixCSR<double>& A)
    {
#ifdef HELLOFEM_WITH_PARDISO
        return std::make_shared<MklIlu0>(A);
#else
        return std::make_shared<IluPreconditioner<double>>(A);
#endif
    }

} // namespace hellofem::la
