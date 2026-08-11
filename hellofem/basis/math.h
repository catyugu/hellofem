#pragma once

#include "mdspan.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/// @brief Mathematical functions.
///
/// @note Functions in this namespace are designed to be called multiple
/// times at runtime, so their performance is critical.
namespace hellofem::basis::math {
    namespace impl {
        /// @brief Partial-pivot LU factorisation of an `n`x`n` matrix stored in a
        /// column-major buffer (LAPACK `dgetrf` semantics). Returns the 0-based
        /// pivot rows; `singular` is set when a pivot is exactly zero.
        template <std::floating_point T>
        std::vector<std::size_t> lu_pivots(T* a, std::size_t n, bool& singular)
        {
            std::vector<std::size_t> piv(n, 0);
            singular = false;
            for (std::size_t k = 0; k < n; ++k) {
                // Find pivot row (max |a[i][k]| for i in [k, n)); first max wins, as in
                // LAPACK idamax.
                std::size_t jmax = k;
                T max_abs = std::abs(a[k + k * n]);
                for (std::size_t i = k + 1; i < n; ++i) {
                    const T v = std::abs(a[i + k * n]);
                    if (v > max_abs) {
                        max_abs = v;
                        jmax = i;
                    }
                }
                if (max_abs == T(0))
                    singular = true;

                piv[k] = jmax;

                // Swap rows k and jmax (column-major: element (r, c) at r + c*n)
                if (jmax != k)
                    for (std::size_t c = 0; c < n; ++c)
                        std::swap(a[k + c * n], a[jmax + c * n]);

                // Eliminate below the pivot
                for (std::size_t i = k + 1; i < n; ++i) {
                    a[i + k * n] /= a[k + k * n];
                    for (std::size_t j = k + 1; j < n; ++j)
                        a[i + j * n] -= a[i + k * n] * a[k + j * n];
                }
            }
            return piv;
        }
    } // namespace impl

    /// @brief Compute the outer product of vectors `u` and `v`.
    /// @param u The first vector.
    /// @param v The second vector.
    /// @return The outer product. The type will be the same as `u`.
    template <typename U, typename V>
    std::pair<std::vector<typename U::value_type>, std::array<std::size_t, 2>>
    outer(const U& u, const V& v)
    {
        std::vector<typename U::value_type> result(u.size() * v.size());
        for (std::size_t i = 0; i < u.size(); ++i)
            for (std::size_t j = 0; j < v.size(); ++j)
                result[i * v.size() + j] = u[i] * v[j];
        return {std::move(result), {u.size(), v.size()}};
    }

    /// @brief Compute the cross product u x v.
    /// @param u The first vector. It must have size 3.
    /// @param v The second vector. It must have size 3.
    /// @return The cross product `u x v`. The type will be the same as `u`.
    template <typename U, typename V>
    std::array<typename U::value_type, 3> cross(const U& u, const V& v)
    {
        assert(u.size() == 3);
        assert(v.size() == 3);
        return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]};
    }

    /// @brief Compute the eigenvalues and eigenvectors of a square symmetric
    /// matrix A (input in row-major storage).
    /// @param[in] A Input matrix, row-major storage.
    /// @param[in] n Number of rows.
    /// @return Eigenvalues (0) and eigenvectors (1). The eigenvector array uses
    /// column-major storage, with each column being an eigenvector.
    /// @pre The matrix `A` must be symmetric.
    template <std::floating_point T>
    std::pair<std::vector<T>, std::vector<T>> eigh(std::span<const T> A,
        std::size_t n)
    {
        using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        Mat M(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                M(i, j) = A[i * n + j];

        Eigen::SelfAdjointEigenSolver<Mat> es(M);
        if (es.info() != Eigen::Success)
            throw std::runtime_error("Eigenvalue computation did not converge.");

        // Eigenvalues (ascending)
        std::vector<T> w(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = es.eigenvalues()(i);

        // Eigenvectors, column-major storage (each column an eigenvector)
        std::vector<T> vecs(n * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                vecs[i + j * n] = es.eigenvectors()(i, j);

        return {std::move(w), std::move(vecs)};
    }

    /// @brief Solve A X = B.
    /// @param[in] A The matrix.
    /// @param[in] B Right-hand side matrix/vector.
    /// @return A^{-1} B, in row-major storage.
    template <std::floating_point T>
    std::vector<T> solve(md::mdspan<const T, md::dextents<std::size_t, 2>> A,
        md::mdspan<const T, md::dextents<std::size_t, 2>> B)
    {
        using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Mat EA(A.extent(0), A.extent(1));
        for (std::size_t i = 0; i < A.extent(0); ++i)
            for (std::size_t j = 0; j < A.extent(1); ++j)
                EA(i, j) = A(i, j);

        Mat EB(B.extent(0), B.extent(1));
        for (std::size_t i = 0; i < B.extent(0); ++i)
            for (std::size_t j = 0; j < B.extent(1); ++j)
                EB(i, j) = B(i, j);

        Eigen::PartialPivLU<Mat> lu(EA);
        if (lu.info() != Eigen::Success)
            throw std::runtime_error("Call to solve failed.");

        const Mat X = lu.solve(EB);

        std::vector<T> rb(B.extent(0) * B.extent(1));
        for (std::size_t i = 0; i < B.extent(0); ++i)
            for (std::size_t j = 0; j < B.extent(1); ++j)
                rb[i * B.extent(1) + j] = X(i, j);

        return rb;
    }

    /// @brief Check if A is a singular matrix.
    /// @param[in] A The matrix.
    /// @return A bool indicating if the matrix is singular.
    template <std::floating_point T>
    bool is_singular(md::mdspan<const T, md::dextents<std::size_t, 2>> A)
    {
        // Copy to column-major storage
        const std::size_t n = A.extent(0);
        std::vector<T> M(n * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                M[i + j * n] = A(i, j);

        bool singular = false;
        impl::lu_pivots(M.data(), n, singular);
        return singular;
    }

    /// @brief Compute the LU decomposition of the transpose of a square matrix A.
    /// @param[in,out] A The matrix (row-major storage). On exit, holds the LU
    /// factors of A^t in place.
    /// @return The LU permutation, in prepared format (see
    /// precompute::prepare_permutation).
    template <std::floating_point T>
    std::vector<std::size_t>
    transpose_lu(std::pair<std::vector<T>, std::array<std::size_t, 2>>& A)
    {
        std::size_t dim = A.second[0];
        assert(dim == A.second[1]);

        bool singular = false;
        std::vector<std::size_t> perm
            = impl::lu_pivots(A.first.data(), dim, singular);
        if (singular)
            throw std::runtime_error("LU decomposition failed: singular matrix");

        return perm;
    }

    /// @brief Compute C = alpha A * B + beta C
    /// @param[in] A Input matrix
    /// @param[in] B Input matrix
    /// @param[in,out] C Matrix to accumulate into (read when `beta != 0`).
    /// Must be sized correctly before calling this function.
    /// @param[in] alpha Scalar multiplying `A * B`.
    /// @param[in] beta Scalar multiplying the existing contents of `C`.
    template <typename U, typename V, typename W>
    void dot(const U& A, const V& B, W&& C,
        typename std::decay_t<U>::value_type alpha = 1,
        typename std::decay_t<U>::value_type beta = 0)
    {
        assert(A.extent(1) == B.extent(0));
        assert(C.extent(0) == A.extent(0));
        assert(C.extent(1) == B.extent(1));

        // The matrices are element-level and small, so a direct loop suffices
        // (replaces the BLAS GEMM path).
        for (std::size_t i = 0; i < A.extent(0); ++i) {
            for (std::size_t j = 0; j < B.extent(1); ++j) {
                const auto C0 = C(i, j);
                C(i, j) = 0;
                auto& _C = C(i, j);
                for (std::size_t k = 0; k < A.extent(1); ++k)
                    _C += A(i, k) * B(k, j);
                _C = alpha * _C + beta * C0;
            }
        }
    }

    /// @brief Build an identity matrix.
    /// @param[in] n The number of rows/columns.
    /// @return Identity matrix using row-major storage.
    template <std::floating_point T>
    std::vector<T> eye(std::size_t n)
    {
        std::vector<T> I(n * n, 0);
        md::mdspan<T, md::dextents<std::size_t, 2>> Iview(I.data(), n, n);
        for (std::size_t i = 0; i < n; ++i)
            Iview(i, i) = 1;
        return I;
    }

    /// @brief Orthonormalise the rows of a matrix (in place).
    /// @param[in,out] wcoeffs The matrix, orthonormalised on exit.
    /// @param[in] start The row to start from. The rows before this should
    /// already be orthonormal.
    template <std::floating_point T>
    void orthogonalise(md::mdspan<T, md::dextents<std::size_t, 2>> wcoeffs,
        std::size_t start = 0)
    {
        using mdspan2_t = md::mdspan<T, md::dextents<std::size_t, 2>>;
        using cmdspan2_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;

        const std::size_t ndofs = wcoeffs.extent(0);
        const std::size_t psize = wcoeffs.extent(1);

        std::vector<T> a_vec;
        std::vector<T> update;
        for (std::size_t i = start; i < ndofs; ++i) {
            T norm = 0;
            for (std::size_t k = 0; k < psize; ++k)
                norm += wcoeffs(i, k) * wcoeffs(i, k);

            norm = std::sqrt(norm);
            if (norm < 2 * std::numeric_limits<T>::epsilon()) {
                throw std::runtime_error("Cannot orthogonalise the rows of a matrix "
                                         "with incomplete row rank");
            }

            for (std::size_t k = 0; k < psize; ++k)
                wcoeffs(i, k) /= norm;

            const std::size_t nrem = ndofs - i - 1;
            if (nrem == 0)
                continue;

            cmdspan2_t row_col(&wcoeffs(i, 0), psize, 1);
            cmdspan2_t row_row(&wcoeffs(i, 0), 1, psize);
            cmdspan2_t tail(&wcoeffs(i + 1, 0), nrem, psize);

            a_vec.resize(nrem);
            mdspan2_t a_view(a_vec.data(), nrem, 1);
            dot(tail, row_col, a_view);

            update.resize(nrem * psize);
            mdspan2_t update_view(update.data(), nrem, psize);
            dot(cmdspan2_t(a_vec.data(), nrem, 1), row_row, update_view);

            mdspan2_t tail_mut(&wcoeffs(i + 1, 0), nrem, psize);
            for (std::size_t j = 0; j < nrem; ++j)
                for (std::size_t k = 0; k < psize; ++k)
                    tail_mut(j, k) -= update_view(j, k);
        }
    }
} // namespace hellofem::basis::math
