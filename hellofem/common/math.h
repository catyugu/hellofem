// hellofem::math — small dense matrix/vector helpers
// SPDX-License-Identifier: MIT

#pragma once

#include "types.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace hellofem::math {

    /// Cross product of two length-3 vectors.
    template <typename U, typename V>
        requires scalar<typename U::value_type>
                 && std::same_as<typename U::value_type, typename V::value_type>
    constexpr std::array<typename U::value_type, 3> cross(const U& u, const V& v)
    {
        assert(u.size() == 3);
        assert(v.size() == 3);
        return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]};
    }

    /// Kahan's difference of products a*d - b*c with fused multiply-adds
    /// (absolute error bounded by ~1.5 ulps).
    template <std::floating_point T>
    T difference_of_products(T a, T b, T c, T d) noexcept
    {
        T w = b * c;
        T err = std::fma(-b, c, w);
        T diff = std::fma(a, d, -w);
        return diff + err;
    }

    /// Determinant of a 1x1, 2x2 or 3x3 matrix, using Kahan's method for
    /// the 3x3 case.
    template <typename U>
        requires MDSpanRank2<U> && std::floating_point<typename U::value_type>
    auto det(U A)
    {
        assert(A.extent(0) == A.extent(1));

        using value_type = typename U::value_type;
        const int nrows = static_cast<int>(A.extent(0));
        switch (nrows) {
        case 1:
            return A(0, 0);
        case 2:
            return difference_of_products(A(0, 0), A(0, 1), A(1, 0), A(1, 1));
        case 3: {
            value_type w0
                = difference_of_products(A(1, 1), A(1, 2), A(2, 1), A(2, 2));
            value_type w1
                = difference_of_products(A(1, 0), A(1, 2), A(2, 0), A(2, 2));
            value_type w2
                = difference_of_products(A(1, 0), A(1, 1), A(2, 0), A(2, 1));
            value_type w3 = difference_of_products(A(0, 0), A(0, 1), w1, w0);
            return std::fma(A(0, 2), w2, w3);
        }
        default:
            throw std::runtime_error(
                std::format("math::det is not implemented for {}x{} matrices.",
                            A.extent(0), A.extent(1)));
        }
    }

    /// In-place inverse of a 1x1, 2x2 or 3x3 matrix into preallocated `B`.
    /// Does not check invertibility.
    template <typename U, typename V>
        requires MDSpanRank2<U> && MDSpanRank2<V>
                 && std::floating_point<typename U::value_type>
    void inv(U A, V B)
    {
        using value_type = typename U::value_type;
        const std::size_t nrows = A.extent(0);
        switch (nrows) {
        case 1:
            B(0, 0) = value_type{1} / A(0, 0);
            break;
        case 2: {
            value_type idet = 1. / det(A);
            B(0, 0) = idet * A(1, 1);
            B(0, 1) = -idet * A(0, 1);
            B(1, 0) = -idet * A(1, 0);
            B(1, 1) = idet * A(0, 0);
            break;
        }
        case 3: {
            value_type w0
                = difference_of_products(A(1, 1), A(1, 2), A(2, 1), A(2, 2));
            value_type w1
                = difference_of_products(A(1, 0), A(1, 2), A(2, 0), A(2, 2));
            value_type w2
                = difference_of_products(A(1, 0), A(1, 1), A(2, 0), A(2, 1));
            value_type w3 = difference_of_products(A(0, 0), A(0, 1), w1, w0);
            value_type detv = std::fma(A(0, 2), w2, w3);
            assert(detv != 0.);
            value_type idet = 1 / detv;

            B(0, 0) = w0 * idet;
            B(1, 0) = -w1 * idet;
            B(2, 0) = w2 * idet;
            B(0, 1) = difference_of_products(A(0, 2), A(0, 1), A(2, 2), A(2, 1))
                      * idet;
            B(0, 2) = difference_of_products(A(0, 1), A(0, 2), A(1, 1), A(1, 2))
                      * idet;
            B(1, 1) = difference_of_products(A(0, 0), A(0, 2), A(2, 0), A(2, 2))
                      * idet;
            B(1, 2) = difference_of_products(A(1, 0), A(0, 0), A(1, 2), A(0, 2))
                      * idet;
            B(2, 1) = difference_of_products(A(2, 0), A(0, 0), A(2, 1), A(0, 1))
                      * idet;
            B(2, 2) = difference_of_products(A(0, 0), A(1, 0), A(0, 1), A(1, 1))
                      * idet;
            break;
        }
        default:
            throw std::runtime_error(
                std::format("math::inv is not implemented for {}x{} matrices.",
                            A.extent(0), A.extent(1)));
        }
    }

    /// Accumulate C += A * B (or, when `transpose` is set, C += A^T * B^T).
    template <typename U, typename V, typename P>
        requires MDSpanRank2<U> && MDSpanRank2<V> && MDSpanRank2<P>
                 && scalar<typename U::value_type> && scalar<typename V::value_type>
                 && scalar<typename P::value_type>
    constexpr void dot(U A, V B, P C, bool transpose = false)
    {
        if (transpose) {
            assert(A.extent(0) == B.extent(1));
            for (std::size_t i = 0; i < A.extent(1); i++)
                for (std::size_t j = 0; j < B.extent(0); j++)
                    for (std::size_t k = 0; k < A.extent(0); k++)
                        C(i, j) += A(k, i) * B(j, k);
        }
        else {
            assert(A.extent(1) == B.extent(0));
            for (std::size_t i = 0; i < A.extent(0); i++)
                for (std::size_t j = 0; j < B.extent(1); j++)
                    for (std::size_t k = 0; k < A.extent(1); k++)
                        C(i, j) += A(i, k) * B(k, j);
        }
    }

    /// Left pseudo-inverse P of a full-rank rectangular matrix A such that
    /// P * A = I. A is 3x2, 3x1 or 2x1.
    template <typename U, typename V>
        requires MDSpanRank2<U> && MDSpanRank2<V>
                 && std::floating_point<typename U::value_type>
    void pinv(U A, V P)
    {
        assert(A.extent(0) > A.extent(1));
        assert(P.extent(1) == A.extent(0));
        assert(P.extent(0) == A.extent(1));
        using T = typename U::value_type;
        if (A.extent(1) == 2) {
            assert(A.extent(0) == 3);
            std::array<T, 6> ATb;
            std::array<T, 4> ATAb, Invb;
            md::mdspan<T, md::extents<std::size_t, 2, 3>> AT(ATb.data(), 2, 3);
            md::mdspan<T, md::extents<std::size_t, 2, 2>> ATA(ATAb.data(), 2, 2);
            md::mdspan<T, md::extents<std::size_t, 2, 2>> Inv(Invb.data(), 2, 2);

            for (std::size_t i = 0; i < AT.extent(0); ++i)
                for (std::size_t j = 0; j < AT.extent(1); ++j)
                    AT(i, j) = A(j, i);

            std::ranges::fill(ATAb, 0.0);
            for (std::size_t i = 0; i < P.extent(0); ++i)
                for (std::size_t j = 0; j < P.extent(1); ++j)
                    P(i, j) = 0;

            // pinv(A) = (A^T * A)^-1 * A^T
            dot(AT, A, ATA);
            inv(ATA, Inv);
            dot(Inv, AT, P);
        }
        else if (A.extent(1) == 1) {
            T res = 0;
            for (std::size_t i = 0; i < A.extent(0); ++i)
                for (std::size_t j = 0; j < A.extent(1); ++j)
                    res += A(i, j) * A(i, j);

            for (std::size_t i = 0; i < A.extent(0); ++i)
                for (std::size_t j = 0; j < A.extent(1); ++j)
                    P(j, i) = (1 / res) * A(i, j);
        }
        else {
            throw std::runtime_error(
                std::format("math::pinv is not implemented for {}x{} matrices.",
                            A.extent(0), A.extent(1)));
        }
    }

} // namespace hellofem::math
