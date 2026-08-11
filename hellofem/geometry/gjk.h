// hellofem::geometry — Gilbert-Johnson-Keerthi (GJK) distance
// SPDX-License-Identifier: MIT
//
// Distance between two convex point sets (the hulls of their vertices),
// via the Minkowski difference and the closest-point-on-simplex method
// (Ericson, Real-Time Collision Detection, ch. 9). The internal
// computation runs in an extended-precision type (double-double, ~106
// mantissa bits) so that the simplex-case decisions (which vertex/edge/
// facet the origin projects onto) stay correct under the cancellation
// inherent to difference-of-dot-products; the support-point search runs
// in the cheap original precision.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace hellofem::geometry {

    namespace impl {

        /// Double-double: sum of two doubles, hi + lo, giving ~106
        /// mantissa bits (Dekker/Knuth error-free transforms). Operations
        /// needed by the GJK loop only; see the comment at the top of
        /// this file.
        struct double_double {
            double hi = 0.0;
            double lo = 0.0;

            double_double() = default;
            constexpr double_double(double h, double l = 0.0)
                : hi(h), lo(l)
            {
            }

            // Comparisons
            friend bool operator<(double_double a, double_double b)
            {
                return a.hi < b.hi or (a.hi == b.hi and a.lo < b.lo);
            }
            friend bool operator>(double_double a, double_double b)
            {
                return b < a;
            }
            friend bool operator<=(double_double a, double_double b)
            {
                return not(b < a);
            }
            friend bool operator>=(double_double a, double_double b)
            {
                return not(a < b);
            }
            friend bool operator==(double_double a, double_double b)
            {
                return a.hi == b.hi and a.lo == b.lo;
            }
            friend bool operator!=(double_double a, double_double b)
            {
                return not(a == b);
            }

            // a + b, exact split (Knuth 4.2.2 Algorithm A).
            friend double_double operator+(double_double a, double_double b)
            {
                double s = a.hi + b.hi;
                double t = s - a.hi;
                double e = (a.hi - (s - t)) + (b.hi - t);
                double f = a.lo + b.lo;
                double s2 = s + f;
                double e2 = s2 - s;
                return {s2, e + (f - e2) + (a.lo + b.lo - f)};
            }

            // Unary minus
            friend double_double operator-(double_double a)
            {
                return {-a.hi, -a.lo};
            }

            // a - b
            friend double_double operator-(double_double a, double_double b)
            {
                return a + double_double(-b.hi, -b.lo);
            }

            // a + scalar
            friend double_double operator+(double_double a, double b)
            {
                return a + double_double(b);
            }

            // a - scalar
            friend double_double operator-(double_double a, double b)
            {
                return a - double_double(b);
            }

            // a * b via FMA (Dekker split; ~106-bit product).
            friend double_double operator*(double_double a, double_double b)
            {
                double p = a.hi * b.hi;
                double e = std::fma(a.hi, b.hi, -p)
                    + a.hi * b.lo + a.lo * b.hi;
                return {p, e};
            }

            // a / b, one Newton refinement of the quotient.
            friend double_double operator/(double_double a, double_double b)
            {
                double q = a.hi / b.hi;
                // r = a - b*q, then q += r / b.hi.
                double r_hi = a.hi - b.hi * q;
                double r = r_hi - b.lo * q + a.lo;
                return double_double(q) + double_double(r / b.hi);
            }

            double_double& operator+=(double_double b)
            {
                *this = *this + b;
                return *this;
            }
            double_double& operator-=(double_double b)
            {
                *this = *this - b;
                return *this;
            }
            double_double& operator*=(double_double b)
            {
                *this = *this * b;
                return *this;
            }
            double_double& operator/=(double_double b)
            {
                *this = *this / b;
                return *this;
            }

            /// Explicit conversion to double (the low word is dropped).
            double to_double() const { return hi; }
        };

        /// Machine epsilon of the extended type (~2^-105).
        constexpr double_double epsilon_dd()
        {
            constexpr double eps = std::numeric_limits<double>::epsilon()
                * std::numeric_limits<double>::epsilon();
            return double_double(eps);
        }

        /// sqrt(a), one Newton refinement.
        inline double_double sqrt(double_double a)
        {
            if (a.hi <= 0.0)
                return double_double(0.0);
            double x = std::sqrt(a.hi);
            double_double r = a - double_double(x) * double_double(x);
            return double_double(x) + r / double_double(2.0 * x);
        }

        // Scalar conversion helpers for generic code.
        template <typename T>
        struct scalar_traits;

        template <>
        struct scalar_traits<double> {
            static double epsilon() { return std::numeric_limits<double>::epsilon(); }
            static double sqrt(double x) { return std::sqrt(x); }
        };
        template <>
        struct scalar_traits<double_double> {
            static double_double epsilon() { return epsilon_dd(); }
            static double_double sqrt(double_double x) { return impl::sqrt(x); }
        };

        /// Convert an extended-precision value back to a base scalar.
        template <typename U>
        double to_base(const U& x)
        {
            return x.to_double();
        }
        template <>
        inline double to_base<double>(const double& x)
        {
            return x;
        }

        /// Dot product of two 3-vectors.
        template <typename V>
        typename V::value_type dot3(const V& a, const V& b)
        {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        /// Signed cofactors of the four 3x3 matrices in a 4x3 vertex set;
        /// equivalently the orientation of each tetrahedron face.
        /// @param s Flattened 4x3 vertices (row-major).
        template <typename T>
        inline std::array<T, 4> det4(const std::array<T, 12>& s)
        {
            std::span<const T, 3> s0(s.begin(), 3);
            std::span<const T, 3> s1(s.begin() + 3, 3);
            std::span<const T, 3> s2(s.begin() + 6, 3);
            std::span<const T, 3> s3(s.begin() + 9, 3);

            std::array<T, 4> w;
            T c0 = s2[1] * s3[2] - s2[2] * s3[1];
            T c1 = s2[0] * s3[2] - s2[2] * s3[0];
            T c2 = s2[0] * s3[1] - s2[1] * s3[0];
            w[2] = -s0[0] * c0 + s0[1] * c1 - s0[2] * c2;
            w[3] = s1[0] * c0 - s1[1] * c1 + s1[2] * c2;

            c0 = s0[1] * s1[2] - s0[2] * s1[1];
            c1 = s0[0] * s1[2] - s0[2] * s1[0];
            c2 = s0[0] * s1[1] - s0[1] * s1[0];
            w[0] = -s2[0] * c0 + s2[1] * c1 - s2[2] * c2;
            w[1] = s3[0] * c0 - s3[1] * c1 + s3[2] * c2;

            return w;
        }

        /// Index of the point in `bd` (flattened (n,3)) maximising `p . v`.
        template <typename T>
        inline std::int32_t support(std::span<const T> bd, const std::array<T, 3>& v)
        {
            T best = bd[0] * v[0] + bd[1] * v[1] + bd[2] * v[2];
            std::int32_t best_i = 0;
            for (std::size_t i = 1; i < bd.size() / 3; ++i) {
                const T val = bd[3 * i] * v[0] + bd[3 * i + 1] * v[1]
                    + bd[3 * i + 2] * v[2];
                if (val > best) {
                    best = val;
                    best_i = static_cast<std::int32_t>(i);
                }
            }
            return best_i;
        }

        /// Barycentric coordinates (into `coordinates`) of the point in
        /// the simplex `s` closest to the origin, for an interval (2),
        /// triangle (3) or tetrahedron (4). `s` holds 3*simplex_size
        /// values; `coordinates` is sized for a tetrahedron.
        template <typename T, std::size_t simplex_size>
        void nearest_simplex(const std::array<T, 12>& s,
            std::array<T, 4>& coordinates)
        {
            if constexpr (simplex_size == 2) {
                // Interval: origin projects on the segment or on an end.
                std::span<const T, 3> s0(s.begin(), 3);
                std::span<const T, 3> s1(s.begin() + 3, 3);
                T lm = dot3(s0, s0) - dot3(s0, s1);
                if (lm < 0.0) {
                    coordinates = {1.0, 0.0, 0.0, 0.0};
                    return;
                }
                T mu = dot3(s1, s1) - dot3(s1, s0);
                if (mu < 0.0) {
                    coordinates = {0.0, 1.0, 0.0, 0.0};
                    return;
                }
                T f = 1.0 / (lm + mu);
                coordinates = {mu * f, lm * f, 0.0, 0.0};
            }
            else if constexpr (simplex_size == 3) {
                // Triangle: 7 regions (near a vertex, on an edge, interior).
                std::span<const T, 3> a(s.begin(), 3);
                std::span<const T, 3> b(s.begin() + 3, 3);
                std::span<const T, 3> c(s.begin() + 6, 3);

                const T aa = dot3(a, a), ab = dot3(a, b), ac = dot3(a, c);
                const T d1 = aa - ab, d2 = aa - ac;
                if (d1 < 0.0 and d2 < 0.0) {
                    coordinates = {1.0, 0.0, 0.0, 0.0};
                    return;
                }
                const T bb = dot3(b, b), bc = dot3(b, c);
                const T d3 = bb - ab, d4 = bb - bc;
                if (d3 < 0.0 and d4 < 0.0) {
                    coordinates = {0.0, 1.0, 0.0, 0.0};
                    return;
                }
                const T cc = dot3(c, c);
                const T d5 = cc - ac, d6 = cc - bc;
                if (d5 < 0.0 and d6 < 0.0) {
                    coordinates = {0.0, 0.0, 1.0, 0.0};
                    return;
                }

                const T vc = d4 * d1 - d1 * d3 + d3 * d2;
                if (vc < 0.0 and d1 > 0.0 and d3 > 0.0) {
                    const T f = 1.0 / (d1 + d3);
                    coordinates = {d3 * f, d1 * f, 0.0, 0.0};
                    return;
                }
                const T vb = d1 * d5 - d5 * d2 + d2 * d6;
                if (vb < 0.0 and d2 > 0.0 and d5 > 0.0) {
                    const T f = 1.0 / (d2 + d5);
                    coordinates = {d5 * f, 0.0, d2 * f, 0.0};
                    return;
                }
                const T va = d3 * d6 - d6 * d4 + d4 * d5;
                if (va < 0.0 and d4 > 0.0 and d6 > 0.0) {
                    const T f = 1.0 / (d4 + d6);
                    coordinates = {0.0, d6 * f, d4 * f, 0.0};
                    return;
                }

                const T f = 1.0 / (va + vb + vc);
                coordinates = {va * f, vb * f, vc * f, 0.0};
            }
            else if constexpr (simplex_size == 4) {
                // Tetrahedron: 4 vertices, 6 edges, 4 facets or interior.
                std::ranges::fill(coordinates, 0.0);

                // Vertex cases via the d[i][j] = |s_i|^2 - s_i.s_j matrix.
                T d[4][4] = {};
                for (int i = 0; i < 4; ++i) {
                    std::span<const T, 3> si(s.begin() + i * 3, 3);
                    const T sii = dot3(si, si);
                    bool out = true;
                    for (int j = 0; j < 4; ++j) {
                        if (i != j) {
                            std::span<const T, 3> sj(s.begin() + j * 3, 3);
                            d[i][j] = sii - dot3(si, sj);
                            if (d[i][j] > 0.0)
                                out = false;
                        }
                    }
                    if (out) {
                        coordinates[i] = 1.0;
                        return;
                    }
                }

                // Edge cases: 6 edges, j0-j1 with opposing edge j2-j3.
                constexpr int edges[6][2]
                    = {{2, 3}, {1, 3}, {1, 2}, {0, 3}, {0, 2}, {0, 1}};
                T v[6][2] = {};
                for (int i = 0; i < 6; ++i) {
                    const int j0 = edges[i][0], j1 = edges[i][1];
                    const int j2 = edges[5 - i][0], j3 = edges[5 - i][1];
                    v[i][0] = d[j1][j2] * d[j0][j1] - d[j0][j1] * d[j1][j0]
                        + d[j1][j0] * d[j0][j2];
                    v[i][1] = d[j1][j3] * d[j0][j1] - d[j0][j1] * d[j1][j0]
                        + d[j1][j0] * d[j0][j3];
                    if (v[i][0] <= 0.0 and v[i][1] <= 0.0
                        and d[j0][j1] >= 0.0 and d[j1][j0] >= 0.0) {
                        const T f = 1.0 / (d[j0][j1] + d[j1][j0]);
                        coordinates[j0] = f * d[j1][j0];
                        coordinates[j1] = f * d[j0][j1];
                        return;
                    }
                }

                // Facet cases via det4 cofactors (signed face orientations).
                std::array<T, 4> w = det4(s);
                T wsum = w[0] + w[1] + w[2] + w[3];
                if (wsum < 0.0) {
                    w[0] = -w[0];
                    w[1] = -w[1];
                    w[2] = -w[2];
                    w[3] = -w[3];
                    wsum = -wsum;
                }
                if (w[0] < 0.0 and v[2][0] > 0.0 and v[4][0] > 0.0
                    and v[5][0] > 0.0) {
                    const T f = 1.0 / (v[2][0] + v[4][0] + v[5][0]);
                    coordinates = {v[2][0] * f, v[4][0] * f, v[5][0] * f, 0.0};
                    return;
                }
                if (w[1] < 0.0 and v[1][0] > 0.0 and v[3][0] > 0.0
                    and v[5][1] > 0.0) {
                    const T f = 1.0 / (v[1][0] + v[3][0] + v[5][1]);
                    coordinates = {v[1][0] * f, v[3][0] * f, 0.0, v[5][1] * f};
                    return;
                }
                if (w[2] < 0.0 and v[0][0] > 0.0 and v[3][1] > 0.0
                    and v[4][1] > 0.0) {
                    const T f = 1.0 / (v[0][0] + v[3][1] + v[4][1]);
                    coordinates = {v[0][0] * f, 0.0, v[3][1] * f, v[4][1] * f};
                    return;
                }
                if (w[3] < 0.0 and v[0][1] > 0.0 and v[1][1] > 0.0
                    and v[2][1] > 0.0) {
                    const T f = 1.0 / (v[0][1] + v[1][1] + v[2][1]);
                    coordinates = {0.0, v[0][1] * f, v[1][1] * f, v[2][1] * f};
                    return;
                }

                // Interior.
                coordinates = {w[3] / wsum, w[2] / wsum, w[1] / wsum,
                    w[0] / wsum};
            }
            else {
                static_assert(simplex_size >= 2 and simplex_size <= 4,
                    "GJK simplex must have 2, 3 or 4 vertices.");
            }
        }

    } // namespace impl

    /// Shortest vector between the convex hulls of two point sets.
    ///
    /// @param[in] p0 Body 1 points, flat `(num_points, 3)` row-major.
    /// @param[in] q0 Body 2 points, flat `(num_points, 3)` row-major.
    /// @tparam T Input scalar type.
    /// @tparam U Internal precision (must be higher than T for accuracy).
    /// @return Shortest vector from body 2 to body 1 (near-zero if they
    /// touch or intersect).
    template <std::floating_point T,
        typename U = impl::double_double>
    std::array<T, 3> compute_distance_gjk(std::span<const T> p0,
        std::span<const T> q0)
    {
        assert(p0.size() % 3 == 0);
        assert(q0.size() % 3 == 0);

        constexpr int maxk = 15;
        const U eps = U(1000) * impl::scalar_traits<U>::epsilon();

        // Initialise the distance vector and the simplex with the first
        // Minkowski-difference point.
        std::array<U, 3> x_k
            = {U(p0[0]) - U(q0[0]), U(p0[1]) - U(q0[1]), U(p0[2]) - U(q0[2])};
        std::array<U, 12> s = {U(0)};
        s[0] = x_k[0];
        s[1] = x_k[1];
        s[2] = x_k[2];
        std::array<U, 4> lmn = {U(0)};
        std::size_t simplex_size = 1;

        int k;
        for (k = 0; k < maxk; ++k) {
            const U x_norm2 = impl::dot3(x_k, x_k);
            std::array<U, 3> x_k_normalized = x_k;
            if (x_norm2 > eps * eps) {
                const U inv_norm
                    = U(1.0) / impl::scalar_traits<U>::sqrt(x_norm2);
                for (std::size_t i = 0; i < 3; ++i)
                    x_k_normalized[i] *= inv_norm;
            }

            // Support points in the cheap original precision.
            const std::array<T, 3> dir_p {T(-to_base(x_k_normalized[0])),
                T(-to_base(x_k_normalized[1])), T(-to_base(x_k_normalized[2]))};
            const std::array<T, 3> dir_q {T(to_base(x_k_normalized[0])),
                T(to_base(x_k_normalized[1])), T(to_base(x_k_normalized[2]))};
            const std::int32_t ip = impl::support(p0, dir_p);
            const std::int32_t iq = impl::support(q0, dir_q);

            const std::array<U, 3> s_k {U(p0[3 * ip]) - U(q0[3 * iq]),
                U(p0[3 * ip + 1]) - U(q0[3 * iq + 1]),
                U(p0[3 * ip + 2]) - U(q0[3 * iq + 2])};

            // Exit if the support point is already a simplex vertex.
            std::size_t m;
            for (m = 0; m < simplex_size; ++m) {
                const U* it = s.data() + 3 * m;
                if (std::equal(it, it + 3, s_k.begin()))
                    break;
            }
            if (m != simplex_size)
                break;

            // Exit if no progress: (x_k - s_k) . x_k ~ 0.
            const U xs_diff = x_norm2 - impl::dot3(x_k, s_k);
            if (xs_diff < (eps * x_norm2) or xs_diff < eps)
                break;

            std::ranges::copy(s_k, s.begin() + 3 * simplex_size);
            ++simplex_size;

            switch (simplex_size) {
            case 2:
                impl::nearest_simplex<U, 2>(s, lmn);
                break;
            case 3:
                impl::nearest_simplex<U, 3>(s, lmn);
                break;
            case 4:
                impl::nearest_simplex<U, 4>(s, lmn);
                break;
            default:
                throw std::runtime_error("GJK: invalid simplex size");
            }

            // x_k = weighted sum of the kept vertices; drop zero-weight
            // vertices and compact the simplex.
            std::size_t j = 0;
            x_k = {U(0), U(0), U(0)};
            for (std::size_t i = 0; i < simplex_size; ++i) {
                std::span<const U> sc(s.data() + 3 * i, 3);
                if (lmn[i] > U(0)) {
                    for (std::size_t c = 0; c < 3; ++c)
                        x_k[c] += lmn[i] * sc[c];
                    if (i > j)
                        std::ranges::copy(sc, s.begin() + 3 * j);
                    ++j;
                }
            }
            simplex_size = j;

            // Strict monotonicity; touching/intersecting.
            const U x_next_norm2 = impl::dot3(x_k, x_k);
            if (x_norm2 <= x_next_norm2)
                break;
            if (x_next_norm2 < eps * eps)
                break;
        }

        if (k == maxk)
            throw std::runtime_error("GJK: max iteration limit reached");
        return {static_cast<T>(to_base(x_k[0])), static_cast<T>(to_base(x_k[1])),
            static_cast<T>(to_base(x_k[2]))};
    }

} // namespace hellofem::geometry
