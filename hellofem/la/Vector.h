// hellofem::la — distributed vector, single-process
// SPDX-License-Identifier: MIT

#pragma once

#include "common/IndexMap.h"
#include "common/types.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace hellofem::la {

    /// Process-local vector with an index map layout (single-process:
    /// no ghosts, so the owned range is the whole array).
    ///
    /// @tparam T Value type (real or complex).
    /// @tparam Container Backing storage, default `std::vector<T>`.
    template <typename T, typename Container = std::vector<T>>
    class Vector {
    public:
        /// Value type
        using value_type = T;

        /// Backing container type
        using container_type = Container;

        /// Create a zero vector of length `bs * map->size_local()`.
        /// @param[in] map Index map describing the layout.
        /// @param[in] bs Block size (physical entries per block index).
        Vector(std::shared_ptr<const common::IndexMap> map, int bs)
            : _map(std::move(map)), _bs(bs), _x(static_cast<std::size_t>(_bs) * _map->size_local())
        {
        }

        /// Copy constructor
        Vector(const Vector& x) = default;

        /// Move constructor
        Vector(Vector&& x) = default;

        /// Copy assignment
        Vector& operator=(const Vector& x) = default;

        /// Move assignment
        Vector& operator=(Vector&& x) = default;

        /// Destructor
        ~Vector() = default;

        /// No-op ghost scatter (single-process identity; retained for
        /// parallel parity).
        void scatter_fwd_begin() { }

        /// No-op
        void scatter_fwd_end() { }

        /// No-op
        void scatter_fwd() { }

        /// No-op
        void scatter_rev_begin() { }

        /// No-op
        void scatter_rev_end() { }

        /// No-op
        template <class BinaryOperation>
        void scatter_rev(BinaryOperation) { }

        /// Index map describing the layout.
        std::shared_ptr<const common::IndexMap> index_map() const
        {
            return _map;
        }

        /// Block size.
        int bs() const noexcept { return _bs; }

        /// Mutable access to the data (owned entries).
        container_type& array() { return _x; }

        /// Read-only access to the data.
        const container_type& array() const { return _x; }

        /// Fill all entries with `v`.
        void set(value_type v) { std::ranges::fill(_x, v); }

    private:
        // Index map describing the layout
        std::shared_ptr<const common::IndexMap> _map;

        // Block size
        int _bs;

        // Vector data
        container_type _x;
    };

    /// Compute the inner product `a^{H} b` of two vectors with the same
    /// layout.
    template <class V>
    auto inner_product(const V& a, const V& b)
    {
        using T = typename V::value_type;
        const std::int32_t local_size = a.bs() * a.index_map()->size_local();
        if (local_size != b.bs() * b.index_map()->size_local())
            throw std::runtime_error("Incompatible vector sizes");

        const T local = std::transform_reduce(a.array().begin(),
            std::next(a.array().begin(), local_size), b.array().begin(),
            static_cast<T>(0), std::plus {}, [](T a, T b) -> T {
                if constexpr (std::is_same<T, std::complex<double>>::value
                    or std::is_same<T, std::complex<float>>::value) {
                    return std::conj(a) * b;
                }
                else
                    return a * b;
            });

        // Single-process: no reduction needed.
        return local;
    }

    /// Compute the squared Euclidean norm of a vector.
    template <class V>
    auto squared_norm(const V& a)
    {
        return std::real(inner_product(a, a));
    }

    /// Compute the norm of a vector.
    /// @param[in] x Vector to compute the norm of.
    /// @param[in] type Norm type (l1, l2, linf).
    template <class V>
    auto norm(const V& x, Norm type = Norm::l2)
    {
        using T = typename V::value_type;
        switch (type) {
        case Norm::l1: {
            const std::int32_t local_size = x.bs() * x.index_map()->size_local();
            using U = hellofem::scalar_value_t<T>;
            return std::accumulate(x.array().begin(),
                std::next(x.array().begin(), local_size), U(0),
                [](auto n, auto v) { return n + std::abs(v); });
        }
        case Norm::l2:
            return std::sqrt(squared_norm(x));
        case Norm::linf: {
            const std::int32_t local_size = x.bs() * x.index_map()->size_local();
            auto max_pos = std::max_element(x.array().begin(),
                std::next(x.array().begin(), local_size),
                [](T a, T b) { return std::norm(a) < std::norm(b); });
            return std::abs(*max_pos);
        }
        default:
            throw std::runtime_error("Norm type not supported");
        }
    }

} // namespace hellofem::la
