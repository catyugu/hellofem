// hellofem::la — sparse matrix in compressed sparse row (CSR) format
// SPDX-License-Identifier: MIT

#pragma once

#include "LinearOperator.h"
#include "SparsityPattern.h"
#include "common/IndexMap.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hellofem::la {

    /// Block layout mode of a MatrixCSR.
    enum class BlockMode : int {
        compact = 0, ///< Values stored as nnz blocks of size bs0*bs1
        expanded = 1, ///< Values expanded to scalar entries (deferred)
    };

    namespace impl {

        /// Insert a block of data with the same block size as the matrix.
        /// Data layout is row-major over (rows, cols, bs0, bs1).
        template <typename T, int BS0, int BS1, typename OP, typename X,
            typename Y>
        void insert_csr(std::vector<T>& data,
            std::span<const std::int32_t> cols,
            std::span<const std::int64_t> row_ptr, const X& x, const Y& xrows,
            const Y& xcols, OP op)
        {
            const std::size_t nc = xcols.size();
            assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
            for (std::size_t r = 0; r < xrows.size(); ++r) {
                const std::int32_t row = xrows[r];
                const T* xr = x.data() + r * nc * BS0 * BS1;

                auto cit0 = cols.begin() + row_ptr[row];
                auto cit1 = cols.begin() + row_ptr[row + 1];
                for (std::size_t c = 0; c < nc; ++c) {
                    // Find the column slot in the (sorted) row
                    auto it = std::lower_bound(cit0, cit1, xcols[c]);
                    if (it == cit1 or *it != xcols[c])
                        throw std::runtime_error("Entry not in sparsity");

                    std::size_t d
                        = std::ranges::distance(cols.begin(), it) * BS0 * BS1;
                    std::size_t xi = c * BS1;
                    assert(d < data.size());
                    for (int i = 0; i < BS0; ++i) {
                        for (int j = 0; j < BS1; ++j)
                            op(data[d + j], xr[xi + j]);
                        d += BS1;
                        xi += nc * BS1;
                    }
                }
            }
        }

        /// Insert blocked data into a non-blocked matrix (bs=={1,1}).
        template <typename T, int BS0, int BS1, typename OP, typename X,
            typename Y>
        void insert_blocked_csr(std::vector<T>& data,
            std::span<const std::int32_t> cols,
            std::span<const std::int64_t> row_ptr, const X& x, const Y& xrows,
            const Y& xcols, OP op)
        {
            const std::size_t nc = xcols.size();
            assert(x.size() == xrows.size() * xcols.size() * BS0 * BS1);
            for (std::size_t r = 0; r < xrows.size(); ++r) {
                const std::int32_t row = xrows[r] * BS0;

                for (int i = 0; i < BS0; ++i) {
                    const T* xr = x.data() + (r * BS0 + i) * nc * BS1;

                    auto cit0 = cols.begin() + row_ptr[row + i];
                    auto cit1 = cols.begin() + row_ptr[row + i + 1];
                    for (std::size_t c = 0; c < nc; ++c) {
                        auto it = std::lower_bound(cit0, cit1, xcols[c] * BS1);
                        if (it == cit1 or *it != xcols[c] * BS1)
                            throw std::runtime_error("Entry not in sparsity");

                        std::size_t d = std::ranges::distance(cols.begin(), it);
                        assert(d < data.size());
                        std::size_t xi = c * BS1;
                        for (int j = 0; j < BS1; ++j)
                            op(data[d + j], xr[xi + j]);
                    }
                }
            }
        }

        /// Insert non-blocked data into a blocked matrix.
        template <typename T, typename OP, typename X, typename Y>
        void insert_nonblocked_csr(std::vector<T>& data,
            std::span<const std::int32_t> cols,
            std::span<const std::int64_t> row_ptr, const X& x, const Y& xrows,
            const Y& xcols, OP op, int bs0, int bs1)
        {
            const std::size_t nc = xcols.size();
            const int nbs = bs0 * bs1;

            assert(x.size() == xrows.size() * xcols.size());
            for (std::size_t r = 0; r < xrows.size(); ++r) {
                auto rdiv = std::div(xrows[r], bs0);
                const T* xr = x.data() + r * nc;

                auto cit0 = cols.begin() + row_ptr[rdiv.quot];
                auto cit1 = cols.begin() + row_ptr[rdiv.quot + 1];
                for (std::size_t c = 0; c < nc; ++c) {
                    auto cdiv = std::div(xcols[c], bs1);
                    auto it = std::lower_bound(cit0, cit1, cdiv.quot);
                    if (it == cit1 or *it != cdiv.quot)
                        throw std::runtime_error("Entry not in sparsity");

                    std::size_t d = std::ranges::distance(cols.begin(), it);
                    std::size_t di = d * nbs + rdiv.rem * bs1 + cdiv.rem;
                    assert(di < data.size());
                    op(data[di], xr[c]);
                }
            }
        }

        /// Sparse matrix-vector product `y += A x` for a blocked CSR
        /// matrix. Block entries are stored row-major: element at
        /// row-offset `k0`, column-offset `k1` of block `j` is at
        /// `values[j*bs0*bs1 + k0*bs1 + k1]`.
        template <typename T>
        void spmv(std::span<const T> values,
            std::span<const std::int64_t> row_ptr,
            std::span<const std::int32_t> indices, std::span<const T> x,
            std::span<T> y, int bs0, int bs1)
        {
            for (int k0 = 0; k0 < bs0; ++k0) {
                for (std::size_t i = 0; i < row_ptr.size() - 1; ++i) {
                    T vi {0};
                    for (std::int64_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                        for (int k1 = 0; k1 < bs1; ++k1)
                            vi += values[j * bs0 * bs1 + k0 * bs1 + k1]
                                * x[indices[j] * bs1 + k1];
                    }
                    y[i * bs0 + k0] += vi;
                }
            }
        }

        /// Sparse matrix-vector transpose product `y += A^T x`.
        template <typename T>
        void spmvT(std::span<const T> values,
            std::span<const std::int64_t> row_ptr,
            std::span<const std::int32_t> indices, std::span<const T> x,
            std::span<T> y, int bs0, int bs1)
        {
            for (int k0 = 0; k0 < bs0; ++k0) {
                for (std::size_t i = 0; i < row_ptr.size() - 1; ++i) {
                    const T xval = x[i * bs0 + k0];
                    for (std::int64_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                        for (int k1 = 0; k1 < bs1; ++k1)
                            y[indices[j] * bs1 + k1]
                                += values[j * bs0 * bs1 + k0 * bs1 + k1] * xval;
                    }
                }
            }
        }

    } // namespace impl

    /// Sparse matrix in CSR format, built from a finalized sparsity
    /// pattern. Values are stored block-row-major: `nnz * bs0 * bs1`
    /// entries, block `j` occupying `[j*bs0*bs1, (j+1)*bs0*bs1)`.
    ///
    /// This is the matrix that FEM assembly writes into via `set`/`add`,
    /// and the assembled matrix the Krylov solvers act on.
    template <typename T, typename Container = std::vector<T>,
        typename ColContainer = std::vector<std::int32_t>,
        typename RowPtrContainer = std::vector<std::int64_t>>
    class MatrixCSR {
    public:
        /// Value type
        using value_type = T;

        /// Value container type
        using container_type = Container;

        /// Column index container type
        using col_container_type = ColContainer;

        /// Row pointer container type
        using rowptr_container_type = RowPtrContainer;

        /// Build a matrix from a finalized sparsity pattern, with all
        /// values initialized to zero.
        /// @param[in] pattern The finalized pattern.
        /// @param[in] mode Block layout mode (compact; expanded deferred).
        explicit MatrixCSR(const SparsityPattern& pattern,
            BlockMode mode = BlockMode::compact)
            : _index_maps({pattern.index_map(0), pattern.index_map(1)}), _block_mode(mode), _bs({pattern.block_size(0), pattern.block_size(1)})
        {
            if (mode == BlockMode::expanded)
                throw std::runtime_error(
                    "MatrixCSR expanded block mode is not yet implemented.");

            auto [edges, offsets] = pattern.graph();
            _cols.assign(edges.begin(), edges.end());
            _row_ptr.assign(offsets.begin(), offsets.end());
            _data.assign(
                static_cast<std::size_t>(_row_ptr.back()) * _bs[0] * _bs[1], 0);
        }

        /// Copy constructor
        MatrixCSR(const MatrixCSR& A) = default;

        /// Move constructor
        MatrixCSR(MatrixCSR&& A) = default;

        /// Copy assignment
        MatrixCSR& operator=(const MatrixCSR& A) = default;

        /// Move assignment
        MatrixCSR& operator=(MatrixCSR&& A) = default;

        /// Destructor
        ~MatrixCSR() = default;

        /// Accumulate `y += A x`.
        /// @param[in] x Input vector.
        /// @param[in,out] y Accumulator vector.
        void mult(const Vector<T>& x, Vector<T>& y) const
        {
            std::span<const T> values(_data);
            std::span<const std::int64_t> row_ptr(_row_ptr);
            std::span<const std::int32_t> cols(_cols);
            std::span<const T> xa(x.array());
            std::span<T> ya(y.array());
            impl::spmv(values, row_ptr, cols, xa, ya, _bs[0], _bs[1]);
        }

        /// Accumulate `y += A^T x`.
        void multT(const Vector<T>& x, Vector<T>& y) const
        {
            std::span<const T> values(_data);
            std::span<const std::int64_t> row_ptr(_row_ptr);
            std::span<const std::int32_t> cols(_cols);
            std::span<const T> xa(x.array());
            std::span<T> ya(y.array());
            impl::spmvT(values, row_ptr, cols, xa, ya, _bs[0], _bs[1]);
        }

        /// Set the block at (rows, cols) to the given values (overwrite).
        /// @tparam BS0 Data row block size.
        /// @tparam BS1 Data column block size.
        template <int BS0 = 1, int BS1 = 1>
        void set(std::span<const value_type> x,
            std::span<const std::int32_t> rows,
            std::span<const std::int32_t> cols)
        {
            auto op = [](value_type& a, value_type b) { a = b; };
            std::span<const std::int64_t> row_ptr(_row_ptr);
            if (_bs[0] == BS0 and _bs[1] == BS1)
                impl::insert_csr<value_type, BS0, BS1>(
                    _data, _cols, row_ptr, x, rows, cols, op);
            else if (_bs[0] == 1 and _bs[1] == 1)
                impl::insert_blocked_csr<value_type, BS0, BS1>(
                    _data, _cols, row_ptr, x, rows, cols, op);
            else {
                assert(BS0 == 1 and BS1 == 1);
                impl::insert_nonblocked_csr<value_type>(
                    _data, _cols, row_ptr, x, rows, cols, op, _bs[0], _bs[1]);
            }
        }

        /// Add the block at (rows, cols) to the existing values.
        template <int BS0 = 1, int BS1 = 1>
        void add(std::span<const value_type> x,
            std::span<const std::int32_t> rows,
            std::span<const std::int32_t> cols)
        {
            auto op = [](value_type& a, value_type b) { a += b; };
            std::span<const std::int64_t> row_ptr(_row_ptr);
            if (_bs[0] == BS0 and _bs[1] == BS1)
                impl::insert_csr<value_type, BS0, BS1>(
                    _data, _cols, row_ptr, x, rows, cols, op);
            else if (_bs[0] == 1 and _bs[1] == 1)
                impl::insert_blocked_csr<value_type, BS0, BS1>(
                    _data, _cols, row_ptr, x, rows, cols, op);
            else {
                assert(BS0 == 1 and BS1 == 1);
                impl::insert_nonblocked_csr<value_type>(
                    _data, _cols, row_ptr, x, rows, cols, op, _bs[0], _bs[1]);
            }
        }

        /// Return a functor matching the `MatSet` concept that calls
        /// `set<BS0,BS1>`: `f(rows, cols, values) -> int`.
        template <int BS0 = 1, int BS1 = 1>
        auto mat_set_values()
        {
            return [this](std::span<const std::int32_t> rows,
                       std::span<const std::int32_t> cols,
                       std::span<const value_type> values) -> int {
                this->set<BS0, BS1>(values, rows, cols);
                return 0;
            };
        }

        /// Return a functor matching the `MatSet` concept that calls
        /// `add<BS0,BS1>`.
        template <int BS0 = 1, int BS1 = 1>
        auto mat_add_values()
        {
            return [this](std::span<const std::int32_t> rows,
                       std::span<const std::int32_t> cols,
                       std::span<const value_type> values) -> int {
                this->add<BS0, BS1>(values, rows, cols);
                return 0;
            };
        }

        /// Number of owned rows.
        std::int32_t num_owned_rows() const
        {
            return _index_maps[0]->size_local();
        }

        /// Number of rows including any ghost rows (none in the
        /// single-process build, so this equals owned rows).
        std::int32_t num_all_rows() const
        {
            return static_cast<std::int32_t>(_row_ptr.size() - 1);
        }

        /// Matrix values as a row-major dense array, block-expanded.
        std::vector<value_type> to_dense() const
        {
            const std::int32_t nrows = num_owned_rows();
            const std::int32_t ncols = _index_maps[1]->size_local();
            const std::int32_t nrows_b = nrows * _bs[0];
            const std::int32_t ncols_b = ncols * _bs[1];
            std::vector<value_type> dense(
                static_cast<std::size_t>(nrows_b) * ncols_b, 0);

            for (std::int32_t r = 0; r < nrows; ++r) {
                for (std::int64_t j = _row_ptr[r]; j < _row_ptr[r + 1]; ++j) {
                    const std::int32_t c = _cols[j];
                    for (int i0 = 0; i0 < _bs[0]; ++i0)
                        for (int i1 = 0; i1 < _bs[1]; ++i1)
                            dense[static_cast<std::size_t>((r * _bs[0] + i0)
                                    * ncols_b
                                + c * _bs[1] + i1)]
                                = _data[j * _bs[0] * _bs[1] + i0 * _bs[1] + i1];
                }
            }

            return dense;
        }

        /// Index map for the given dimension (0 = rows, 1 = columns).
        std::shared_ptr<const common::IndexMap> index_map(int dim) const
        {
            return _index_maps.at(dim);
        }

        /// Block sizes for rows and columns.
        std::array<int, 2> block_size() const { return _bs; }

        /// Block layout mode.
        BlockMode block_mode() const { return _block_mode; }

        /// Mutable access to the values.
        Container& values() { return _data; }

        /// Read-only access to the values.
        const Container& values() const { return _data; }

        /// Read-only access to the row pointers.
        const RowPtrContainer& row_ptr() const { return _row_ptr; }

        /// Read-only access to the column indices.
        const ColContainer& cols() const { return _cols; }

        /// Wrap `*this` as an accumulating linear operator (copies the
        /// matrix into the operator).
        LinearOperator<T> as_operator() const
        {
            MatrixCSR A = *this;
            return LinearOperator<T>([A = std::move(A)](
                                         const Vector<T>& x, Vector<T>& y) {
                A.mult(x, y);
            });
        }

    private:
        // Index maps for rows and columns
        std::array<std::shared_ptr<const common::IndexMap>, 2> _index_maps;

        // Block layout mode
        BlockMode _block_mode;

        // Block sizes for rows and columns
        std::array<int, 2> _bs;

        // Nonzero values, block-row-major
        Container _data;

        // Column index of each nonzero block entry
        ColContainer _cols;

        // Row pointers into `_cols`/`_data`
        RowPtrContainer _row_ptr;
    };

} // namespace hellofem::la
