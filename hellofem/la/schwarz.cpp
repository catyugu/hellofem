// hellofem::la — additive Schwarz (overlapping domain decomposition)
// SPDX-License-Identifier: MIT

#include "schwarz.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace hellofem::la {

    template <typename T>
    struct SchwarzPreconditioner<T>::Impl {
        // Local factorization of each subdomain (SparseLU is not movable,
        // so held by pointer).
        std::vector<std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<T>>>>
            solvers;
        // Row index (in the scalar matrix) of each local row, per
        // subdomain: defines the restriction R_s and prolongation R_s^T.
        std::vector<std::vector<std::int32_t>> sub;
        std::int32_t nrows = 0;
    };

    namespace {

        /// Contiguous subdomain partition of `nrows` rows into `nparts`
        /// blocks, then grow each by `overlap` graph-adjacency layers.
        std::vector<std::vector<std::int32_t>> make_subdomains(
            const MatrixCSR<double>& S, std::int32_t nparts, int overlap)
        {
            const std::int32_t nrows = S.num_owned_rows();
            const auto& row_ptr = S.row_ptr();
            const auto& cols = S.cols();

            // Base contiguous partition.
            std::vector<std::vector<std::int32_t>> base(nparts);
            for (std::int32_t p = 0; p < nparts; ++p) {
                const std::int32_t lo = nrows * p / nparts;
                const std::int32_t hi = nrows * (p + 1) / nparts;
                base[p].reserve(static_cast<std::size_t>(hi - lo));
                for (std::int32_t r = lo; r < hi; ++r)
                    base[p].push_back(r);
            }

            // Grow each subdomain by `overlap` layers: a row joins a
            // subdomain if it is within `overlap` edges of a base row of
            // that subdomain (repeated one-step BFS growth).
            std::vector<std::vector<std::int32_t>> result(nparts);
            for (std::int32_t p = 0; p < nparts; ++p) {
                std::vector<std::int8_t> in(nrows, 0);
                for (std::int32_t r : base[p])
                    in[static_cast<std::size_t>(r)] = 1;
                std::vector<std::int32_t> front = base[p];
                for (int layer = 0; layer < overlap; ++layer) {
                    std::vector<std::int32_t> next;
                    for (std::int32_t r : front)
                        for (std::int64_t j = row_ptr[r]; j < row_ptr[r + 1]; ++j) {
                            const std::int32_t c = cols[static_cast<std::size_t>(j)];
                            if (!in[static_cast<std::size_t>(c)]) {
                                in[static_cast<std::size_t>(c)] = 1;
                                next.push_back(c);
                            }
                        }
                    front.swap(next);
                    if (front.empty())
                        break;
                }
                result[p].reserve(nrows);
                for (std::int32_t r = 0; r < nrows; ++r)
                    if (in[static_cast<std::size_t>(r)])
                        result[p].push_back(r);
            }
            return result;
        }

        /// Extract the principal submatrix of the scalar CSR `S` on the
        /// rows `sub` (both rows and columns restricted), as an Eigen
        /// sparse matrix.
        Eigen::SparseMatrix<double> extract_submatrix(const MatrixCSR<double>& S,
            const std::vector<std::int32_t>& sub)
        {
            const std::int32_t m = static_cast<std::int32_t>(sub.size());
            // Map global row -> local row within the subdomain.
            std::vector<std::int32_t> local_of(S.num_owned_rows(), -1);
            for (std::int32_t i = 0; i < m; ++i)
                local_of[static_cast<std::size_t>(sub[static_cast<std::size_t>(i)])] = i;

            Eigen::SparseMatrix<double> E(m, m);
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(static_cast<std::size_t>(S.cols().size()));
            const auto& row_ptr = S.row_ptr();
            const auto& cols = S.cols();
            const auto& values = S.values();
            for (std::int32_t i = 0; i < m; ++i) {
                const std::int32_t r = sub[static_cast<std::size_t>(i)];
                for (std::int64_t j = row_ptr[r]; j < row_ptr[r + 1]; ++j) {
                    const std::int32_t c = cols[static_cast<std::size_t>(j)];
                    const std::int32_t lc = local_of[static_cast<std::size_t>(c)];
                    if (lc >= 0) // column inside the subdomain
                        triplets.emplace_back(i, lc, values[static_cast<std::size_t>(j)]);
                }
            }
            E.setFromTriplets(triplets.begin(), triplets.end());
            return E;
        }

    } // namespace

    template <typename T>
    SchwarzPreconditioner<T>::SchwarzPreconditioner(
        const MatrixCSR<T>& A, int nparts, int overlap)
        : _S(A.to_scalar()), _nrows(_S.num_owned_rows())
    {
        if (nparts < 1)
            throw std::runtime_error("Schwarz: nparts must be >= 1.");
        if (overlap < 0)
            throw std::runtime_error("Schwarz: overlap must be >= 0.");

        _impl = std::make_shared<Impl>();
        _impl->nrows = _nrows;
        _impl->sub = make_subdomains(_S, nparts, overlap);
        _impl->solvers.resize(_impl->sub.size());

        for (std::size_t p = 0; p < _impl->sub.size(); ++p) {
            auto solver = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<T>>>();
            Eigen::SparseMatrix<T> E = extract_submatrix(_S, _impl->sub[p]);
            solver->compute(E);
            if (solver->info() != Eigen::Success)
                throw std::runtime_error(
                    "Schwarz: local factorization failed for a subdomain.");
            _impl->solvers[p] = std::move(solver);
        }
    }

    template <typename T>
    void SchwarzPreconditioner<T>::apply(const Vector<T>& x, Vector<T>& y) const
    {
        std::span<const T> xa = x.array();
        auto& ya = y.array();
        std::ranges::fill(ya, 0);

        // Parallel over subdomains; each accumulates into its own buffer
        // (overlapping rows are summed in the final reduction).
        const std::size_t nparts = _impl->sub.size();
        std::vector<std::vector<T>> contributions(nparts);
        for (std::size_t p = 0; p < nparts; ++p)
            contributions[p].assign(ya.size(), 0);

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, nparts),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t p = r.begin(); p < r.end(); ++p) {
                    const auto& sub = _impl->sub[p];
                    const std::int32_t m = static_cast<std::int32_t>(sub.size());
                    // Restrict: x_local = R_s x.
                    Eigen::Matrix<T, Eigen::Dynamic, 1> xl(m);
                    for (std::int32_t i = 0; i < m; ++i)
                        xl(i) = xa[static_cast<std::size_t>(sub[static_cast<std::size_t>(i)])];
                    // Solve A_ss yl = xl.
                    Eigen::Matrix<T, Eigen::Dynamic, 1> yl = _impl->solvers[p]->solve(xl);
                    // Prolongate and accumulate: y += R_s^T yl.
                    auto& contrib = contributions[p];
                    for (std::int32_t i = 0; i < m; ++i)
                        contrib[static_cast<std::size_t>(sub[static_cast<std::size_t>(i)])]
                            += yl(i);
                }
            });

        for (std::size_t p = 0; p < nparts; ++p)
            for (std::int32_t i = 0; i < _nrows; ++i)
                ya[static_cast<std::size_t>(i)] += contributions[p][static_cast<std::size_t>(i)];
    }

    template class SchwarzPreconditioner<double>;

} // namespace hellofem::la
