// hellofem::mesh — programmatic mesh generation
// SPDX-License-Identifier: MIT

#include "generation.h"

#include "mesh/utils.h"

#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hellofem::mesh {

    std::shared_ptr<Mesh<double>> create_rectangle(
        std::array<double, 2> p0, std::array<double, 2> p1,
        std::array<int, 2> n)
    {
        if (n[0] < 1 or n[1] < 1)
            throw std::invalid_argument("Intervals must be >= 1");

        const std::int64_t nx = n[0] + 1, ny = n[1] + 1;
        const auto v = [nx](std::int64_t i, std::int64_t j) {
            return i + j * nx;
        };

        // Two triangles per rectangle, diagonal (i, j) -> (i+1, j+1).
        std::vector<std::int64_t> cells;
        cells.reserve(6LL * n[0] * n[1]);
        for (std::int64_t j = 0; j < n[1]; ++j)
            for (std::int64_t i = 0; i < n[0]; ++i) {
                cells.insert(cells.end(),
                    {v(i, j), v(i + 1, j), v(i, j + 1)});
                cells.insert(cells.end(),
                    {v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)});
            }

        const std::int64_t num_nodes = nx * ny;
        std::vector<double> x(2 * static_cast<std::size_t>(num_nodes), 0.0);
        for (std::int64_t j = 0; j < ny; ++j)
            for (std::int64_t i = 0; i < nx; ++i) {
                x[2 * v(i, j) + 0]
                    = p0[0] + (p1[0] - p0[0]) * static_cast<double>(i) / n[0];
                x[2 * v(i, j) + 1]
                    = p0[1] + (p1[1] - p0[1]) * static_cast<double>(j) / n[1];
            }

        return std::make_shared<Mesh<double>>(create_mesh(
            std::span<const std::int64_t>(cells), CellType::triangle, x, 2));
    }

    std::shared_ptr<Mesh<double>> create_unit_square(int n)
    {
        return create_rectangle({0.0, 0.0}, {1.0, 1.0}, {n, n});
    }

    std::shared_ptr<Mesh<double>> create_box(
        std::array<double, 3> p0, std::array<double, 3> p1,
        std::array<int, 3> n)
    {
        if (n[0] < 1 or n[1] < 1 or n[2] < 1)
            throw std::invalid_argument("Intervals must be >= 1");

        const std::int64_t nx = n[0] + 1, ny = n[1] + 1, nz = n[2] + 1;
        const auto v = [nx, ny](std::int64_t i, std::int64_t j,
                          std::int64_t k) {
            return i + j * nx + k * nx * ny;
        };

        // One hexahedron per box element, basix vertex ordering.
        std::vector<std::int64_t> cells;
        cells.reserve(8LL * n[0] * n[1] * n[2]);
        for (std::int64_t k = 0; k < n[2]; ++k)
            for (std::int64_t j = 0; j < n[1]; ++j)
                for (std::int64_t i = 0; i < n[0]; ++i) {
                    cells.insert(cells.end(),
                        {v(i, j, k), v(i + 1, j, k), v(i, j + 1, k),
                            v(i + 1, j + 1, k), v(i, j, k + 1),
                            v(i + 1, j, k + 1), v(i, j + 1, k + 1),
                            v(i + 1, j + 1, k + 1)});
                }

        const std::int64_t num_nodes = nx * ny * nz;
        std::vector<double> x(3 * static_cast<std::size_t>(num_nodes), 0.0);
        for (std::int64_t k = 0; k < nz; ++k)
            for (std::int64_t j = 0; j < ny; ++j)
                for (std::int64_t i = 0; i < nx; ++i) {
                    const std::int64_t id = v(i, j, k);
                    x[3 * id + 0]
                        = p0[0] + (p1[0] - p0[0]) * static_cast<double>(i) / n[0];
                    x[3 * id + 1]
                        = p0[1] + (p1[1] - p0[1]) * static_cast<double>(j) / n[1];
                    x[3 * id + 2]
                        = p0[2] + (p1[2] - p0[2]) * static_cast<double>(k) / n[2];
                }

        return std::make_shared<Mesh<double>>(create_mesh(
            std::span<const std::int64_t>(cells), CellType::hexahedron, x, 3));
    }

} // namespace hellofem::mesh
