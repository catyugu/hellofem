// hellofem::mesh — programmatic mesh generation
// SPDX-License-Identifier: MIT

#include "generation.h"

#include "basis/element-families.h"
#include "basis/finite-element.h"
#include "fem/CoordinateElement.h"

#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hellofem::mesh {

    std::shared_ptr<Mesh<double>> create_unit_square(int n)
    {
        if (n < 1)
            throw std::invalid_argument("n must be >= 1");

        const std::int64_t nx = n + 1;
        const auto v = [nx](std::int64_t i, std::int64_t j) { return i + j * nx; };

        // Two triangles per square, diagonal (i, j) -> (i+1, j+1).
        std::vector<std::int64_t> cells;
        std::vector<std::int64_t> orig;
        cells.reserve(6 * n * n);
        orig.reserve(2 * n * n);
        for (std::int64_t j = 0; j < n; ++j)
            for (std::int64_t i = 0; i < n; ++i) {
                cells.insert(cells.end(),
                    {v(i, j), v(i + 1, j), v(i, j + 1)});
                orig.push_back(0);
                cells.insert(cells.end(),
                    {v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)});
                orig.push_back(0);
            }

        auto topology = std::make_shared<Topology>(create_topology(
            std::span<const std::int64_t>(cells),
            std::span<const std::int64_t>(orig), CellType::triangle, 1));

        // Linear geometry: one node per vertex, P1 map. `create_geometry`
        // expects node coordinates flattened as (num_nodes, dim).
        const int dim = 2;
        const std::int64_t num_nodes = nx * nx;
        std::vector<double> x(static_cast<std::size_t>(num_nodes) * dim, 0.0);
        for (std::int64_t j = 0; j < nx; ++j)
            for (std::int64_t i = 0; i < nx; ++i) {
                x[dim * v(i, j) + 0] = static_cast<double>(i) / n;
                x[dim * v(i, j) + 1] = static_cast<double>(j) / n;
            }
        std::vector<std::int64_t> nodes(num_nodes);
        std::iota(nodes.begin(), nodes.end(), 0);

        auto c_to_v = topology->connectivity(2, 0);
        std::vector<std::int64_t> xdofs;
        xdofs.reserve(c_to_v->num_nodes() * 3);
        for (std::int32_t c = 0; c < c_to_v->num_nodes(); ++c)
            for (auto w : c_to_v->links(c))
                xdofs.push_back(w);

        fem::CoordinateElement<double> coord_el(
            CellType::triangle, 1, basis::element::lagrange_variant::equispaced);
        auto geometry = create_geometry(*topology,
            std::vector<fem::CoordinateElement<double>> {coord_el},
            std::span<const std::int64_t>(nodes),
            std::span<const std::int64_t>(xdofs), x, dim);

        return std::make_shared<Mesh<double>>(
            std::move(topology), std::move(geometry));
    }

} // namespace hellofem::mesh
