// hellofem::mesh — compute entity permutations and reflections
// SPDX-License-Identifier: MIT

#include "permutationcomputation.h"

#include "Topology.h"
#include "cell_types.h"
#include "common/Timer.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"

using namespace hellofem;
using namespace hellofem::mesh;

namespace {
    constexpr int _BITSETSIZE = 32;

    /// Number of rotations and reflections to map the local orientation
    /// of a triangle facet to its global orientation. The local
    /// orientation is described by `e_vertices` (local cell vertex
    /// indices on the facet) and `vertices` (the global vertex indices
    /// of the same facet).
    std::pair<std::int8_t, std::int8_t>
    compute_triangle_rot_reflect(const std::vector<std::int32_t>& e_vertices,
        const std::vector<std::int64_t>& vertices)
    {
        // Number of rotations.
        const std::uint8_t min_v = std::ranges::distance(
            e_vertices.begin(), std::ranges::min_element(e_vertices));

        // pre is the (local) number of the next vertex clockwise from the
        // lowest numbered vertex; post is the next vertex anticlockwise.
        const int pre = e_vertices[(min_v + 2) % 3];
        const int post = e_vertices[(min_v + 1) % 3];

        const std::uint8_t g_min_v = std::ranges::distance(
            vertices.begin(), std::ranges::min_element(vertices));
        const int g_pre = vertices[(g_min_v + 2) % 3];
        const int g_post = vertices[(g_min_v + 1) % 3];

        std::uint8_t rots = 0;
        if (g_post > g_pre)
            rots = (g_min_v + 3 - min_v) % 3;
        else
            rots = (min_v + 3 - g_min_v) % 3;

        return {(post > pre) == (g_post < g_pre), rots};
    }

    /// Same as compute_triangle_rot_reflect, for quadrilateral facets.
    std::pair<std::int8_t, std::int8_t>
    compute_quad_rot_reflect(const std::vector<std::int32_t>& e_vertices,
        const std::vector<std::int64_t>& vertices)
    {
        // Find minimum local cell vertex on the facet.
        const std::uint8_t min_v = std::ranges::distance(
            e_vertices.begin(), std::ranges::min_element(e_vertices));

        // Table of previous vertices: 0 - 2 / 1 - 3.
        const std::array<std::int8_t, 4> prev = {2, 0, 3, 1};

        const std::int32_t pre = e_vertices[prev[min_v]];
        const std::int32_t post = e_vertices[prev[3 - min_v]];

        // If min_v is 2 or 3, swap to compute the number of anticlockwise
        // rotations correctly.
        std::uint8_t min_v_rot = min_v;
        if (min_v_rot == 2 or min_v_rot == 3)
            min_v_rot = 5 - min_v_rot;

        const std::uint8_t g_min_v = std::ranges::distance(
            vertices.begin(), std::ranges::min_element(vertices));
        const std::int64_t g_pre = vertices[prev[g_min_v]];
        const std::int64_t g_post = vertices[prev[3 - g_min_v]];

        std::uint8_t g_min_v_rot = g_min_v;
        if (g_min_v_rot == 2 or g_min_v_rot == 3)
            g_min_v_rot = 5 - g_min_v_rot;

        std::uint8_t rots = 0;
        if (g_post > g_pre)
            rots = (g_min_v_rot - min_v_rot + 4) % 4;
        else
            rots = (min_v_rot - g_min_v_rot + 4) % 4;
        return {(post > pre) == (g_post < g_pre), rots};
    }

    /// Compute the permutations of the triangle/quadrilateral facets of
    /// the cells of cell type `cell_index`.
    template <int BITSETSIZE>
    std::vector<std::bitset<BITSETSIZE>>
    compute_triangle_quad_face_permutations(
        const Topology& topology, int cell_index)
    {
        common::Timer t_perm("* Compute triangle/quad face permutations");
        const std::vector<CellType>& cell_types = topology.entity_types(3);
        const CellType cell_type = cell_types.at(cell_index);

        const std::vector<CellType>& mesh_face_types = topology.entity_types(2);
        std::vector<CellType> cell_face_types(cell_num_entities(cell_type, 2));
        for (std::size_t i = 0; i < cell_face_types.size(); ++i)
            cell_face_types[i] = cell_facet_type(cell_type, i);

        std::vector<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>>
            c_to_f;
        std::vector<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>>
            f_to_v;

        const int tdim = topology.dim();
        std::vector<std::vector<int>> face_type_indices(mesh_face_types.size());
        for (std::size_t i = 0; i < mesh_face_types.size(); ++i) {
            for (std::size_t j = 0; j < cell_face_types.size(); ++j)
                if (mesh_face_types[i] == cell_face_types[j])
                    face_type_indices[i].push_back(j);
            c_to_f.push_back(topology.connectivity({tdim, cell_index}, {2, int(i)}));
            f_to_v.push_back(topology.connectivity({2, int(i)}, {0, 0}));
        }

        auto c_to_v = topology.connectivity({tdim, cell_index}, {0, 0});
        assert(c_to_v);

        const std::int32_t num_cells = c_to_v->num_nodes();
        std::vector<std::bitset<BITSETSIZE>> face_perm(num_cells, 0);
        auto im = topology.index_map(0);

        for (std::size_t t = 0; t < face_type_indices.size(); ++t) {
            spdlog::info("Computing permutations for face type {}", t);
            if (face_type_indices[t].empty())
                continue;

            auto compute_refl_rots
                = (mesh_face_types[t] == CellType::triangle)
                ? compute_triangle_rot_reflect
                : compute_quad_rot_reflect;

            tbb::parallel_for(
                tbb::blocked_range<std::int32_t>(0, num_cells),
                [&, compute_refl_rots](
                    const tbb::blocked_range<std::int32_t>& range) {
                    std::vector<std::int64_t> cell_vertices, vertices;
                    std::vector<std::int32_t> e_vertices;
                    for (std::int32_t c = range.begin(); c < range.end(); ++c) {
                        cell_vertices.resize(c_to_v->links(c).size());
                        im->local_to_global(c_to_v->links(c), cell_vertices);
                        auto cell_faces = c_to_f[t]->links(c);
                        for (std::size_t j = 0; j < cell_faces.size(); ++j) {
                            // Get the face.
                            const int face = cell_faces[j];
                            e_vertices.resize(f_to_v[t]->num_links(face));
                            vertices.resize(f_to_v[t]->num_links(face));
                            im->local_to_global(f_to_v[t]->links(face), vertices);

                            // Find the local cell vertex indices of the
                            // face vertices.
                            for (std::size_t k = 0; k < vertices.size(); ++k) {
                                auto it = std::ranges::find(
                                    cell_vertices, vertices[k]);
                                assert(it != cell_vertices.end());
                                e_vertices[k] = std::ranges::distance(
                                    cell_vertices.begin(), it);
                            }

                            auto [refl, rots] = compute_refl_rots(e_vertices, vertices);

                            const int fi = face_type_indices[t][j];
                            face_perm[c][3 * fi] = refl;
                            face_perm[c][3 * fi + 1] = rots % 2;
                            face_perm[c][3 * fi + 2] = rots / 2;
                        }
                    }
                },
                tbb::simple_partitioner {});
        }

        return face_perm;
    }

    /// Compute the reflections of the edges of the cells.
    template <int BITSETSIZE>
    std::vector<std::bitset<BITSETSIZE>>
    compute_edge_reflections(const Topology& topology)
    {
        common::Timer t_perm("* Compute edge reflections");

        const CellType cell_type = topology.cell_type();
        const int tdim = topology.dim();
        const int edges_per_cell = cell_num_entities(cell_type, 1);

        const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();

        auto c_to_v = topology.connectivity(tdim, 0);
        assert(c_to_v);
        auto c_to_e = topology.connectivity(tdim, 1);
        assert(c_to_e);
        auto e_to_v = topology.connectivity(1, 0);
        assert(e_to_v);

        auto im = topology.index_map(0);
        assert(im);

        std::vector<std::bitset<BITSETSIZE>> edge_perm(num_cells, 0);

        tbb::parallel_for(
            tbb::blocked_range<std::int32_t>(0, num_cells),
            [&](const tbb::blocked_range<std::int32_t>& range) {
                std::vector<std::int64_t> cell_vertices;
                std::vector<std::int64_t> vertices;
                for (int c = range.begin(); c < range.end(); ++c) {
                    cell_vertices.resize(c_to_v->num_links(c));
                    im->local_to_global(c_to_v->links(c), cell_vertices);
                    auto cell_edges = c_to_e->links(c);
                    for (int edge = 0; edge < edges_per_cell; ++edge) {
                        vertices.resize(e_to_v->links(cell_edges[edge]).size());
                        im->local_to_global(
                            e_to_v->links(cell_edges[edge]), vertices);

                        // The entity is an interval: orient it pointing from
                        // the lowest to the highest vertex.
                        auto it0 = std::ranges::find(cell_vertices, vertices[0]);
                        auto it1 = std::ranges::find(cell_vertices, vertices[1]);
                        edge_perm[c][edge]
                            = (it1 < it0) == (vertices[1] > vertices[0]);
                    }
                }
            },
            tbb::simple_partitioner {});

        return edge_perm;
    }

    template <int BITSETSIZE>
    std::vector<std::bitset<BITSETSIZE>>
    compute_face_permutations(const Topology& topology)
    {
        if (topology.entity_types(3).size() > 1)
            throw std::runtime_error(
                "Cannot compute permutations for mixed topology mesh.");

        [[maybe_unused]] const int tdim = topology.dim();
        assert(tdim > 2);
        if (!topology.index_map(2))
            throw std::runtime_error("Faces have not been computed.");

        return compute_triangle_quad_face_permutations<BITSETSIZE>(topology, 0);
    }

} // namespace

//-----------------------------------------------------------------------------
std::pair<std::vector<std::uint8_t>, std::vector<std::uint32_t>>
mesh::compute_entity_permutations(const Topology& topology, int num_threads)
{
    if (num_threads < 1)
        throw std::runtime_error("num_threads must be >= 1.");

    common::Timer t_perm("Compute entity permutations");

    const int tdim = topology.dim();
    const CellType cell_type = topology.cell_type();
    const std::int32_t num_cells = topology.connectivity(tdim, 0)->num_nodes();
    const int facets_per_cell
        = (tdim > 0) ? cell_num_entities(cell_type, tdim - 1) : 0;

    std::vector<std::uint32_t> cell_permutation_info(num_cells, 0);
    std::vector<std::uint8_t> facet_permutations(num_cells * facets_per_cell);
    std::int32_t used_bits = 0;
    if (tdim > 2) {
        spdlog::info("Compute face permutations");
        const int faces_per_cell = cell_num_entities(cell_type, 2);
        const auto face_perm = compute_face_permutations<_BITSETSIZE>(topology);
        for (int c = 0; c < num_cells; ++c)
            cell_permutation_info[c] = face_perm[c].to_ulong();

        // 3 bits are used per face.
        used_bits += faces_per_cell * 3;
        assert(tdim == 3);
        for (int c = 0; c < num_cells; ++c)
            for (int i = 0; i < facets_per_cell; ++i)
                facet_permutations[c * facets_per_cell + i]
                    = (cell_permutation_info[c] >> (3 * i)) & 7;
    }

    if (tdim > 1) {
        spdlog::info("Compute edge permutations");
        const int edges_per_cell = cell_num_entities(cell_type, 1);
        const auto edge_perm = compute_edge_reflections<_BITSETSIZE>(topology);
        for (int c = 0; c < num_cells; ++c)
            cell_permutation_info[c] |= edge_perm[c].to_ulong() << used_bits;

        used_bits += edges_per_cell;
        if (tdim == 2) {
            for (int c = 0; c < num_cells; ++c)
                for (int i = 0; i < facets_per_cell; ++i)
                    facet_permutations[c * facets_per_cell + i]
                        = edge_perm[c][i];
        }
    }
    assert(used_bits < _BITSETSIZE);

    return {std::move(facet_permutations), std::move(cell_permutation_info)};
}
//-----------------------------------------------------------------------------
