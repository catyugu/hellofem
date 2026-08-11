// hellofem::mesh — mesh topology (entities and connectivity)
// SPDX-License-Identifier: MIT

#include "Topology.h"

#include "cell_types.h"
#include "common/Timer.h"
#include "permutationcomputation.h"
#include "spdlog/spdlog.h"
#include "topologycomputation.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <format>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace hellofem;
using namespace hellofem::mesh;

namespace {

    /// Entity types present in a mesh given its cell types:
    /// `entity_types[d]` is the list of distinct entity types of
    /// dimension `d`.
    std::vector<std::vector<CellType>>
    build_entity_types(const std::vector<CellType>& cell_types)
    {
        const int tdim = cell_dim(cell_types.front());
        std::vector<std::vector<CellType>> entity_types(tdim + 1);

        entity_types[0] = {CellType::point};
        entity_types[tdim] = cell_types;
        if (tdim > 1)
            entity_types[1] = {CellType::interval};
        if (tdim > 2) {
            std::set<CellType> e_types;
            for (auto c : entity_types[tdim])
                for (int i = 0; i < cell_num_entities(c, 2); ++i)
                    e_types.insert(cell_facet_type(c, i));
            entity_types[2] = std::vector(e_types.begin(), e_types.end());
        }
        return entity_types;
    }

    /// Map each global vertex index appearing in `cells` to a
    /// contiguous local index. If the vertex indices are contiguous
    /// starting at 0 (the common single-process case), the map is the
    /// identity and no lookups are needed.
    ///
    /// @return The number of vertices and a mapping function.
    struct VertexRenumber {
        // Number of distinct vertices.
        std::int32_t num_vertices;
        // Identity when true (local index == global index).
        bool is_identity;
        // Global -> local map (non-identity case only).
        std::unordered_map<std::int64_t, std::int32_t> map;

        std::int32_t operator()(std::int64_t g) const
        {
            if (is_identity)
                return static_cast<std::int32_t>(g);
            auto it = map.find(g);
            assert(it != map.end());
            return it->second;
        }
    };

    VertexRenumber
    renumber_vertices(const std::vector<std::span<const std::int64_t>>& cells,
        std::int32_t num_cell_vertices_total)
    {
        // Collect all vertices and sort them.
        std::vector<std::int64_t> vertices;
        vertices.reserve(num_cell_vertices_total);
        for (auto c : cells)
            vertices.insert(vertices.end(), c.begin(), c.end());
        std::ranges::sort(vertices);
        auto [unique_end, range_end] = std::ranges::unique(vertices);
        vertices.erase(unique_end, range_end);

        const std::int32_t n = static_cast<std::int32_t>(vertices.size());
        if (vertices.empty())
            return {0, true, {}};

        const bool is_identity = vertices.front() == 0
            and vertices.back() == static_cast<std::int64_t>(n) - 1;
        if (is_identity)
            return {n, true, {}};

        VertexRenumber renum {n, false, {}};
        renum.map.reserve(vertices.size());
        for (std::size_t i = 0; i < vertices.size(); ++i)
            renum.map.emplace(vertices[i], static_cast<std::int32_t>(i));
        return renum;
    }

} // namespace

//-----------------------------------------------------------------------------
Topology::Topology(std::vector<CellType> cell_types,
    std::shared_ptr<const common::IndexMap> vertex_map,
    std::vector<std::shared_ptr<const common::IndexMap>> cell_maps,
    std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> cells,
    const std::optional<std::vector<std::vector<std::int64_t>>>& original_index,
    [[maybe_unused]] int num_threads)
    : original_cell_index(
          original_index ? *original_index
                         : std::vector<std::vector<std::int64_t>>())
    , _entity_types(build_entity_types(cell_types))
{
    assert(!cell_types.empty());
    const int tdim = cell_dim(cell_types.front());
#ifndef NDEBUG
    for (auto ct : cell_types)
        assert(cell_dim(ct) == tdim);
#endif

    _index_maps.insert({{0, 0}, std::move(vertex_map)});
    _connectivity.insert(
        {{{0, 0}, {0, 0}},
            std::make_shared<graph::AdjacencyList<std::int32_t>>(
                _index_maps.at({0, 0})->size_local()
                + _index_maps.at({0, 0})->num_ghosts())});
    if (tdim > 0) {
        for (std::size_t i = 0; i < cell_types.size(); ++i) {
            _index_maps.insert({{tdim, (int)i}, cell_maps[i]});
            _connectivity.insert(
                {{{tdim, int(i)}, {0, 0}}, cells[i]});
        }
    }
}
//-----------------------------------------------------------------------------
int Topology::dim() const noexcept { return cell_dim(_entity_types.back().front()); }
//-----------------------------------------------------------------------------
const std::vector<CellType>& Topology::entity_types(int dim) const
{
    return _entity_types.at(dim);
}
//-----------------------------------------------------------------------------
CellType Topology::cell_type() const
{
    std::vector<CellType> cell_types = entity_types(this->dim());
    if (cell_types.size() > 1)
        throw std::runtime_error(
            "Multiple cell types of this dimension. Call cell_types instead.");
    return cell_types.front();
}
//-----------------------------------------------------------------------------
std::vector<CellType> Topology::cell_types() const { return entity_types(dim()); }
//-----------------------------------------------------------------------------
std::vector<std::shared_ptr<const common::IndexMap>>
Topology::index_maps(int dim) const
{
    std::vector<std::shared_ptr<const common::IndexMap>> maps;
    for (std::size_t i = 0; i < _entity_types[dim].size(); ++i) {
        auto it = _index_maps.find({dim, int(i)});
        if (it != _index_maps.end())
            maps.push_back(it->second);
    }
    return maps;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Topology::index_map(int dim) const
{
    if (_entity_types[dim].size() > 1)
        throw std::runtime_error(
            "Multiple index maps of this dimension. Call index_maps instead.");

    std::vector<std::shared_ptr<const common::IndexMap>> im
        = this->index_maps(dim);
    if (im.empty())
        throw std::runtime_error(std::format(
            "Missing IndexMap in Topology. Maybe you need to create_entities({}).",
            dim));
    return im.at(0);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(std::array<int, 2> d0, std::array<int, 2> d1) const
{
    if (auto it = _connectivity.find({d0, d1}); it == _connectivity.end())
        return nullptr;
    else
        return it->second;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const graph::AdjacencyList<std::int32_t>>
Topology::connectivity(int d0, int d1) const
{
    if (this->entity_types(d0).size() > 1 or this->entity_types(d1).size() > 1)
        throw std::runtime_error(
            "Multiple entity types in mesh. Call connectivity specifying "
            "entity type.");
    return this->connectivity({d0, 0}, {d1, 0});
}
//-----------------------------------------------------------------------------
const std::vector<std::uint32_t>& Topology::get_cell_permutation_info() const
{
    assert(this->index_map(this->dim()));
    if (auto i_map = this->index_map(this->dim());
        _cell_permutations.empty()
        and i_map->size_local() + i_map->num_ghosts() > 0) {
        throw std::runtime_error(
            "create_entity_permutations must be called before using this data.");
    }
    return _cell_permutations;
}
//-----------------------------------------------------------------------------
const std::vector<std::uint8_t>& Topology::get_facet_permutations() const
{
    if (auto i_map = this->index_map(this->dim() - 1);
        !i_map
        or (_facet_permutations.empty()
            and (i_map->size_local() + i_map->num_ghosts() > 0))) {
        throw std::runtime_error(
            "create_entity_permutations must be called before using this data.");
    }
    return _facet_permutations;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& Topology::interprocess_facets(int index) const
{
    if (_interprocess_facets.empty())
        throw std::runtime_error("Interprocess facets have not been computed.");
    return _interprocess_facets.at(index);
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& Topology::interprocess_facets() const
{
    return this->interprocess_facets(0);
}
//-----------------------------------------------------------------------------
bool Topology::create_entities(int dim, int num_threads)
{
    // Skip if already computed (vertices always exist).
    bool entities_created = true;
    for (int ent_type_idx = 0, num_ent_types = this->entity_types(dim).size();
        ent_type_idx < num_ent_types; ++ent_type_idx) {
        if (!this->connectivity({dim, ent_type_idx}, {0, 0})) {
            entities_created = false;
            break;
        }
    }
    if (entities_created)
        return false;

    for (auto entity = this->entity_types(dim).begin();
        entity != this->entity_types(dim).end(); ++entity) {
        const int index
            = std::ranges::distance(this->entity_types(dim).begin(), entity);

        auto [cell_entity, entity_vertex, index_map, interprocess_entities]
            = compute_entities(*this, dim, *entity, num_threads);
        for (std::size_t k = 0; k < cell_entity.size(); ++k) {
            if (cell_entity[k])
                _connectivity.insert(
                    {{{this->dim(), int(k)}, {dim, index}}, cell_entity[k]});
        }

        if (entity_vertex)
            _connectivity.insert({{{dim, index}, {0, 0}}, entity_vertex});

        _index_maps.insert({{dim, index}, index_map});

        if (dim == this->dim() - 1) {
            std::ranges::sort(interprocess_entities);
            _interprocess_facets.push_back(std::move(interprocess_entities));
        }
    }

    return true;
}
//-----------------------------------------------------------------------------
void Topology::create_connectivity(int d0, int d1)
{
    this->create_entities(d0);
    this->create_entities(d1);

    const int num_d0 = this->entity_types(d0).size();
    const int num_d1 = this->entity_types(d1).size();

    for (int i0 = 0; i0 < num_d0; ++i0) {
        for (int i1 = 0; i1 < num_d1; ++i1) {
            auto [c_d0_d1, c_d1_d0]
                = compute_connectivity(*this, {d0, i0}, {d1, i1});
            if (c_d0_d1)
                _connectivity.insert({{{d0, i0}, {d1, i1}}, c_d0_d1});
            if (c_d1_d0)
                _connectivity.insert({{{d1, i1}, {d0, i0}}, c_d1_d0});
        }
    }
}
//-----------------------------------------------------------------------------
void Topology::create_entity_permutations(int num_threads)
{
    if (!_cell_permutations.empty())
        return;

    const int tdim = this->dim();
    for (int d = 0; d < tdim; ++d)
        create_entities(d, num_threads);

    auto [facet_permutations, cell_permutations]
        = compute_entity_permutations(*this, num_threads);
    _facet_permutations = std::move(facet_permutations);
    _cell_permutations = std::move(cell_permutations);
}
//-----------------------------------------------------------------------------
Topology mesh::create_topology(std::vector<CellType> cell_types,
    std::vector<std::span<const std::int64_t>> cells,
    std::vector<std::span<const std::int64_t>> original_cell_index,
    int num_threads)
{
    if (num_threads < 1)
        throw std::runtime_error("num_threads must be >= 1.");

    common::Timer timer("Topology: create");
    spdlog::info("Create topology (generalised)");

    assert(cell_types.size() == cells.size());
    assert(original_cell_index.size() == cells.size());

    // Check cell data consistency.
    std::vector<std::int32_t> num_local_cells(cell_types.size());
    std::int32_t num_cell_vertex_entries = 0;
    for (std::size_t i = 0; i < cell_types.size(); ++i) {
        const int num_vertices = num_cell_vertices(cell_types[i]);
        if (cells[i].size() % num_vertices != 0)
            throw std::runtime_error(std::format(
                "Inconsistent number of cell vertices. Got {}, expected "
                "multiple of {}.",
                cells[i].size(), num_vertices));
        num_local_cells[i] = cells[i].size() / num_vertices;
        num_cell_vertex_entries += cells[i].size();
    }

    // Renumber the global vertex indices to a contiguous local range.
    // Every vertex is owned by the single process, so this is a plain
    // global->local renumbering (the identity map when the input is
    // already contiguous from 0).
    const VertexRenumber renum
        = renumber_vertices(cells, num_cell_vertex_entries);

    auto index_map_v
        = std::make_shared<common::IndexMap>(0, renum.num_vertices);

    // Build local cell-vertex connectivity.
    std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>> cells_c;
    cells_c.reserve(cell_types.size());
    std::vector<std::shared_ptr<const common::IndexMap>> index_map_c;
    std::int32_t cell_offset = 0;
    for (std::size_t i = 0; i < cell_types.size(); ++i) {
        const int num_vertices = num_cell_vertices(cell_types[i]);
        std::vector<std::int32_t> local_cells;
        local_cells.reserve(cells[i].size());
        std::ranges::transform(
            cells[i], std::back_inserter(local_cells),
            [&renum](std::int64_t g) { return renum(g); });

        cells_c.push_back(std::make_shared<graph::AdjacencyList<std::int32_t>>(
            graph::regular_adjacency_list(std::move(local_cells), num_vertices)));
        index_map_c.push_back(
            std::make_shared<common::IndexMap>(cell_offset, num_local_cells[i]));
        cell_offset += num_local_cells[i];
    }

    // Save original cell index.
    std::vector<std::vector<std::int64_t>> orig_index;
    orig_index.reserve(original_cell_index.size());
    for (auto idx : original_cell_index)
        orig_index.push_back(std::vector<std::int64_t>(idx.begin(), idx.end()));

    return Topology(std::move(cell_types), std::move(index_map_v),
        std::move(index_map_c), std::move(cells_c), std::move(orig_index));
}
//-----------------------------------------------------------------------------
Topology mesh::create_topology(std::span<const std::int64_t> cells,
    std::span<const std::int64_t> original_cell_index, CellType cell_type,
    int num_threads)
{
    spdlog::info("Create topology (single cell type)");
    return create_topology({cell_type}, {cells}, {original_cell_index},
        num_threads);
}
//-----------------------------------------------------------------------------
std::tuple<Topology, std::vector<std::int32_t>, std::vector<std::int32_t>>
mesh::create_subtopology(const Topology& topology, int dim,
    std::span<const std::int32_t> entities)
{
    // Create a map from an entity in the sub-topology to the
    // corresponding entity in the topology.
    std::shared_ptr<common::IndexMap> submap;
    std::vector<std::int32_t> subentities;
    {
        std::vector<std::int32_t> _entities(entities.begin(), entities.end());
        std::ranges::sort(_entities);
        auto [unique_end, range_end] = std::ranges::unique(_entities);
        _entities.erase(unique_end, range_end);

        auto [_submap, _subentities]
            = common::create_sub_index_map(*topology.index_map(dim), _entities);
        submap = std::make_shared<common::IndexMap>(std::move(_submap));
        subentities = std::move(_subentities);
    }

    // Create the vertex map and sub-vertices.
    auto map0 = topology.index_map(0);
    assert(map0);
    std::shared_ptr<common::IndexMap> submap0;
    std::vector<std::int32_t> subvertices0;
    {
        std::pair<common::IndexMap, std::vector<std::int32_t>> map_data
            = common::create_sub_index_map(
                *map0, compute_incident_entities(topology, subentities, dim, 0),
                common::IndexMapOrder::any);
        submap0 = std::make_shared<common::IndexMap>(std::move(map_data.first));
        subvertices0 = std::move(map_data.second);
    }

    // Sub-topology entity-to-vertex connectivity.
    const CellType entity_type = cell_entity_type(topology.cell_type(), dim, 0);
    const int num_vertices_per_entity = cell_num_entities(entity_type, 0);
    auto e_to_v = topology.connectivity(dim, 0);
    assert(e_to_v);
    std::vector<std::int32_t> sub_e_to_v_vec;
    sub_e_to_v_vec.reserve(subentities.size() * num_vertices_per_entity);
    std::vector<std::int32_t> sub_e_to_v_offsets(1, 0);
    sub_e_to_v_offsets.reserve(subentities.size() + 1);

    // Vertex-to-subvertex map (inverse of subvertex_to_vertex).
    std::vector<std::int32_t> vertex_to_subvertex(
        map0->size_local() + map0->num_ghosts(), -1);
    for (std::size_t i = 0; i < subvertices0.size(); ++i)
        vertex_to_subvertex[subvertices0[i]] = i;

    for (std::int32_t e : subentities) {
        for (std::int32_t v : e_to_v->links(e)) {
            const std::int32_t v_sub = vertex_to_subvertex[v];
            assert(v_sub != -1);
            sub_e_to_v_vec.push_back(v_sub);
        }
        sub_e_to_v_offsets.push_back(sub_e_to_v_vec.size());
    }

    auto sub_e_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
        std::move(sub_e_to_v_vec), std::move(sub_e_to_v_offsets));

    return {Topology({entity_type}, submap0, {submap}, {sub_e_to_v}),
        std::move(subentities), std::move(subvertices0)};
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t>
mesh::entities_to_index(const Topology& topology, int dim,
    std::span<const std::int32_t> entities)
{
    spdlog::info("Build list of mesh entity indices from the entity vertices.");

    auto map_e = topology.index_map(dim);
    if (!map_e)
        throw std::runtime_error(
            std::format("Mesh entities of dimension {} have not been created.", dim));

    auto e_to_v = topology.connectivity(dim, 0);
    assert(e_to_v);

    const int num_vertices_per_entity
        = cell_num_entities(cell_entity_type(topology.cell_type(), dim, 0), 0);

    // Map from sorted local vertex indices (key) to entity index.
    std::map<std::vector<std::int32_t>, std::int32_t> entity_key_to_index;
    std::vector<std::int32_t> key(num_vertices_per_entity);
    for (std::int32_t e = 0; e < map_e->size_local() + map_e->num_ghosts(); ++e) {
        auto vertices = e_to_v->links(e);
        std::ranges::copy(vertices, key.begin());
        std::ranges::sort(key);
        auto ins = entity_key_to_index.insert({key, e});
        if (!ins.second)
            throw std::runtime_error("Duplicate mesh entity detected.");
    }

    assert(entities.size() % num_vertices_per_entity == 0);

    std::vector<std::int32_t> indices;
    indices.reserve(entities.size() / num_vertices_per_entity);
    std::vector<std::int32_t> vertices(num_vertices_per_entity);
    for (std::size_t e = 0; e < entities.size(); e += num_vertices_per_entity) {
        auto v = entities.subspan(e, num_vertices_per_entity);
        std::ranges::copy(v, vertices.begin());
        std::ranges::sort(vertices);
        if (auto it = entity_key_to_index.find(vertices); it != entity_key_to_index.end())
            indices.push_back(it->second);
        else
            indices.push_back(-1);
    }

    return indices;
}
//-----------------------------------------------------------------------------
std::vector<std::vector<std::int32_t>>
mesh::compute_mixed_cell_pairs(const Topology& topology, CellType facet_type)
{
    const int tdim = topology.dim();
    const std::vector<CellType>& cell_types = topology.entity_types(tdim);
    const std::vector<CellType>& facet_types = topology.entity_types(tdim - 1);

    int facet_index = -1;
    for (std::size_t i = 0; i < facet_types.size(); ++i) {
        if (facet_types[i] == facet_type) {
            facet_index = i;
            break;
        }
    }
    if (facet_index == -1)
        throw std::runtime_error("Cannot find facet type in topology");

    std::vector<std::vector<std::int32_t>> facet_pair_lists;
    for (std::size_t i = 0; i < cell_types.size(); ++i) {
        for (std::size_t j = 0; j < cell_types.size(); ++j) {
            std::vector<std::int32_t> facet_pairs_ij;
            auto fci = topology.connectivity(
                {tdim - 1, facet_index}, {tdim, static_cast<int>(i)});
            auto cfi = topology.connectivity(
                {tdim, static_cast<int>(i)}, {tdim - 1, facet_index});

            auto local_facet = [](const auto& cf, std::int32_t c, std::int32_t f) {
                auto it = std::find(cf->links(c).begin(), cf->links(c).end(), f);
                assert(it != cf->links(c).end()
                    && "Facet-cell and cell-facet connectivity are inconsistent.");
                return std::ranges::distance(cf->links(c).begin(), it);
            };

            if (i == j) {
                if (fci) {
                    for (std::int32_t k = 0; k < fci->num_nodes(); ++k) {
                        if (fci->num_links(k) == 2) {
                            const std::int32_t c0 = fci->links(k)[0];
                            const std::int32_t c1 = fci->links(k)[1];
                            facet_pairs_ij.insert(facet_pairs_ij.end(),
                                {c0, static_cast<std::int32_t>(local_facet(cfi, c0, k)),
                                    c1, static_cast<std::int32_t>(local_facet(cfi, c1, k))});
                        }
                    }
                }
            }
            else {
                auto fcj = topology.connectivity(
                    {tdim - 1, facet_index}, {tdim, static_cast<int>(j)});
                auto cfj = topology.connectivity(
                    {tdim, static_cast<int>(j)}, {tdim - 1, facet_index});
                if (fci and fcj) {
                    assert(fci->num_nodes() == fcj->num_nodes());
                    for (std::int32_t k = 0; k < fci->num_nodes(); ++k) {
                        if (fci->num_links(k) == 1 and fcj->num_links(k) == 1) {
                            const std::int32_t ci = fci->links(k)[0];
                            const std::int32_t cj = fcj->links(k)[0];
                            facet_pairs_ij.insert(facet_pairs_ij.end(),
                                {ci, static_cast<std::int32_t>(local_facet(cfi, ci, k)),
                                    cj, static_cast<std::int32_t>(local_facet(cfj, cj, k))});
                        }
                    }
                }
            }
            facet_pair_lists.push_back(std::move(facet_pairs_ij));
        }
    }

    return facet_pair_lists;
}
//-----------------------------------------------------------------------------
