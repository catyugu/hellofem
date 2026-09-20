// hellofem::app tests — shared mesh fixtures and solve helpers
// SPDX-License-Identifier: MIT
#pragma once

#include "field.h"
#include "mesh/Mesh.h"
#include "mesh/MeshTags.h"
#include "mesh/generation.h"
#include "mesh/utils.h"
#include "mesh_loader.h"

#include <array>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace hellofem::app::test {

    /// 1-based boundary id for a 3D box face by constant coordinate:
    /// 1=x-,2=x+,3=y-,4=y+,5=z-,6=z+.
    ///
    /// The mesh is prepared for the app first (see `prepare_topology`): the
    /// tags read the facets and their neighbouring cells.
    inline std::shared_ptr<mesh::MeshTags<int>> boundary_tags(
        const mesh::Mesh<double>& mesh)
    {
        prepare_topology(mesh);
        auto topo = mesh.topology();
        const auto [vc, shape] = mesh::compute_vertex_coords(mesh);
        const std::size_t nv = shape[1];
        auto vert = [&](std::int32_t v, int d) { return vc[d * nv + v]; };

        auto f_to_c = topo->connectivity(2, 3);
        auto f_to_v = topo->connectivity(2, 0);
        std::vector<std::int32_t> idx;
        std::vector<int> vals;
        for (std::int32_t f = 0; f < f_to_v->num_nodes(); ++f) {
            if (f_to_c->num_links(f) != 1)
                continue; // interior
            auto vs = f_to_v->links(f);
            double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9, zmin = 1e9,
                   zmax = -1e9;
            for (auto v : vs) {
                xmin = std::min(xmin, vert(v, 0));
                xmax = std::max(xmax, vert(v, 0));
                ymin = std::min(ymin, vert(v, 1));
                ymax = std::max(ymax, vert(v, 1));
                zmin = std::min(zmin, vert(v, 2));
                zmax = std::max(zmax, vert(v, 2));
            }
            int id = 0;
            if (xmin == xmax)
                id = (xmin < 0.5) ? 1 : 2;
            else if (ymin == ymax)
                id = (ymin < 0.5) ? 3 : 4;
            else
                id = (zmin < 0.5) ? 5 : 6;
            idx.push_back(f);
            vals.push_back(id);
        }
        std::vector<std::size_t> order(idx.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(
            order, [&](std::size_t a, std::size_t b) { return idx[a] < idx[b]; });
        std::vector<std::int32_t> sidx;
        std::vector<int> svals;
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (i == 0 or idx[order[i]] != idx[order[i - 1]]) {
                sidx.push_back(idx[order[i]]);
                svals.push_back(vals[order[i]]);
            }
        }
        return std::make_shared<mesh::MeshTags<int>>(topo, 2,
            std::move(sidx), std::move(svals), "boundary");
    }

    /// A box mesh with the standard 6-face boundary tags and one domain.
    struct BoxFixture {
        std::shared_ptr<mesh::Mesh<double>> mesh;
        std::shared_ptr<mesh::MeshTags<int>> boundary;
        std::shared_ptr<mesh::MeshTags<int>> cells;
    };

    inline BoxFixture make_box_fixture(std::array<double, 3> lo,
        std::array<double, 3> hi, std::array<int, 3> n)
    {
        BoxFixture f;
        f.mesh = mesh::create_box(lo, hi, n);
        f.boundary = boundary_tags(*f.mesh);
        const std::size_t nc = f.mesh->topology()->index_map(3)->size_local();
        std::vector<std::int32_t> cell_idx(nc);
        std::iota(cell_idx.begin(), cell_idx.end(), 0);
        f.cells = std::make_shared<mesh::MeshTags<int>>(f.mesh->topology(), 3,
            std::move(cell_idx), std::vector<int>(nc, 1), "cells");
        return f;
    }

    /// A DG0 property holding a constant on domain 1 (the single domain of
    /// the box fixtures).
    inline std::shared_ptr<DomainProperty> constant_property(
        const std::shared_ptr<const mesh::Mesh<double>>& mesh,
        const std::shared_ptr<const mesh::MeshTags<int>>& tags, double value)
    {
        auto property = std::make_shared<DomainProperty>(mesh, tags,
            std::unordered_map<std::string, double> {});
        property->set_expression(1, std::to_string(value));
        property->update(0.0); // values are otherwise written on refresh
        return property;
    }

    /// A boundary property holding a constant on one boundary.
    inline std::shared_ptr<FacetProperty> constant_facet_property(
        const std::shared_ptr<const mesh::Mesh<double>>& mesh,
        const std::shared_ptr<const mesh::MeshTags<int>>& tags, int boundary,
        double value)
    {
        auto property = std::make_shared<FacetProperty>(mesh, tags, boundary,
            std::unordered_map<std::string, double> {});
        property->set_expression(std::to_string(value));
        property->update(0.0); // values are otherwise written on refresh
        return property;
    }

} // namespace hellofem::app::test
