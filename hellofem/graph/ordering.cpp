// hellofem::graph — Reverse Cuthill-McKee ordering implementation
// SPDX-License-Identifier: MIT

#include "ordering.h"

#include "../common/Timer.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

using namespace hellofem;
using namespace hellofem::graph;

namespace {
    // Level structure of `graph` rooted at node `s`: nodes grouped by BFS
    // distance from `s`.
    AdjacencyList<int> create_level_structure(const AdjacencyList<int>& graph,
                                              int s)
    {
        common::Timer t("Graph: create_level_structure");

        std::vector<std::int8_t> labelled(graph.num_nodes(), false);
        labelled[s] = true;

        int l = 0;
        std::vector<int> level_offsets{0};
        level_offsets.reserve(graph.offsets().size());
        std::vector<int> level_structure = {s};
        level_structure.reserve(graph.array().size());
        while (static_cast<int>(level_structure.size()) > level_offsets.back()) {
            level_offsets.push_back(level_structure.size());
            for (int i = level_offsets[l]; i < level_offsets[l + 1]; ++i) {
                const int node = level_structure[i];
                for (int idx : graph.links(node)) {
                    if (labelled[idx])
                        continue;
                    level_structure.push_back(idx);
                    labelled[idx] = true;
                }
            }
            ++l;
        }

        return AdjacencyList(std::move(level_structure),
                             std::move(level_offsets));
    }

    // RCM reordering of the unlabelled part of the graph. `rlabel` holds -1
    // for nodes not yet assigned an ordering.
    std::vector<std::int32_t>
    rcm_reorder_unlabelled(const AdjacencyList<std::int32_t>& graph,
                           std::span<const std::int32_t> rlabel)
    {
        common::Timer timer("Reverse Cuthill-McKee ordering");

        const std::int32_t n = graph.num_nodes();

        auto cmp_degree = [&graph](auto a, auto b) {
            return graph.num_links(a) < graph.num_links(b);
        };

        // Start from an unlabelled vertex of minimal degree.
        std::int32_t v = 0;
        std::int32_t dmin = std::numeric_limits<std::int32_t>::max();
        for (std::int32_t i = 0; i < n; ++i) {
            if (int d = graph.num_links(i); rlabel[i] == -1 and d < dmin) {
                v = i;
                dmin = d;
            }
        }

        // George-Liu double sweep: move to the minimum-degree vertex of the
        // deepest level until the level structure stops growing.
        AdjacencyList<int> lv = create_level_structure(graph, v);
        bool done = false;
        while (!done) {
            auto lv_final = lv.links(lv.num_nodes() - 1);
            int s = *std::ranges::min_element(lv_final, cmp_degree);
            AdjacencyList<int> lstmp = create_level_structure(graph, s);
            if (lstmp.num_nodes() > lv.num_nodes()) {
                v = s;
                lv = std::move(lstmp);
            }
            else
                done = true;
        }

        // BFS from the root, appending each vertex's not-yet-visited
        // neighbours in increasing degree order (the discovery order the
        // standard algorithm relies on).
        std::vector<std::int8_t> labelled(n, false);
        std::vector<int> rv;
        rv.reserve(n);
        rv.push_back(v);
        labelled[v] = true;

        std::vector<int> nbr;
        for (std::size_t current = 0; current < rv.size(); ++current) {
            nbr.clear();
            for (int w : graph.links(rv[current])) {
                if (!labelled[w]) {
                    nbr.push_back(w);
                    labelled[w] = true;
                }
            }
            std::ranges::sort(nbr, cmp_degree);
            rv.insert(rv.end(), nbr.begin(), nbr.end());
        }

        std::ranges::reverse(rv);
        return rv;
    }
} // namespace

std::vector<std::int32_t>
graph::reorder_rcm(const AdjacencyList<std::int32_t>& graph)
{
    const std::int32_t n = graph.num_nodes();
    std::vector<std::int32_t> r(n, -1);

    // Repeat for each disconnected component.
    int count = 0;
    while (count < n) {
        std::vector<std::int32_t> rv = rcm_reorder_unlabelled(graph, r);
        assert(!rv.empty());
        for (std::int32_t q : rv)
            r[q] = count++;
    }

    assert(std::find(r.begin(), r.end(), -1) == r.end());
    return r;
}
