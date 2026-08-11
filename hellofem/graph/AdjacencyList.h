// hellofem::graph — static adjacency list (CSR-style)
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdint>
#include <format>
#include <iterator>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hellofem::graph {

    /// Static adjacency list: for each node in [0, n) stores its outgoing
    /// links in a contiguous array with per-node offsets. Strictly local
    /// (not parallel aware).
    ///
    /// `LinkData` is the edge type (may carry a weight or be a plain index);
    /// `NodeData` optionally attaches a value to each node.
    template <typename LinkData, typename NodeData = std::nullptr_t>
    class AdjacencyList {
    public:
        using link_type = LinkData;
        using node_data_type = NodeData;

        /// Trivial list where each of `n` nodes is connected to itself.
        explicit AdjacencyList(std::int32_t n) : _array(n), _offsets(n + 1)
        {
            std::iota(_array.begin(), _array.end(), 0);
            std::iota(_offsets.begin(), _offsets.end(), 0);
        }

        /// Build from a flat data array and per-node offsets, where the last
        /// offset equals the data length.
        template <typename U, typename V>
            requires std::is_convertible_v<std::remove_cvref_t<U>,
                                           std::vector<LinkData>>
                     and std::is_convertible_v<std::remove_cvref_t<V>,
                                               std::vector<std::int32_t>>
        AdjacencyList(U&& data, V&& offsets)
            : _array(std::forward<U>(data)), _offsets(std::forward<V>(offsets))
        {
            if (_offsets.back() != static_cast<std::int32_t>(_array.size()))
                throw std::runtime_error(
                    "Last offset must equal the size of the data array.");
        }

        /// Same as above, with per-node data attached.
        template <typename U, typename V, typename W>
            requires std::is_convertible_v<std::remove_cvref_t<U>,
                                           std::vector<LinkData>>
                     and std::is_convertible_v<std::remove_cvref_t<V>,
                                               std::vector<std::int32_t>>
                     and std::is_convertible_v<std::remove_cvref_t<W>,
                                               std::vector<NodeData>>
        AdjacencyList(U&& data, V&& offsets, W&& node_data)
            : _array(std::forward<U>(data)), _offsets(std::forward<V>(offsets)),
              _node_data(std::forward<W>(node_data))
        {
            if (!_node_data.has_value()
                or _node_data->size() != _offsets.size() - 1)
                throw std::runtime_error(
                    "Node data size must equal the number of nodes.");
            if (_offsets.back() != static_cast<std::int32_t>(_array.size()))
                throw std::runtime_error(
                    "Last offset must equal the size of the data array.");
        }

        /// Build from a container of containers (one range of links per node).
        template <typename X>
        explicit AdjacencyList(const std::vector<X>& data)
        {
            _offsets.reserve(data.size() + 1);
            _offsets.push_back(0);
            for (const auto& row : data)
                _offsets.push_back(_offsets.back() + row.size());

            _array.reserve(_offsets.back());
            for (const auto& e : data)
                _array.insert(_array.end(), e.begin(), e.end());
        }

        /// Copy constructor
        AdjacencyList(const AdjacencyList& list) = default;
        /// Move constructor
        AdjacencyList(AdjacencyList&& list) = default;
        /// Destructor
        ~AdjacencyList() = default;
        /// Copy assignment
        AdjacencyList& operator=(const AdjacencyList& list) = default;
        /// Move assignment
        AdjacencyList& operator=(AdjacencyList&& list) = default;

        /// Equality operator
        bool operator==(const AdjacencyList& list) const
        {
            return _array == list._array and _offsets == list._offsets;
        }

        /// Number of nodes.
        std::int32_t num_nodes() const { return _offsets.size() - 1; }

        /// Number of outgoing links for a node.
        int num_links(std::size_t node) const
        {
            assert((node + 1) < _offsets.size());
            return _offsets[node + 1] - _offsets[node];
        }

        /// Outgoing links for a node.
        std::span<LinkData> links(std::size_t node)
        {
            auto it = std::next(_offsets.begin(), node);
            return std::span<LinkData>(std::next(_array.begin(), *it),
                                       std::next(_array.begin(), *(it + 1)));
        }

        /// Outgoing links for a node (const).
        std::span<const LinkData> links(std::size_t node) const
        {
            auto it = std::next(_offsets.begin(), node);
            return std::span<const LinkData>(
                std::next(_array.begin(), *it),
                std::next(_array.begin(), *(it + 1)));
        }

        /// Contiguous link array for all nodes.
        const std::vector<LinkData>& array() const { return _array; }
        std::vector<LinkData>& array() { return _array; }

        /// Per-node offsets into array().
        const std::vector<std::int32_t>& offsets() const { return _offsets; }
        std::vector<std::int32_t>& offsets() { return _offsets; }

        /// Per-node data (if present).
        const std::optional<std::vector<NodeData>>& node_data() const
        {
            return _node_data;
        }
        std::optional<std::vector<NodeData>>& node_data() { return _node_data; }

        /// Informal string representation.
        std::string str() const
        {
            std::string s
                = std::format("<AdjacencyList> with {} nodes\n",
                              this->num_nodes());
            for (std::size_t e = 0; e < _offsets.size() - 1; ++e) {
                std::format_to(std::back_inserter(s), "  {}: [", e);
                for (auto link : this->links(e))
                    std::format_to(std::back_inserter(s), "{} ", link);
                s += "]\n";
            }
            return s;
        }

    private:
        // Links for all nodes, contiguous
        std::vector<LinkData> _array;

        // First link position of each node
        std::vector<std::int32_t> _offsets;

        // Optional per-node data
        std::optional<std::vector<NodeData>> _node_data = std::nullopt;
    };

    /// @private Deduction guides
    template <typename T, typename U>
    AdjacencyList(T, U) -> AdjacencyList<typename T::value_type, std::nullptr_t>;

    template <typename T, typename U, typename W>
    AdjacencyList(T, U, W)
        -> AdjacencyList<typename T::value_type, typename W::value_type>;

    /// Build a constant-degree adjacency list from a flat array where every
    /// node has exactly `degree` links.
    template <typename V = std::nullptr_t, typename U>
        requires requires {
            typename std::decay_t<U>::value_type;
            requires std::convertible_to<
                U, std::vector<typename std::decay_t<U>::value_type>>;
        }
    AdjacencyList<typename std::decay_t<U>::value_type, V>
    regular_adjacency_list(U&& data, int degree)
    {
        if (degree == 0 and !data.empty())
            throw std::runtime_error(
                "Degree is zero but data is not empty for constant degree "
                "AdjacencyList");

        if (degree > 0 and data.size() % degree != 0)
            throw std::runtime_error("Incompatible data size and degree for "
                                     "constant degree AdjacencyList");

        std::int32_t num_nodes = degree == 0 ? data.size() : data.size() / degree;
        std::vector<std::int32_t> offsets(num_nodes + 1, 0);
        for (std::size_t i = 1; i < offsets.size(); ++i)
            offsets[i] = offsets[i - 1] + degree;
        return AdjacencyList<typename std::decay_t<U>::value_type, V>(
            std::forward<U>(data), std::move(offsets));
    }

} // namespace hellofem::graph
