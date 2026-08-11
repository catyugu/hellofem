// hellofem::mesh — mesh: topology + geometry
// SPDX-License-Identifier: MIT

#pragma once

#include "Geometry.h"

#include <concepts>
#include <memory>
#include <string>
#include <utility>

namespace hellofem::mesh {

    class Topology;

    /// A Mesh consists of a connected, numbered set of topological
    /// entities and the geometry imposed on them.
    template <std::floating_point T>
    class Mesh {
    public:
        /// Value type.
        using geometry_type = Geometry<T>;

        /// Create a mesh from a topology and a geometry.
        ///
        /// @note Not normally called by user code; use
        /// mesh::create_mesh instead.
        template <typename V>
            requires std::is_convertible_v<std::remove_cvref_t<V>, Geometry<T>>
        Mesh(std::shared_ptr<Topology> topology, V&& geometry)
            : _topology(std::move(topology)), _geometry(std::forward<V>(geometry))
        {
            // Do nothing
        }

        /// Copy constructor
        Mesh(const Mesh& mesh) = default;
        /// Move constructor
        Mesh(Mesh&& mesh) = default;
        /// Destructor
        ~Mesh() = default;

        // Copy assignment (deleted)
        Mesh& operator=(const Mesh& mesh) = delete;
        /// Move assignment
        Mesh& operator=(Mesh&& mesh) = default;

        /// Get the mesh topology.
        std::shared_ptr<Topology> topology() { return _topology; }

        /// Get the mesh topology (const version).
        std::shared_ptr<const Topology> topology() const { return _topology; }

        /// Get the mesh topology (mutable version; the topology caches
        /// lazily computed connectivity, so const meshes can still need
        /// mutation).
        std::shared_ptr<Topology> topology_mutable() const { return _topology; }

        /// Get the mesh geometry.
        Geometry<T>& geometry() { return _geometry; }

        /// Get the mesh geometry (const version).
        const Geometry<T>& geometry() const { return _geometry; }

        /// Name.
        std::string name = "mesh";

    private:
        // Mesh topology. Non-const because topology caches lazily
        // computed data.
        std::shared_ptr<Topology> _topology;

        // Mesh geometry.
        Geometry<T> _geometry;
    };

    /// Template type deduction.
    template <typename V>
    Mesh(std::shared_ptr<Topology>, V)
        -> Mesh<typename std::remove_cvref_t<typename V::value_type>>;

} // namespace hellofem::mesh
