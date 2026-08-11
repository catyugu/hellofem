// hellofem::fem — function space
// SPDX-License-Identifier: MIT

#pragma once

#include "DofMap.h"
#include "FiniteElement.h"
#include "mesh/Mesh.h"

#include <concepts>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace hellofem::fem {

    /// A function space is a finite element space defined on a mesh: a
    /// finite element (with dof layout) plus a dofmap assigning cell
    /// dofs to global dofs.
    template <std::floating_point T>
    class FunctionSpace {
    public:
        /// Geometry type of the mesh.
        using geometry_type = T;

        /// Create a function space on a mesh for a single element.
        /// @param[in] mesh The mesh.
        /// @param[in] element The finite element.
        /// @param[in] dofmap The dofmap.
        FunctionSpace(std::shared_ptr<const mesh::Mesh<T>> mesh,
            std::shared_ptr<const FiniteElement<T>> element,
            std::shared_ptr<const DofMap> dofmap)
            : _mesh(std::move(mesh)), _elements({std::move(element)}), _dofmaps({std::move(dofmap)})
        {
        }

        /// The mesh the space is defined on.
        std::shared_ptr<const mesh::Mesh<T>> mesh() const { return _mesh; }

        /// The finite element.
        std::shared_ptr<const FiniteElement<T>> element() const
        {
            return _elements.front();
        }

        /// The dofmap.
        std::shared_ptr<const DofMap> dofmap() const { return _dofmaps.front(); }

        /// Coordinates of every dof in the space (one row of `gdim`
        /// entries per dof), obtained by evaluating the geometry map at
        /// the element's dof points.
        ///
        /// @param[in] transpose Layout: when true the result is
        /// `(gdim, num_dofs)`, otherwise `(num_dofs, gdim)`.
        std::vector<geometry_type> tabulate_dof_coordinates(bool transpose) const;

    private:
        // Mesh the space is defined on.
        std::shared_ptr<const mesh::Mesh<T>> _mesh;

        // Elements, one per cell type (mixed topology).
        std::vector<std::shared_ptr<const FiniteElement<T>>> _elements;

        // Dofmaps, one per cell type.
        std::vector<std::shared_ptr<const DofMap>> _dofmaps;
    };

} // namespace hellofem::fem
