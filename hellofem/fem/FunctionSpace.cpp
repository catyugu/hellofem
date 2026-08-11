// hellofem::fem — function space
// SPDX-License-Identifier: MIT

#include "FunctionSpace.h"

#include "basis/finite-element.h"

#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace hellofem::fem {

    template <std::floating_point T>
    std::vector<T> FunctionSpace<T>::tabulate_dof_coordinates(
        bool transpose) const
    {
        if (_elements.front()->is_mixed())
            throw std::runtime_error(
                "Cannot tabulate dof coordinates for a mixed element.");

        const mesh::Topology& topology = *_mesh->topology();
        const mesh::Geometry<T>& geometry = _mesh->geometry();
        const int gdim = geometry.dim();
        const int tdim = topology.dim();
        const std::size_t num_cells
            = geometry.dofmaps().front().extent(0);

        const FiniteElement<T>& element = *_elements.front();
        const DofMap& dofmap = *_dofmaps.front();
        const int bs = dofmap.bs();
        const int ndofs = element.space_dimension();
        const int index_map_bs = dofmap.index_map_bs();

        // One reference dof point per dof block.
        auto [points, pshape] = element.interpolation_points();
        const int num_points = static_cast<int>(pshape[0]);
        if (num_points * bs != ndofs)
            throw std::runtime_error(
                "tabulate_dof_coordinates: inconsistent dof point count.");

        // Physical coordinates of every scalar dof: (num_scalar_dofs, gdim)
        // with scalar dof `d` stored at row `d`.
        const std::int64_t num_scalar_dofs
            = index_map_bs * dofmap.index_map->size_local();
        std::vector<T> dofcoords(
            static_cast<std::size_t>(num_scalar_dofs) * gdim, 0.0);

        const auto x_dofmap = geometry.dofmaps().front();
        const CoordinateElement<T>& cmap = geometry.cmaps().front();
        const std::size_t ngeom_dofs = x_dofmap.extent(1);

        // Geometry basis at the dof points (values only), row-major
        // (num_points, ngeom_dofs).
        std::array<std::size_t, 2> xshape {
            static_cast<std::size_t>(num_points), static_cast<std::size_t>(tdim)};
        std::vector<T> phi(cmap.tabulate_shape(0, num_points)[2] * num_points);
        cmap.tabulate(0, points, xshape, phi);

        // Cell dof points in physical coordinates: (num_points, gdim).
        std::vector<T> Xc(static_cast<std::size_t>(num_points) * gdim);
        // Gathered cell geometry nodes: (ngeom_dofs, gdim).
        std::vector<T> cdofs(ngeom_dofs * gdim);
        std::span<const T> x = geometry.x();

        std::vector<std::int32_t> cells(num_cells);
        std::iota(cells.begin(), cells.end(), 0);

        for (std::int32_t c : cells) {
            // Gather the cell's geometry nodes.
            for (std::size_t j = 0; j < ngeom_dofs; ++j)
                for (int k = 0; k < gdim; ++k)
                    cdofs[j * gdim + k]
                        = x[static_cast<std::size_t>(3 * x_dofmap(c, j)) + k];

            // Push the dof points forward: Xc[p] = sum_j phi[p,j] cdofs[j].
            for (int p = 0; p < num_points; ++p)
                for (int k = 0; k < gdim; ++k) {
                    T acc = 0;
                    for (std::size_t j = 0; j < ngeom_dofs; ++j)
                        acc += phi[static_cast<std::size_t>(p * ngeom_dofs + j)]
                            * cdofs[j * gdim + k];
                    Xc[static_cast<std::size_t>(p * gdim + k)] = acc;
                }

            // Assign to the global dofs of this cell. Physical dof
            // index = bs * block + component.
            auto dofs = dofmap.cell_dofs(c);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (int k = 0; k < bs; ++k) {
                    const std::int64_t d
                        = static_cast<std::int64_t>(bs) * dofs[i] + k;
                    for (int q = 0; q < gdim; ++q)
                        dofcoords[static_cast<std::size_t>(d * gdim + q)]
                            = Xc[static_cast<std::size_t>(i * gdim + q)];
                }
        }

        if (transpose) {
            // Convert (num_dofs, gdim) to (gdim, num_dofs).
            std::vector<T> t(static_cast<std::size_t>(num_scalar_dofs) * gdim);
            for (std::int64_t d = 0; d < num_scalar_dofs; ++d)
                for (int q = 0; q < gdim; ++q)
                    t[static_cast<std::size_t>(q * num_scalar_dofs + d)]
                        = dofcoords[static_cast<std::size_t>(d * gdim + q)];
            return t;
        }

        return dofcoords;
    }

    template class FunctionSpace<double>;
    template class FunctionSpace<float>;

} // namespace hellofem::fem
