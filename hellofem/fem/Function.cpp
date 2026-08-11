// hellofem::fem — finite element function
// SPDX-License-Identifier: MIT

#include "Function.h"

#include "basis/finite-element.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace hellofem::fem {

    template <std::floating_point T>
    void Function<T>::interpolate(const eval_fn& f)
    {
        const FiniteElement<T>& element = *_V->element();
        const DofMap& dofmap = *_V->dofmap();
        const int bs = dofmap.bs();
        const int ndofs = element.space_dimension();

        const mesh::Topology& topology = *_V->mesh()->topology();
        const mesh::Geometry<T>& geometry = _V->mesh()->geometry();
        const int gdim = geometry.dim();
        const int tdim = topology.dim();

        // Reference interpolation points (one per dof block) and the
        // per-cell interpolation operator Pi (ndofs, num_points).
        auto [points, pshape] = element.interpolation_points();
        const int num_points = static_cast<int>(pshape[0]);
        auto [Pi, sshape] = element.interpolation_operator();
        const int pi_points = static_cast<int>(sshape[1]);
        if (pi_points != num_points)
            throw std::runtime_error("interpolate: inconsistent operator.");

        const auto x_dofmap = geometry.dofmaps().front();
        const fem::CoordinateElement<T>& cmap = geometry.cmaps().front();
        const std::size_t ngeom_dofs = x_dofmap.extent(1);

        // Geometry basis at the interpolation points (values only).
        std::array<std::size_t, 2> xshape {
            static_cast<std::size_t>(num_points), static_cast<std::size_t>(tdim)};
        std::vector<T> phi(cmap.tabulate_shape(0, num_points)[2] * num_points);
        cmap.tabulate(0, points, xshape, phi);

        std::vector<T> Xc(static_cast<std::size_t>(num_points) * gdim);
        std::vector<T> cdofs(ngeom_dofs * gdim);
        std::vector<T> fx(static_cast<std::size_t>(num_points)
            * element.value_size());
        std::vector<T> coeffs(static_cast<std::size_t>(ndofs), 0.0);
        std::span<const T> x = geometry.x();

        const std::size_t num_cells = dofmap.map().extent(0);
        std::vector<T>& xa = _x->array();
        for (std::size_t c = 0; c < num_cells; ++c) {
            // Push the reference interpolation points forward.
            for (std::size_t j = 0; j < ngeom_dofs; ++j)
                for (int k = 0; k < gdim; ++k)
                    cdofs[j * gdim + k]
                        = x[static_cast<std::size_t>(3 * x_dofmap(c, j)) + k];
            for (int p = 0; p < num_points; ++p)
                for (int k = 0; k < gdim; ++k) {
                    T acc = 0;
                    for (std::size_t j = 0; j < ngeom_dofs; ++j)
                        acc += phi[static_cast<std::size_t>(p * ngeom_dofs + j)]
                            * cdofs[j * gdim + k];
                    Xc[static_cast<std::size_t>(p * gdim + k)] = acc;
                }

            // Evaluate the field at the physical points.
            auto [values, vshape] = f(Xc, {static_cast<std::size_t>(num_points), static_cast<std::size_t>(gdim)});
            if (vshape[0] != static_cast<std::size_t>(num_points)
                or vshape[1] != static_cast<std::size_t>(element.value_size()))
                throw std::runtime_error("interpolate: bad field values.");
            fx.assign(values.begin(), values.end());

            // Coefficients = Pi * f_x.
            for (int d = 0; d < ndofs; ++d) {
                T acc = 0;
                for (int p = 0; p < num_points; ++p)
                    acc += Pi[static_cast<std::size_t>(d) * num_points + p]
                        * fx[static_cast<std::size_t>(p)];
                coeffs[static_cast<std::size_t>(d)] = acc;
            }

            // Distribute block coefficients to the physical dofs.
            auto dofs = dofmap.cell_dofs(static_cast<std::int32_t>(c));
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (int k = 0; k < bs; ++k)
                    xa[static_cast<std::size_t>(bs * dofs[i] + k)]
                        = coeffs[static_cast<std::size_t>(bs * i + k)];
        }
    }

    template <std::floating_point T>
    void Function<T>::interpolate(const Function<T>& u)
    {
        if (u._V->dofmap() != _V->dofmap())
            throw std::runtime_error(
                "interpolate: functions are on different spaces.");
        std::copy(u._x->array().begin(), u._x->array().end(),
            _x->array().begin());
    }

    template class Function<double>;
    template class Function<float>;

} // namespace hellofem::fem
