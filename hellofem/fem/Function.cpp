// hellofem::fem — finite element function
// SPDX-License-Identifier: MIT

#include "Function.h"

#include "basis/finite-element.h"
#include "geometry/BoundingBoxTree.h"
#include "geometry/utils.h"
#include "graph/AdjacencyList.h"

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

    template <std::floating_point T>
    void Function<T>::eval(std::span<const T> x,
        std::array<std::size_t, 2> xshape, std::span<const std::int32_t> cells,
        std::span<T> u, std::array<std::size_t, 2> ushape, double tol,
        int maxit) const
    {
        if (cells.empty())
            return;

        assert(x.size() == xshape[0] * xshape[1]);
        assert(u.size() == ushape[0] * ushape[1]);
        if (xshape[0] != cells.size())
            throw std::runtime_error("Function::eval: number of points and "
                                     "number of cells must be equal.");
        if (xshape[0] != ushape[0])
            throw std::runtime_error("Function::eval: number of points and "
                                     "number of value rows must be equal.");

        const mesh::Mesh<T>& mesh = *_V->mesh();
        const std::size_t gdim = mesh.geometry().dim();
        const std::size_t tdim = mesh.topology()->dim();

        const FiniteElement<T>& element = *_V->element();
        const int bs_element = element.block_size();
        const std::size_t value_size = element.reference_value_size();
        const std::size_t space_dimension
            = element.space_dimension() / bs_element;

        // Mixed elements are not supported: the component structure would
        // need per-sub-element bookkeeping.
        const int num_sub_elements = element.num_sub_elements();
        if (num_sub_elements > 1 and num_sub_elements != bs_element)
            throw std::runtime_error("Function::eval is not supported for "
                                     "mixed elements. Extract subspaces.");

        const CoordinateElement<T>& cmap = mesh.geometry().cmaps().front();
        const auto x_dofmap = mesh.geometry().dofmaps().front();
        const std::size_t num_dofs_g = cmap.dim();
        std::span<const T> x_g = mesh.geometry().x();

        const DofMap& dofmap = *_V->dofmap();
        const int bs_dof = dofmap.bs();
        std::span<const T> _v = _x->array();

        // Cell orientation data for dof transformations.
        std::span<const std::uint32_t> cell_info;
        if (element.needs_dof_transformations()) {
            mesh.topology_mutable()->create_entity_permutations();
            cell_info = std::span(mesh.topology()->get_cell_permutation_info());
        }

        // Per-point geometry data (reference coords, Jacobian, inverse).
        std::vector<T> Xb(xshape[0] * tdim);
        std::vector<T> J_b(xshape[0] * gdim * tdim);
        std::vector<T> K_b(xshape[0] * tdim * gdim);
        std::vector<T> detJ(xshape[0]);
        std::vector<T> det_scratch(2 * gdim * tdim);

        // Coordinate basis at the reference origin (affine fast path).
        std::vector<T> phi0_b(cmap.tabulate_shape(1, 1)[0]
            * cmap.tabulate_shape(1, 1)[1] * cmap.tabulate_shape(1, 1)[2]
            * cmap.tabulate_shape(1, 1)[3]);
        cmap.tabulate(1, std::vector<T>(tdim), {1, tdim}, phi0_b);
        // dphi0: (tdim, num_dofs_g), derivative k in block (k+1, 0, :, 0).
        std::vector<T> dphi0(tdim * num_dofs_g);
        for (std::size_t k = 0; k < tdim; ++k)
            for (std::size_t j = 0; j < num_dofs_g; ++j)
                dphi0[k * num_dofs_g + j]
                    = phi0_b[((k + 1) * 1 + 0) * num_dofs_g + j];

        // Coordinate basis at pulled-back points (non-affine path).
        std::vector<T> phi_b(cmap.tabulate_shape(1, 1)[0]
            * cmap.tabulate_shape(1, 1)[1] * cmap.tabulate_shape(1, 1)[2]
            * cmap.tabulate_shape(1, 1)[3]);
        std::vector<T> dphi(tdim * num_dofs_g);

        std::vector<T> pull_back_scratch(
            cmap.is_affine() ? 0 : cmap.pull_back_working_size(gdim));

        std::vector<T> coord_dofs_b(num_dofs_g * gdim);
        std::vector<T> xp_b(gdim);
        std::vector<T> Xpb(tdim);

        // Push-forward of reference basis to physical values.
        using xu_t = md::mdspan<T, md::dextents<std::size_t, 2>>;
        using xU_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;
        using xJ_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;
        using xK_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;
        auto push_forward_fn
            = element.basix_element().template map_fn<xu_t, xU_t, xJ_t, xK_t>();
        auto apply_dof_transformation
            = element.template dof_transformation_fn<T>(doftransform::standard);

        // Element tensor from a single point's reference basis.
        std::vector<T> basis_values(space_dimension * value_size);

        for (std::size_t p = 0; p < xshape[0]; ++p) {
            const std::int32_t c = cells[p];
            if (c < 0)
                continue;

            // Gather the cell geometry nodes.
            for (std::size_t i = 0; i < num_dofs_g; ++i)
                for (std::size_t j = 0; j < gdim; ++j)
                    coord_dofs_b[i * gdim + j]
                        = x_g[3 * x_dofmap(static_cast<std::size_t>(c), i) + j];

            for (std::size_t j = 0; j < gdim; ++j)
                xp_b[j] = x[p * xshape[1] + j];

            md::mdspan<T, md::dextents<std::size_t, 2>> Xp(Xpb.data(), 1, tdim);
            md::mdspan<T, md::dextents<std::size_t, 2>> xp(xp_b.data(), 1, gdim);
            md::mdspan<const T, md::dextents<std::size_t, 2>> coord_dofs(
                coord_dofs_b.data(), num_dofs_g, gdim);
            md::mdspan<T, md::dextents<std::size_t, 2>> J(
                J_b.data() + p * gdim * tdim, gdim, tdim);
            md::mdspan<T, md::dextents<std::size_t, 2>> K(
                K_b.data() + p * tdim * gdim, tdim, gdim);

            if (cmap.is_affine()) {
                // Affine map: pull back with X = K(x - x0), where x0 is
                // the first geometry node and K = inv(J) evaluated at the
                // reference origin.
                md::mdspan<const T, md::dextents<std::size_t, 2>> dphi0_view(
                    dphi0.data(), tdim, num_dofs_g);
                CoordinateElement<T>::compute_jacobian(dphi0_view, coord_dofs, J);
                CoordinateElement<T>::compute_jacobian_inverse(J, K);
                std::array<T, 3> x0 {0, 0, 0};
                for (std::size_t i = 0; i < gdim; ++i)
                    x0[i] = coord_dofs(0, i);
                CoordinateElement<T>::pull_back_affine(Xp, K, x0, xp);
                detJ[p] = static_cast<T>(
                    CoordinateElement<T>::compute_jacobian_determinant(
                        J, std::span(det_scratch.data(), det_scratch.size())));
            }
            else {
                // Non-affine map: Newton pull-back then tabulate at the
                // pulled-back reference point.
                cmap.pull_back_nonaffine(
                    Xp, xp, coord_dofs, std::span(pull_back_scratch), tol, maxit);
                cmap.tabulate(1, std::span(Xpb.data(), tdim), {1, tdim}, phi_b);
                for (std::size_t k = 0; k < tdim; ++k)
                    for (std::size_t j = 0; j < num_dofs_g; ++j)
                        dphi[k * num_dofs_g + j]
                            = phi_b[((k + 1) * 1 + 0) * num_dofs_g + j];
                md::mdspan<const T, md::dextents<std::size_t, 2>> dphi_view(
                    dphi.data(), tdim, num_dofs_g);
                CoordinateElement<T>::compute_jacobian(dphi_view, coord_dofs, J);
                CoordinateElement<T>::compute_jacobian_inverse(J, K);
                detJ[p] = static_cast<T>(
                    CoordinateElement<T>::compute_jacobian_determinant(
                        J, std::span(det_scratch.data(), det_scratch.size())));
            }

            for (std::size_t j = 0; j < tdim; ++j)
                Xb[p * tdim + j] = Xpb[j];
        }

        // Tabulate the reference basis at all pulled-back points and map
        // to physical values, then expand the coefficients.
        auto [basis_all, bshape] = element.tabulate(
            std::span<const T>(Xb), {xshape[0], tdim}, 0);
        // bshape: (1, num_points, space_dim, reference_value_size).
        const std::size_t num_basis_values = space_dimension * value_size;

        std::vector<T> coefficients(space_dimension * bs_element);

        std::ranges::fill(u, 0);
        for (std::size_t p = 0; p < xshape[0]; ++p) {
            const std::int32_t c = cells[p];
            if (c < 0)
                continue;

            // Reference basis of this point (values, no derivatives).
            std::vector<T> U_point(num_basis_values);
            for (std::size_t i = 0; i < num_basis_values; ++i)
                U_point[i] = basis_all[p * num_basis_values + i];

            // Apply the cell's dof transformation to the basis values.
            apply_dof_transformation(std::span(U_point), cell_info, c,
                static_cast<int>(value_size));

            md::mdspan<T, md::dextents<std::size_t, 2>> basis_values_view(
                basis_values.data(), space_dimension, value_size);
            md::mdspan<const T, md::dextents<std::size_t, 2>> U_view(
                U_point.data(), space_dimension, value_size);
            md::mdspan<const T, md::dextents<std::size_t, 2>> J_view(
                J_b.data() + p * gdim * tdim, gdim, tdim);
            md::mdspan<const T, md::dextents<std::size_t, 2>> K_view(
                K_b.data() + p * tdim * gdim, tdim, gdim);
            push_forward_fn(basis_values_view, U_view, J_view, detJ[p], K_view);

            // Gather the cell coefficients.
            auto dofs = dofmap.cell_dofs(c);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (int k = 0; k < bs_dof; ++k)
                    coefficients[bs_dof * i + k]
                        = _v[static_cast<std::size_t>(bs_dof * dofs[i]) + k];

            // Expand: u[p, j] += sum_i coefficients[bs*i + k] * basis(i, j).
            for (int k = 0; k < bs_element; ++k) {
                for (std::size_t i = 0; i < space_dimension; ++i) {
                    for (std::size_t j = 0; j < value_size; ++j) {
                        u[p * ushape[1] + j * bs_element + k]
                            += coefficients[bs_element * i + k]
                            * basis_values[i * value_size + j];
                    }
                }
            }
        }
    }

    template <std::floating_point T>
    std::pair<std::vector<T>, std::array<std::size_t, 2>>
    Function<T>::eval(std::span<const T> x,
        std::array<std::size_t, 2> xshape) const
    {
        const std::size_t num_points = xshape[0];
        const std::size_t gdim = xshape[1];

        // The geometry collision queries work on 3D points; pad the input
        // to (num_points, 3) with zero trailing coordinates.
        std::vector<T> pts(3 * num_points, 0);
        for (std::size_t p = 0; p < num_points; ++p)
            for (std::size_t j = 0; j < gdim; ++j)
                pts[3 * p + j] = x[p * gdim + j];

        // Locate the containing cell of each point with a bounding-box
        // collision query refined by GJK.
        geometry::BoundingBoxTree<T> tree(
            *_V->mesh(), _V->mesh()->topology()->dim(), 0.0);
        auto collisions = geometry::compute_collisions(tree, pts);
        auto cells = geometry::compute_colliding_cells(*_V->mesh(), collisions, pts);

        std::vector<std::int32_t> cell(num_points);
        for (std::size_t p = 0; p < num_points; ++p) {
            auto links = cells.links(static_cast<std::int32_t>(p));
            if (links.empty())
                cell[p] = -1;
            else
                cell[p] = links.front();
        }

        const std::size_t value_size
            = static_cast<std::size_t>(_V->element()->value_size());
        std::vector<T> u(num_points * value_size, 0);
        eval(x, xshape, cell, u, {num_points, value_size});
        return {std::move(u), {num_points, value_size}};
    }

    template class Function<double>;
    template class Function<float>;

} // namespace hellofem::fem
