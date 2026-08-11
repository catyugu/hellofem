// hellofem::fem — model of a finite element
// SPDX-License-Identifier: MIT

#include "FiniteElement.h"

#include "basis/interpolation.h"

#include <format>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace hellofem::fem {

    namespace {

        /// Compute the block size from a value shape: product of the shape
        /// entries, or `n(n+1)/2` for a symmetric square rank-2 tensor.
        int _compute_block_size(
            const std::optional<std::vector<std::size_t>>& value_shape,
            bool symmetric)
        {
            if (symmetric and value_shape) {
                if (value_shape->size() != 2
                    or (value_shape->front() != value_shape->back())) {
                    throw std::runtime_error(
                        "Symmetric elements require square rank-2 value shape.");
                }

                return value_shape->front()
                    * (value_shape->front() + 1) / 2;
            }
            else if (value_shape) {
                return std::accumulate(value_shape->begin(),
                    value_shape->end(), 1, std::multiplies {});
            }
            else
                return 1;
        }

        /// Recursively build an ElementDofLayout from a finite element,
        /// constructing sub-layouts for each sub-element.
        template <std::floating_point T>
        ElementDofLayout _create_element_dof_layout(
            const FiniteElement<T>& element, const std::vector<int>& parent_map)
        {
            std::vector<int> offsets {0};
            std::vector<ElementDofLayout> sub_doflayout;
            const int bs = element.block_size();
            for (int i = 0; i < element.num_sub_elements(); ++i) {
                // For blocked elements all sub-elements are identical; mixed
                // elements carry one distinct sub-element per entry.
                std::shared_ptr<const FiniteElement<T>> sub_e
                    = element.sub_elements()[bs > 1 ? 0 : i];

                // Mixed elements order dofs element-by-element (offset =
                // sub-space dimension); blocked elements use xyzxyz ordering
                // (offset 1 per sub-element).
                std::vector<int> parent_map_sub(
                    sub_e->space_dimension(), offsets.back());
                for (std::size_t j = 0; j < parent_map_sub.size(); ++j)
                    parent_map_sub[j] += bs * j;
                offsets.push_back(offsets.back()
                    + (bs > 1 ? 1 : sub_e->space_dimension()));

                sub_doflayout.push_back(
                    _create_element_dof_layout(*sub_e, parent_map_sub));
            }

            return ElementDofLayout(bs, element.entity_dofs(),
                element.entity_closure_dofs(), parent_map, sub_doflayout);
        }

    } // namespace

    template <std::floating_point T>
    FiniteElement<T>::FiniteElement(
        const basis::FiniteElement<T>& element,
        const std::optional<std::vector<std::size_t>>& value_shape,
        bool symmetric)
        : _value_shape(value_shape.value_or(element.value_shape())),
          _bs(_compute_block_size(value_shape, symmetric)),
          _cell_type(mesh::cell_type_from_basix_type(element.cell_type())),
          _space_dim(_bs * element.dim()),
          _reference_value_shape(element.value_shape()),
          _element(std::make_unique<basis::FiniteElement<T>>(element)),
          _symmetric(symmetric),
          _needs_dof_permutations(
              !element.dof_transformations_are_identity()
              and element.dof_transformations_are_permutations()),
          _needs_dof_transformations(
              !element.dof_transformations_are_identity()
              and !element.dof_transformations_are_permutations()),
          _entity_dofs(element.entity_dofs()),
          _entity_closure_dofs(element.entity_closure_dofs())
    {
        if (value_shape and !element.value_shape().empty()) {
            throw std::runtime_error("Blocked finite elements can be "
                                     "constructed only from scalar base "
                                     "elements.");
        }

        if (value_shape) {
            _sub_elements
                = std::vector<std::shared_ptr<const FiniteElement<T>>>(
                    _bs, std::make_shared<FiniteElement<T>>(element));
        }

        std::string family;
        switch (_element->family()) {
        case basis::element::family::P:
            family = "Lagrange";
            break;
        case basis::element::family::DPC:
            family = "Discontinuous Lagrange";
            break;
        default:
            family = "unknown";
            break;
        }

        _signature = std::format("Basix element {} {}", family, _bs);
    }

    template <std::floating_point T>
    FiniteElement<T>::FiniteElement(
        const std::vector<std::shared_ptr<const FiniteElement<T>>>& elements)
        : _value_shape(std::nullopt), _bs(1),
          _cell_type(elements.front()->cell_type()), _space_dim(-1),
          _sub_elements(elements), _reference_value_shape(std::nullopt),
          _symmetric(false), _needs_dof_permutations(false),
          _needs_dof_transformations(false)
    {
        _signature = "Mixed element (";

        const std::vector<std::vector<std::vector<int>>>& ed
            = elements.front()->entity_dofs();
        _entity_dofs.resize(ed.size());
        _entity_closure_dofs.resize(ed.size());
        for (std::size_t i = 0; i < ed.size(); ++i) {
            _entity_dofs[i].resize(ed[i].size());
            _entity_closure_dofs[i].resize(ed[i].size());
        }

        int dof_offset = 0;
        for (auto& e : elements) {
            _signature += e->signature() + ", ";

            if (e->needs_dof_permutations())
                _needs_dof_permutations = true;
            if (e->needs_dof_transformations())
                _needs_dof_transformations = true;

            const std::size_t sub_bs = e->block_size();
            for (std::size_t i = 0; i < _entity_dofs.size(); ++i) {
                for (std::size_t j = 0; j < _entity_dofs[i].size(); ++j) {
                    for (auto k : e->entity_dofs()[i][j])
                        for (std::size_t b = 0; b < sub_bs; ++b)
                            _entity_dofs[i][j].push_back(
                                dof_offset + k * sub_bs + b);
                    for (auto k : e->entity_closure_dofs()[i][j])
                        for (std::size_t b = 0; b < sub_bs; ++b)
                            _entity_closure_dofs[i][j].push_back(
                                dof_offset + k * sub_bs + b);
                }
            }

            dof_offset += e->space_dimension();
        }

        _space_dim = dof_offset;
        _signature += ")";
    }

    template <std::floating_point T>
    FiniteElement<T>::FiniteElement(mesh::CellType cell_type,
        std::span<const geometry_type> points,
        std::array<std::size_t, 2> pshape,
        std::vector<std::size_t> value_shape, bool symmetric)
        : _value_shape(value_shape),
          _bs(_compute_block_size(value_shape, symmetric)),
          _cell_type(cell_type),
          _signature(std::format("Quadrature element {} {}", pshape[0], _bs)),
          _space_dim(pshape[0] * _bs), _sub_elements({}),
          _reference_value_shape(std::vector<std::size_t>()), _element(nullptr),
          _symmetric(symmetric), _needs_dof_permutations(false),
          _needs_dof_transformations(false),
          _entity_dofs(mesh::cell_dim(cell_type) + 1),
          _entity_closure_dofs(mesh::cell_dim(cell_type) + 1),
          _points(std::vector<T>(points.begin(), points.end()), pshape)
    {
        const int tdim = mesh::cell_dim(cell_type);
        for (int d = 0; d <= tdim; ++d) {
            const int num_entities = mesh::cell_num_entities(cell_type, d);
            _entity_dofs[d].resize(num_entities);
            _entity_closure_dofs[d].resize(num_entities);
        }

        for (std::size_t i = 0; i < pshape[0]; ++i) {
            _entity_dofs[tdim][0].push_back(i);
            _entity_closure_dofs[tdim][0].push_back(i);
        }
    }

    template <std::floating_point T>
    bool FiniteElement<T>::operator==(const FiniteElement& e) const
    {
        if (!_element or !e._element) {
            throw std::runtime_error(
                "Missing a Basix element. Cannot check for equivalence");
        }

        return *_element == *e._element;
    }

    template <std::floating_point T>
    bool FiniteElement<T>::operator!=(const FiniteElement& e) const
    {
        return !(*this == e);
    }

    template <std::floating_point T>
    mesh::CellType FiniteElement<T>::cell_type() const noexcept
    {
        return _cell_type;
    }

    template <std::floating_point T>
    std::string FiniteElement<T>::signature() const noexcept
    {
        return _signature;
    }

    template <std::floating_point T>
    int FiniteElement<T>::space_dimension() const noexcept
    {
        return _space_dim;
    }

    template <std::floating_point T>
    int FiniteElement<T>::value_size() const
    {
        if (_value_shape) {
            return std::accumulate(_value_shape->begin(), _value_shape->end(),
                1, std::multiplies {});
        }
        else
            throw std::runtime_error("Element does not have a value_shape.");
    }

    template <std::floating_point T>
    std::span<const std::size_t> FiniteElement<T>::value_shape() const
    {
        if (_value_shape)
            return *_value_shape;
        else
            throw std::runtime_error("Element does not have a value_shape.");
    }

    template <std::floating_point T>
    int FiniteElement<T>::reference_value_size() const
    {
        if (_reference_value_shape) {
            return std::accumulate(_reference_value_shape->begin(),
                _reference_value_shape->end(), 1, std::multiplies {});
        }
        else
            throw std::runtime_error(
                "Element does not have a reference_value_shape.");
    }

    template <std::floating_point T>
    std::span<const std::size_t>
    FiniteElement<T>::reference_value_shape() const
    {
        if (_reference_value_shape)
            return *_reference_value_shape;
        else
            throw std::runtime_error(
                "Element does not have a reference_value_shape.");
    }

    template <std::floating_point T>
    const std::vector<std::vector<std::vector<int>>>&
    FiniteElement<T>::entity_dofs() const noexcept
    {
        return _entity_dofs;
    }

    template <std::floating_point T>
    const std::vector<std::vector<std::vector<int>>>&
    FiniteElement<T>::entity_closure_dofs() const noexcept
    {
        return _entity_closure_dofs;
    }

    template <std::floating_point T>
    bool FiniteElement<T>::symmetric() const
    {
        return _symmetric;
    }

    template <std::floating_point T>
    int FiniteElement<T>::block_size() const noexcept
    {
        return _bs;
    }

    template <std::floating_point T>
    void FiniteElement<T>::tabulate(std::span<geometry_type> values,
        std::span<const geometry_type> X, std::array<std::size_t, 2> shape,
        int order) const
    {
        assert(_element);
        _element->tabulate(order, X, shape, values);
    }

    template <std::floating_point T>
    std::pair<std::vector<T>, std::array<std::size_t, 4>>
    FiniteElement<T>::tabulate(std::span<const geometry_type> X,
        std::array<std::size_t, 2> shape, int order) const
    {
        assert(_element);
        return _element->tabulate(order, X, shape);
    }

    template <std::floating_point T>
    int FiniteElement<T>::num_sub_elements() const noexcept
    {
        return _sub_elements.size();
    }

    template <std::floating_point T>
    bool FiniteElement<T>::is_mixed() const noexcept
    {
        return !_reference_value_shape;
    }

    template <std::floating_point T>
    const std::vector<std::shared_ptr<const FiniteElement<T>>>&
    FiniteElement<T>::sub_elements() const noexcept
    {
        return _sub_elements;
    }

    template <std::floating_point T>
    std::shared_ptr<const FiniteElement<T>>
    FiniteElement<T>::extract_sub_element(
        const std::vector<int>& component) const
    {
        if (component.empty())
            throw std::runtime_error("Cannot extract subsystem of finite "
                                     "element. No system was specified");
        if (num_sub_elements() == 0)
            throw std::runtime_error("Cannot extract subsystem of finite "
                                     "element. There are no subsystems.");
        if (component[0] >= num_sub_elements())
            throw std::runtime_error("Cannot extract subsystem of finite "
                                     "element. Requested subsystem out of "
                                     "range.");

        auto sub_element = _sub_elements[component[0]];
        if (component.size() == 1)
            return sub_element;

        std::vector<int> sub_component(component.begin() + 1,
            component.end());
        return sub_element->extract_sub_element(sub_component);
    }

    template <std::floating_point T>
    const basis::FiniteElement<T>& FiniteElement<T>::basix_element() const
    {
        if (_element)
            return *_element;
        else
            throw std::runtime_error(
                "No Basix element available. Maybe this is a mixed element?");
    }

    template <std::floating_point T>
    basis::maps::type FiniteElement<T>::map_type() const
    {
        if (_element)
            return _element->map_type();
        else
            throw std::runtime_error(
                "Cannot element map type - no Basix element available. Maybe "
                "this is a mixed element?");
    }

    template <std::floating_point T>
    bool FiniteElement<T>::map_ident() const noexcept
    {
        if (!_element and _points.second.front() > 0)
            return true; // Quadrature elements use the identity map
        assert(_element);
        return _element->map_type() == basis::maps::type::identity;
    }

    template <std::floating_point T>
    bool FiniteElement<T>::interpolation_ident() const noexcept
    {
        if (!_element and _points.second[0] > 0)
            return true;
        else {
            assert(_element);
            return _element->interpolation_is_identity();
        }
    }

    template <std::floating_point T>
    std::pair<std::vector<T>, std::array<std::size_t, 2>>
    FiniteElement<T>::interpolation_points() const
    {
        if (_points.second[0] > 0)
            return _points;
        else {
            if (!_element) {
                throw std::runtime_error("Cannot get interpolation points - no "
                                         "Basix element available. Maybe this "
                                         "is a mixed element?");
            }

            return _element->points();
        }
    }

    template <std::floating_point T>
    std::pair<std::vector<T>, std::array<std::size_t, 2>>
    FiniteElement<T>::interpolation_operator() const
    {
        if (!_element) {
            throw std::runtime_error(
                "No underlying element for interpolation. Cannot interpolate "
                "mixed elements directly.");
        }

        return _element->interpolation_matrix();
    }

    template <std::floating_point T>
    std::pair<std::vector<T>, std::array<std::size_t, 2>>
    FiniteElement<T>::create_interpolation_operator(
        const FiniteElement& from) const
    {
        assert(_element);
        assert(from._element);
        if (_element->map_type() != from._element->map_type()) {
            throw std::runtime_error("Interpolation between elements with "
                                     "different maps is not supported.");
        }

        if (_bs == 1 or from._bs == 1) {
            // Basix can figure out the matrix size when one element is
            // un-blocked.
            return basis::compute_interpolation_operator<T>(
                *from._element, *_element);
        }
        else if (_bs > 1 and from._bs == _bs) {
            // Both elements blocked with the same block size: build the
            // operator on the base elements and block-diagonally expand.
            const auto [data, dshape]
                = basis::compute_interpolation_operator<T>(
                    *from._element, *_element);
            std::array<std::size_t, 2> shape
                = {dshape[0] * _bs, dshape[1] * _bs};
            std::vector<T> out(shape[0] * shape[1]);
            for (std::size_t i = 0; i < dshape[0]; ++i)
                for (std::size_t j = 0; j < dshape[1]; ++j)
                    for (int k = 0; k < _bs; ++k)
                        out[shape[1] * (i * _bs + k) + (j * _bs + k)]
                            = data[dshape[1] * i + j];

            return {std::move(out), shape};
        }
        else {
            throw std::runtime_error(
                "Interpolation for element combination is not supported.");
        }
    }

    template <std::floating_point T>
    bool FiniteElement<T>::needs_dof_transformations() const noexcept
    {
        return _needs_dof_transformations;
    }

    template <std::floating_point T>
    bool FiniteElement<T>::needs_dof_permutations() const noexcept
    {
        return _needs_dof_permutations;
    }

    template <std::floating_point T>
    void FiniteElement<T>::permute(std::span<std::int32_t> doflist,
        std::uint32_t cell_permutation) const
    {
        assert(_element);
        _element->permute(doflist, cell_permutation);
    }

    template <std::floating_point T>
    void FiniteElement<T>::permute_inv(std::span<std::int32_t> doflist,
        std::uint32_t cell_permutation) const
    {
        assert(_element);
        _element->permute_inv(doflist, cell_permutation);
    }

    template <std::floating_point T>
    std::function<void(std::span<std::int32_t>, std::uint32_t)>
    FiniteElement<T>::dof_permutation_fn(bool inverse,
        bool scalar_element) const
    {
        if (!needs_dof_permutations())
            return [](std::span<std::int32_t>, std::uint32_t) {};

        if (!_sub_elements.empty()) {
            if (_bs == 1) {
                // Mixed element
                std::vector<std::function<void(std::span<std::int32_t>,
                    std::uint32_t)>>
                    sub_element_functions;
                std::vector<int> dims;
                for (std::size_t i = 0; i < _sub_elements.size(); ++i) {
                    sub_element_functions.push_back(
                        _sub_elements[i]->dof_permutation_fn(inverse));
                    dims.push_back(_sub_elements[i]->space_dimension());
                }

                return
                    [dims = std::move(dims),
                        sub_element_functions
                        = std::move(sub_element_functions)](
                        std::span<std::int32_t> doflist,
                        std::uint32_t cell_permutation) {
                        std::size_t start = 0;
                        for (std::size_t e = 0; e < sub_element_functions.size();
                             ++e) {
                            sub_element_functions[e](
                                doflist.subspan(start, dims[e]),
                                cell_permutation);
                            start += dims[e];
                        }
                    };
            }
            else if (!scalar_element) {
                // Blocked element
                std::function<void(std::span<std::int32_t>, std::uint32_t)>
                    sub_element_function
                    = _sub_elements.front()->dof_permutation_fn(inverse);
                const int dim = _sub_elements.front()->space_dimension();
                const int bs = _bs;
                return [sub_element_function = std::move(sub_element_function),
                           bs, subdofs = std::vector<std::int32_t>(dim)](
                           std::span<std::int32_t> doflist,
                           std::uint32_t cell_permutation) mutable {
                    for (int k = 0; k < bs; ++k) {
                        for (std::size_t i = 0; i < subdofs.size(); ++i)
                            subdofs[i] = doflist[bs * i + k];
                        sub_element_function(subdofs, cell_permutation);
                        for (std::size_t i = 0; i < subdofs.size(); ++i)
                            doflist[bs * i + k] = subdofs[i];
                    }
                };
            }
        }

        if (inverse) {
            return [this](std::span<std::int32_t> doflist,
                       std::uint32_t cell_permutation) {
                permute_inv(doflist, cell_permutation);
            };
        }
        else {
            return [this](std::span<std::int32_t> doflist,
                       std::uint32_t cell_permutation) {
                permute(doflist, cell_permutation);
            };
        }
    }

    template <std::floating_point T>
    ElementDofLayout FiniteElement<T>::create_dof_layout() const
    {
        return _create_element_dof_layout(*this, {});
    }

    template class FiniteElement<float>;
    template class FiniteElement<double>;

} // namespace hellofem::fem
