// hellofem::app — model_script accessors
// SPDX-License-Identifier: MIT

#include "model_script.h"

#include <stdexcept>

namespace hellofem::app {

    std::string MaterialProperty::scalar_value() const
    {
        const std::size_t space = value.find(' ');
        return space == std::string::npos ? value : value.substr(0, space);
    }

    const MaterialProperty* Material::property(std::string_view name) const
    {
        for (const auto& p : properties)
            if (p.name == name)
                return &p;
        return nullptr;
    }

    const std::string& PhysicsFeature::required(std::string_view key) const
    {
        const auto it = properties.find(std::string(key));
        if (it == properties.end())
            throw std::runtime_error("feature '" + tag + "' of type '" + type
                + "' states no '" + std::string(key) + "'");
        return it->second;
    }

    const Material* ModelScript::material_on_domain(int domain) const
    {
        // The last material COMSOL lists wins: a domain two materials select
        // takes the later one's properties, which is how a model refines one
        // region of a selection without splitting it.
        const Material* found = nullptr;
        for (const auto& m : materials)
            if (m.domains.contains(domain))
                found = &m;
        return found;
    }

    const Material* ModelScript::material_on_boundary(int boundary) const
    {
        const Material* found = nullptr;
        for (const auto& m : materials)
            if (m.boundaries.contains(boundary))
                found = &m;
        return found;
    }

} // namespace hellofem::app
