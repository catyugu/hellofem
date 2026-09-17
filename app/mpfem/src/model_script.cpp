// hellofem::app — model_script accessors
// SPDX-License-Identifier: MIT

#include "model_script.h"

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

    const Material* ModelScript::material_on_domain(int domain) const
    {
        for (const auto& m : materials)
            if (m.domains.contains(domain))
                return &m;
        return nullptr;
    }

} // namespace hellofem::app
