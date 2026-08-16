// hellofem::app — model_script accessors
// SPDX-License-Identifier: MIT

#include "model_script.h"

namespace hellofem::app {

const Parameter* ModelScript::parameter(const std::string& name) const
{
    for (const auto& p : parameters)
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

const Physics* ModelScript::physics_by_type(const std::string& type) const
{
    for (const auto& p : physics)
        if (p.type == type)
            return &p;
    return nullptr;
}

std::vector<const PhysicsFeature*> ModelScript::features(const std::string& type) const
{
    std::vector<const PhysicsFeature*> out;
    for (const auto& p : physics)
        for (const auto& f : p.features)
            if (f.type == type)
                out.push_back(&f);
    return out;
}

} // namespace hellofem::app
