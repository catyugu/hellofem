// hellofem::app — point expression evaluation (muparser backend)
// SPDX-License-Identifier: MIT
#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hellofem::app {

    /// A scalar expression evaluated at a point (x,y,z,t). Variables may be
    /// bound by name to a mutable value (parameter, field value, property).
    ///
    /// Unit literals like `20[mV]` are converted at parse time via
    /// `parse_si` when the value is a pure numeric-with-unit literal;
    /// unit postfixes inside larger expressions are not handled (COMSOL
    /// scripts reference named parameters for those).
    class Expression {
    public:
        Expression();
        ~Expression();

        Expression(const Expression&) = delete;
        Expression& operator=(const Expression&) = delete;
        Expression(Expression&&) noexcept;
        Expression& operator=(Expression&&) noexcept;

        /// Parse `text` and bind the named variables. `vars` maps a variable
        /// name to a double the expression reads live at evaluation time
        /// (the caller keeps the storage alive).
        void parse(std::string_view text, std::unordered_map<std::string, double*>& vars);

        /// Evaluate at the given point (also updates x/y/z/t bindings).
        double eval(double x, double y, double z, double t);

        /// Expression text (for diagnostics).
        const std::string& expr() const noexcept { return expr_; }

        /// Names of the variables the expression references.
        std::vector<std::string> variables() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        std::string expr_;
    };

} // namespace hellofem::app
